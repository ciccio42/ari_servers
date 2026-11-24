"""Structured registers computed over the whole archive, exposed to the model as tools.

Retrieval returns the k passages most similar to the question, so it cannot answer
questions whose answer is a set, a count or an ordering over many documents ("on which
dates was X present in 2025?", "how many meetings were held in the academic year
2023/24?"). Registers answer them deterministically:

  sessions    one entry per council meeting: date, minutes number, times, presence table
  attendance  one entry per person per meeting: status and note, from the presence tables
  mentions    the meetings whose minutes contain a name or subject, with the passages:
              the text of the minutes is kept in the registers and scanned in full at
              query time, instead of the top-k most similar passages

They are built once from the chunk files used for indexing, because the Qdrant payload
does not carry the table metadata (row spans, minutes number, times):

    python -m rag_pipeline.registers build <chunks_dir> --out registers.json
    python -m rag_pipeline.registers attendance "Auletta" --from 2025-01-01 --to 2025-12-31
    python -m rag_pipeline.registers sessions --from 2025-09-01 --to 2026-08-31

At query time a permissive regex gate (`needs_registers`) decides whether a question may
need a register; only then the model is asked, with the tool specs in TOOLS, whether to
call one and with which arguments. The tool output is plain text for the generator, in
Italian like the rest of the prompt.

A new register is added by writing its builder, a handler and a tool spec, and listing
the tool in TOOLS: the planner prompt does not change.
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import date
from difflib import get_close_matches
from pathlib import Path
from typing import Callable

REGISTERS_VERSION = 2
COUNCIL_DOC_TYPES = ("verbale_consiglio",)
STATUSES = ("Presente", "Assente giustificato", "Assente", "Non rilevato")
MAX_TOOL_CALLS = 3
# Homonyms reported in full; beyond this only their counts are listed.
MAX_FULL_PEOPLE = 3

# "VERBALE 07 - 10.06.25- prot. ..." : meeting date in the file name.
_FILE_DATE_RE = re.compile(r"(?<!\d)(\d{1,2})[./-](\d{1,2})[./-](\d{4}|\d{2})(?!\d)")


def _file_name_date(name: str) -> str:
    """First dd.mm.yyyy (or dd.mm.yy) date in a file name, as ISO; "" if none."""
    m = _FILE_DATE_RE.search(name)
    if not m:
        return ""
    year = int(m.group(3)) if len(m.group(3)) == 4 else 2000 + int(m.group(3))
    try:
        return date(year, int(m.group(2)), int(m.group(1))).isoformat()
    except ValueError:
        return ""


# --------------------------------------------------------------------------
# DATA
# --------------------------------------------------------------------------


@dataclass
class PresenceEntry:
    index: int          # row number in the minutes (1-based)
    name: str           # as written in the minutes
    key: str            # normalized, order-insensitive identity of the person
    group: str
    status: str         # one of STATUSES
    note: str


@dataclass
class Session:
    date: str                                       # ISO date, "" when unknown
    number: str                                     # minutes number, e.g. "1/2026"
    source_path: str
    source_file: str
    time_start: str = ""
    time_end: str = ""
    date_inconsistent: bool = False
    presence: list[PresenceEntry] | None = None     # None = no presence table found
    presence_complete: bool | None = None           # None = completeness unknown
    declared_present: int | None = None             # "Alle 10.40 sono presenti 26 persone"
    computed_present: int | None = None             # same count derived from the table
    # Text of the minutes outside the presence table, as {"path": headings, "text": ...}
    # in document order: searched by the `mentions` tool.
    passages: list[dict] = field(default_factory=list)

    def label(self) -> str:
        """Citation label, in the form the generator is told to use.

        Without a minutes number in the metadata the date identifies the meeting; the
        number in the file name is not used because it can be wrong (two 2025 files are
        both numbered 05).
        """
        if self.number:
            return f"Verbale {self.number} del {_it_date(self.date)}" if self.date else f"Verbale {self.number}"
        return f"Verbale del {_it_date(self.date)}" if self.date else self.source_file


class Registers:
    """Sessions with their attendance tables, plus lookups used by the tools."""

    def __init__(self, sessions: list[Session], built_from: str = ""):
        self.sessions = sorted(sessions, key=lambda s: (s.date or "9999", s.number))
        self.built_from = built_from
        self._people: dict[str, str] | None = None

    # ---- persistence -------------------------------------------------------

    def save(self, path: Path) -> None:
        data = {"version": REGISTERS_VERSION, "built_from": self.built_from,
                "sessions": [asdict(s) for s in self.sessions]}
        path.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "Registers":
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("version") != REGISTERS_VERSION:
            raise ValueError(f"{path}: registers version {data.get('version')}, "
                             f"expected {REGISTERS_VERSION}: rebuild them")
        sessions = []
        for s in data["sessions"]:
            presence = s.pop("presence")
            sessions.append(Session(**s, presence=None if presence is None
                                    else [PresenceEntry(**e) for e in presence]))
        return cls(sessions, data.get("built_from", ""))

    # ---- lookups -------------------------------------------------------------

    def in_period(self, d_from: date | None, d_to: date | None) -> list[Session]:
        """Dated sessions within [d_from, d_to] (either bound may be open), by date."""
        if d_from and d_to and d_from > d_to:
            d_from, d_to = d_to, d_from
        lo = d_from.isoformat() if d_from else ""
        hi = d_to.isoformat() if d_to else "9999-12-31"
        return [s for s in self.sessions if s.date and lo <= s.date <= hi]

    def people(self) -> dict[str, str]:
        """Person key -> most frequent spelling of the name in the minutes."""
        if self._people is None:
            spellings: dict[str, Counter] = defaultdict(Counter)
            for s in self.sessions:
                for e in s.presence or []:
                    spellings[e.key][e.name] += 1
            self._people = {k: c.most_common(1)[0][0] for k, c in spellings.items()}
        return self._people

    def date_range(self) -> tuple[str, str] | None:
        dates = [s.date for s in self.sessions if s.date]
        return (min(dates), max(dates)) if dates else None


# --------------------------------------------------------------------------
# NORMALIZATION
# --------------------------------------------------------------------------

# Honorifics dropped from names before matching ("prof.ssa" becomes "PROFSSA").
_TITLE_TOKENS = {"PROF", "PROFSSA", "PROFESSOR", "PROFESSORE", "PROFESSORESSA", "DOTT",
                 "DOTTSSA", "DOTTOR", "DOTTORE", "DOTTORESSA", "DR", "DRSSA", "ING",
                 "AVV", "SIG", "SIGRA", "SIGNOR", "SIGNORE", "SIGNORA"}


def name_tokens(text: str) -> list[str]:
    """Uppercase name tokens without accents, apostrophes, dots or honorifics.

    The minutes mix straight and typographic apostrophes (D'ANIELLO, D’ACIERNO) and
    upper/lower case first names (NASSA GIOVANNI): both sides of a match go through here.
    """
    s = unicodedata.normalize("NFKD", text)
    s = "".join(ch for ch in s if not unicodedata.combining(ch)).upper()
    s = re.sub(r"['’‘`.]", "", s)
    s = re.sub(r"[^A-Z\s]", " ", s)
    return [t for t in s.split() if t not in _TITLE_TOKENS]


def person_key(name: str) -> str:
    """Order-insensitive identity: "AULETTA Vincenzo" and "Vincenzo Auletta" coincide."""
    return " ".join(sorted(name_tokens(name)))


def normalize_status(text: str) -> str:
    t = text.strip().lower()
    if t.startswith("present"):
        return "Presente"
    if "giustific" in t:
        return "Assente giustificato"
    if t.startswith("assent"):
        return "Assente"
    return "Non rilevato"


def _it_date(iso: str) -> str:
    y, m, d = iso.split("-")
    return f"{d}/{m}/{y}"


def _parse_date(value) -> date | None:
    if not value:
        return None
    try:
        return date.fromisoformat(str(value).strip()[:10])
    except ValueError:
        return None


def _period_text(d_from: date | None, d_to: date | None) -> str:
    if d_from and d_to and d_from > d_to:
        d_from, d_to = d_to, d_from
    if d_from and d_to:
        if d_from == d_to:
            return f"seduta del {_it_date(d_from.isoformat())}"
        return f"dal {_it_date(d_from.isoformat())} al {_it_date(d_to.isoformat())}"
    if d_from:
        return f"dal {_it_date(d_from.isoformat())}"
    if d_to:
        return f"fino al {_it_date(d_to.isoformat())}"
    return "tutto l'archivio"


# --------------------------------------------------------------------------
# BUILD
# --------------------------------------------------------------------------

_DECLARED_RE = re.compile(
    r"\balle(?:\s+ore)?\s+(\d{1,2})[.:](\d{2})\s*,?\s+(?:sono|risultano)\s+presenti\s+(?:n\.\s*)?(\d+)",
    re.IGNORECASE)
_ENTRY_RE = re.compile(r"\b(?:entra|arriva)\w*\s+alle(?:\s+ore)?\s+(\d{1,2})[.:](\d{2})", re.IGNORECASE)
# The sentence that introduces the presence table in the minutes.
_COMPOSITION_RE = re.compile(r"risulta\s+(?:così|cosi'?|cos\u00ec)\s+compost[oa]|risulta\s+compost[oa]",
                             re.IGNORECASE)
_EXIT_RE = re.compile(r"\besce\w*\s+alle(?:\s+ore)?\s+(\d{1,2})[.:](\d{2})", re.IGNORECASE)
# Words the extractor sometimes glues to the name: the start of the note ("AULETTA
# Vincenzo Entra" with note "alle 10.29"), a remark in brackets, a status letter
# ("GRECO Luca P") or a word of the next heading ("SCIARRA Remigio Consiglio").
_NAME_PAREN_RE = re.compile(r"\s*\(([^)]*)\)?\s*$")
_NAME_NOTE_TAIL_RE = re.compile(r"\s+(entra|esce|arriva)$", re.IGNORECASE)
_NAME_JUNK_TAIL_RE = re.compile(r"\s+(?:P|A|G|AG|A\.\s?G|consiglio)\.?$", re.IGNORECASE)
_MAX_NAME_TOKENS = 6


def _clean_name(name: str, note: str) -> tuple[str, str]:
    """Name without the words glued to it, and the note completed with them."""
    lead = []
    if m := _NAME_PAREN_RE.search(name):
        lead.append(m.group(1).strip())
        name = name[:m.start()]
    if m := _NAME_NOTE_TAIL_RE.search(name):
        note = f"{m.group(1).capitalize()} {note}".strip()
        name = name[:m.start()]
    name = _NAME_JUNK_TAIL_RE.sub("", name).strip()
    note = "; ".join(p for p in (*lead, note) if p)
    return name, note


def _is_person_name(name: str) -> bool:
    """False for text of other tables read as a presence row ("3 CFU su logica ...")."""
    return not re.search(r"\d", name) and 2 <= len(name_tokens(name)) <= _MAX_NAME_TOKENS


def _parse_presence_display(display: str) -> list[PresenceEntry]:
    """Rows of a presence chunk, read from its markdown table (# | Nominativo | ...)."""
    from .tables import parse_table_body

    header, rows = parse_table_body(display)
    cols = {h.strip().lower(): i for i, h in enumerate(header)}
    if "nominativo" not in cols:
        return []

    def cell(row: list[str], name: str) -> str:
        i = cols.get(name)
        return row[i].strip() if i is not None and i < len(row) else ""

    out = []
    for row in rows:
        name, note = _clean_name(re.sub(r"\s+", " ", cell(row, "nominativo")),
                                 re.sub(r"\s+", " ", cell(row, "nota")))
        if not _is_person_name(name):
            continue
        idx = cell(row, "#")
        out.append(PresenceEntry(
            index=int(idx) if idx.isdigit() else 0,
            name=name,
            key=person_key(name),
            group=cell(row, "gruppo"),
            status=normalize_status(cell(row, "stato")),
            note=note,
        ))
    return out


def _present_at(entries: list[PresenceEntry], minutes: int) -> int:
    """People marked present who were in the room at the given time of day."""
    n = 0
    for e in entries:
        if e.status != "Presente":
            continue
        if (m := _ENTRY_RE.search(e.note)) and int(m.group(1)) * 60 + int(m.group(2)) > minutes:
            continue
        if (m := _EXIT_RE.search(e.note)) and int(m.group(1)) * 60 + int(m.group(2)) <= minutes:
            continue
        n += 1
    return n


# Late entries and early exits written in the body of the minutes rather than in the
# table: "Alle 11.40 entrano i proff. Moscato Francesco e Rundo Leonardo".
_TITLE_ABBR_RE = re.compile(
    r"\b(?:proff\.|prof\.?\s?ssa\.?|prof\.|sig\.?\s?(?:ra|na)\.?|sig\.|dott\.?\s?ssa\.?|dott\.|ing\.|dr\.)\s*",
    re.IGNORECASE)
_MOVEMENT_RE = re.compile(
    r"\balle(?:\s+ore)?\s+(\d{1,2})[.:,](\d{2})\s*,?\s*(entra|entrano|esce|escono)\s+"
    r"([^.;\n]{3,200}?)(?=\s+(?:ed?\s+)?(?:entra|esce|alle)\b|[.;\n]|$)",
    re.IGNORECASE)
# Words around the names in those sentences, dropped before matching.
_MOVEMENT_STOP = {"IL", "LA", "LO", "I", "LE", "GLI", "E", "ED", "PROF", "PROFF", "PROFSSA",
                  "STUDENTE", "STUDENTESSA", "STUDENTI", "RAPPRESENTANTE", "RAPPRESENTANTI",
                  "DEGLI", "DEL", "DELLA", "DI", "SIG", "SIGRA", "SIGNA", "DOTT", "DOTTSSA"}


def _add_movements_from_text(entries: list[PresenceEntry], texts: list[str]) -> tuple[int, int]:
    """Adds to the notes of present people the entries and exits written in the body.

    A note already present for the same movement is kept, and the status never changes.
    Returns (notes added, names not matched to exactly one present person).
    """
    added = unmatched = 0
    body = _TITLE_ABBR_RE.sub("", "\n".join(texts))
    for m in _MOVEMENT_RE.finditer(body):
        verb = "Entra" if m.group(3).lower().startswith("entra") else "Esce"
        names = re.split(r"\b(?:che|il quale|la quale|per|in quanto|dopo)\b", m.group(4))[0]
        for part in re.split(r"\s*,\s*|\s+e\s+", names):
            tokens = [t for t in name_tokens(part) if t not in _MOVEMENT_STOP]
            if not tokens:
                continue
            want = set(tokens[:2])
            hits = [e for e in entries if e.status == "Presente" and want <= set(e.key.split())]
            if len(hits) != 1:
                unmatched += 1
                continue
            entry = hits[0]
            pattern = _ENTRY_RE if verb == "Entra" else _EXIT_RE
            if pattern.search(entry.note):
                continue
            note = f"{verb} alle {int(m.group(1))}.{m.group(2)}"
            entry.note = f"{entry.note}; {note}" if entry.note else note
            added += 1
    return added, unmatched


def build_registers(chunks: list[dict], doc_types=COUNCIL_DOC_TYPES,
                    built_from: str = "") -> tuple[Registers, dict]:
    """Registers from the chunk dicts used for indexing, plus a quality report."""
    docs: dict[str, dict] = {}
    tables: dict[str, dict[str, dict]] = defaultdict(dict)
    texts: dict[str, list[str]] = defaultdict(list)
    passages: dict[str, list[dict]] = defaultdict(list)

    for c in chunks:
        m = c.get("metadata") or {}
        path = m.get("source_path", "")
        if not path or m.get("doc_type") not in doc_types:
            continue
        docs.setdefault(path, {
            "date": m.get("doc_date") or "",
            "number": str(m.get("verbale_num") or ""),
            "source_file": m.get("source_file", ""),
            "time_start": m.get("time_start") or "",
            "time_end": m.get("time_end") or "",
            "date_inconsistent": bool((m.get("debug") or {}).get("inconsistent_date")),
        })
        if m.get("chunk_type") == "table_presence":
            table = tables[path].setdefault(str(m.get("table_id") or "T1"),
                                            {"rows": {}, "spans": [], "total": None})
            span = m.get("table_row_span")
            if isinstance(span, (list, tuple)) and len(span) == 2:
                table["spans"].append((int(span[0]), int(span[1])))
            if m.get("table_rows_total"):
                table["total"] = int(m["table_rows_total"])
            # Keyed by number and person: the minutes themselves repeat a row number
            # (two "41" in the 2021 template), and both rows are real people.
            for e in _parse_presence_display(c.get("display") or ""):
                table["rows"][(e.index or len(table["rows"]) + 1, e.key)] = e
        else:
            text = c.get("raw_text") or c.get("content") or ""
            texts[path].append(text)
            passages[path].append({"path": " > ".join(m.get("heading_path") or []),
                                   "text": text})

    report: dict = {"documents": len(docs), "multiple_tables": [], "duplicates": [],
                    "incomplete": [], "check_mismatch": [], "check_ok": 0, "check_na": 0,
                    "date_conflicts": [], "unrecognized_table": [],
                    "movements_added": 0, "movements_unmatched": 0}
    sessions: list[Session] = []
    for path, d in docs.items():
        s = Session(source_path=path, **d, passages=passages[path])
        file_date = _file_name_date(s.source_file)
        if file_date and s.date and file_date != s.date:
            # A wrong document date moves the meeting into another period of every count.
            report["date_conflicts"].append(
                f"{s.source_file}: data del documento {_it_date(s.date)}, "
                f"nel nome del file {_it_date(file_date)}")
        if tables.get(path):
            if len(tables[path]) > 1:
                report["multiple_tables"].append(s.label())
            # The presence table is the largest one in the minutes.
            table = max(tables[path].values(), key=lambda t: len(t["rows"]))
            s.presence = [table["rows"][i] for i in sorted(table["rows"])]
            if table["total"] is not None:
                covered = {i for a, b in table["spans"] for i in range(a, b)}
                s.presence_complete = (covered == set(range(table["total"]))
                                       and len(s.presence) == table["total"])
            added, unmatched = _add_movements_from_text(s.presence, texts[path])
            report["movements_added"] += added
            report["movements_unmatched"] += unmatched
            declared = next((m for t in texts[path] if (m := _DECLARED_RE.search(t))), None)
            if declared:
                s.declared_present = int(declared.group(3))
                minutes = int(declared.group(1)) * 60 + int(declared.group(2))
                s.computed_present = _present_at(s.presence, minutes)
        elif any(_COMPOSITION_RE.search(t) for t in texts[path]):
            # The minutes announce the composition of the council but no presence table
            # was produced at ingestion: the table exists and the converter did not read it.
            report["unrecognized_table"].append(s.label())
        sessions.append(s)

    # The same minutes filed twice (e.g. in two folders) would be counted twice.
    best: dict[tuple[str, str], Session] = {}
    for s in sessions:
        key = (s.date, s.number or s.source_path)
        other = best.get(key)
        if other is None:
            best[key] = s
            continue
        report["duplicates"].append(s.label())

        def rank(x: Session) -> tuple:
            return (x.presence_complete is True, len(x.presence or []))

        if rank(s) > rank(other):
            best[key] = s
    sessions = list(best.values())

    statuses: Counter = Counter()
    for s in sessions:
        for e in s.presence or []:
            statuses[e.status] += 1
        if s.presence is not None and s.presence_complete is False:
            report["incomplete"].append(s.label())
        if s.declared_present is None:
            report["check_na"] += s.presence is not None
        elif s.declared_present == s.computed_present:
            report["check_ok"] += 1
        else:
            report["check_mismatch"].append(
                f"{s.label()}: dichiarati {s.declared_present}, dalla tabella {s.computed_present}")
    report.update({
        "sessions": len(sessions),
        "without_date": sum(1 for s in sessions if not s.date),
        "with_presence": sum(1 for s in sessions if s.presence is not None),
        "statuses": dict(statuses),
    })
    return Registers(sessions, built_from), report


def load_registers(path: Path) -> Registers | None:
    """Registers from disk, or None (with a message) when missing or unreadable."""
    if not path.exists():
        print(f"[registers] {path} not found: questions over many meetings will use "
              "retrieval only. Build it with: python -m rag_pipeline.registers build <chunks_dir>",
              file=sys.stderr)
        return None
    try:
        reg = Registers.load(path)
    except Exception as exc:
        print(f"[registers] {path} not loaded: {exc}", file=sys.stderr)
        return None
    print(f"[registers] {len(reg.sessions)} sessions loaded from {path}", file=sys.stderr)
    return reg


# --------------------------------------------------------------------------
# TOOL HANDLERS (text for the generator)
# --------------------------------------------------------------------------

_SCOPE = ("Calcolato su tutti i verbali del Consiglio presenti nell'archivio, "
          "non sui soli documenti recuperati.")


def _status_counts(key: str, sessions: list[Session]) -> dict:
    """Per-status sessions of one person, plus the sessions where they cannot be read."""
    out = {"by_status": defaultdict(list), "not_listed": [], "no_table": [], "incomplete": []}
    for s in sessions:
        if s.presence is None:
            out["no_table"].append(s)
            continue
        entry = next((e for e in s.presence if e.key == key), None)
        if entry is not None:
            out["by_status"][entry.status].append((s, entry.note))
        elif s.presence_complete is False:
            # The missing rows of the table may hold this person: unverifiable.
            out["incomplete"].append(s)
        else:
            out["not_listed"].append(s)
    return out


def _resolve_person(reg: Registers, person: str) -> tuple[list[str], list[str]]:
    """(matching person keys, suggestions when nothing matches)."""
    query = set(name_tokens(person))
    if not query:
        return [], []
    people = reg.people()
    matches = [k for k in people if query <= set(k.split())]
    exact = [k for k in matches if set(k.split()) == query]
    if len(exact) == 1:
        return exact, []
    if matches:
        return sorted(matches, key=lambda k: people[k]), []
    vocabulary = sorted({t for k in people for t in k.split()})
    near = {t2 for t in query for t2 in get_close_matches(t, vocabulary, n=3, cutoff=0.8)}
    suggestions = sorted({people[k] for k in people if near & set(k.split())})
    return [], suggestions[:5]


def _person_lines(name: str, c: dict) -> list[str]:
    """Attendance of one person: sessions per status, then what cannot be read for them."""
    lines = [f"Persona: {name}"]
    for status in STATUSES:
        items = c["by_status"].get(status, [])
        if not items and status == "Non rilevato":
            continue
        heading = "Stato non rilevato nel verbale" if status == "Non rilevato" else status
        lines.append(f"{heading}: {len(items)} {'seduta' if len(items) == 1 else 'sedute'}")
        lines.extend(f"- [{s.label()}]" + (f" ({note})" if note else "") for s, note in items)
    if c["not_listed"]:
        lines.append(f"Non compare nella tabella presenze: {len(c['not_listed'])} sedute "
                     "(probabilmente non faceva parte del Consiglio in quelle date)")
        lines.extend(f"- [{s.label()}]" for s in c["not_listed"])
    if c["incomplete"]:
        lines.append(f"Lacune per {name}: tabelle presenze incomplete in cui il nome non compare "
                     "(dato non verificabile): "
                     + ", ".join(f"[{s.label()}]" for s in c["incomplete"]) + ".")
    return lines


def attendance_report(reg: Registers, person: str, d_from: date | None,
                      d_to: date | None) -> tuple[str, str]:
    title = "Registro presenze (calcolato)"
    period = _period_text(d_from, d_to)
    keys, suggestions = _resolve_person(reg, person)
    if not keys:
        text = (f"Nelle tabelle presenze dei verbali del Consiglio non compare nessuna persona "
                f"corrispondente a «{person}».")
        if suggestions:
            text += f" Nomi simili presenti nelle tabelle: {', '.join(suggestions)}."
        return title, text

    sessions = reg.in_period(d_from, d_to)
    with_table = [s for s in sessions if s.presence is not None]
    people = reg.people()
    lines = [f"Registro presenze — {period}", _SCOPE,
             f"Sedute del Consiglio nel periodo: {len(sessions)} "
             f"(con tabella presenze: {len(with_table)}; senza: {len(sessions) - len(with_table)})."]
    if not sessions:
        span = reg.date_range()
        if span:
            lines.append(f"L'archivio contiene verbali del Consiglio dal {_it_date(span[0])} "
                         f"al {_it_date(span[1])}.")
        return title, "\n".join(lines)

    if len(keys) == 1:
        lines.extend(_person_lines(people[keys[0]], _status_counts(keys[0], sessions)))
    elif len(keys) <= MAX_FULL_PEOPLE:
        lines.append(f"«{person}» corrisponde a {len(keys)} persone nelle tabelle presenze: "
                     "i dati di ciascuna sono riportati qui sotto.")
        for k in keys:
            lines.append("")
            lines.extend(_person_lines(people[k], _status_counts(k, sessions)))
    else:
        lines.append(f"«{person}» corrisponde a {len(keys)} persone nelle tabelle presenze: "
                     "serve il nome per distinguerle. Conteggi di ciascuna:")
        for k in keys[:6]:
            c = _status_counts(k, sessions)
            counts = ", ".join(f"{st.lower()} {len(c['by_status'].get(st, []))}" for st in STATUSES)
            lines.append(f"- {people[k]}: {counts}")
        if len(keys) > 6:
            lines.append(f"- … e altre {len(keys) - 6} persone.")

    no_table = [s for s in sessions if s.presence is None]
    if no_table:
        lines.append("")
        lines.append("Lacune: sedute senza tabella presenze nell'archivio: "
                     + ", ".join(f"[{s.label()}]" for s in no_table) + ".")
    if len(keys) > 1:
        # Last line on purpose: the generator follows it more reliably than a prompt rule.
        names = ", ".join(people[k] for k in keys[:6])
        lines.append("")
        gaps = "prima la riga delle lacune, poi " if no_table else ""
        lines.append(f"Chiusura obbligatoria della risposta: {gaps}la domanda a quale persona "
                     f"ci si riferisce ({names}).")
    return title, "\n".join(lines)


def sessions_report(reg: Registers, d_from: date | None, d_to: date | None) -> tuple[str, str]:
    title = "Registro sedute (calcolato)"
    sessions = reg.in_period(d_from, d_to)
    lines = [f"Registro sedute del Consiglio — {_period_text(d_from, d_to)}", _SCOPE,
             f"Sedute nel periodo: {len(sessions)}"]
    for s in sessions:
        line = f"- [{s.label()}]"
        if s.time_start:
            line += f", ore {s.time_start}" + (f"–{s.time_end}" if s.time_end else "")
        if s.presence is None:
            line += "; tabella presenze non disponibile"
        else:
            present = sum(1 for e in s.presence if e.status == "Presente")
            line += f"; presenti in tabella: {present} su {len(s.presence)}"
            if s.presence_complete is False:
                line += " (tabella incompleta)"
        lines.append(line)
    undated = sum(1 for s in reg.sessions if not s.date)
    if undated and not (d_from or d_to):
        lines.append(f"Verbali senza data riconosciuta, esclusi dall'elenco: {undated}.")
    if not sessions and (span := reg.date_range()):
        lines.append(f"L'archivio contiene verbali del Consiglio dal {_it_date(span[0])} "
                     f"al {_it_date(span[1])}.")
    return title, "\n".join(lines)


# Per meeting and in total: a course named in twenty study plans of one meeting must
# not fill the context with twenty copies of the same row.
MENTION_PASSAGES_PER_SESSION = 3
MENTION_EXCERPT_CHARS = 500
MENTION_MAX_CHARS = 12000


def _fold(text: str) -> str:
    """Lowercase, no accents, words separated by single spaces, padded with spaces."""
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch)).lower()
    return " " + " ".join(re.findall(r"[a-z0-9]+", text)) + " "


def mention_terms(raw) -> list[str]:
    """Search terms from the tool arguments: a list or one string, folded, 3+ characters."""
    values = raw if isinstance(raw, list) else [raw]
    terms = []
    for v in values:
        folded = _fold(str(v or ""))
        if len(folded.strip()) >= 3 and folded not in terms:
            terms.append(folded)
    return terms


def _contains(folded: str, term: str) -> bool:
    """The term as a phrase, or all its words in the same passage.

    The minutes write names surname first ("MONTALTO Rodrigo Alberto") and tables
    split a course name across columns, so the exact phrase alone misses them.
    """
    if term in folded:
        return True
    words = term.split()
    return len(words) > 1 and all(f" {w} " in folded for w in words)


def _excerpt(text: str, term: str) -> str:
    """About MENTION_EXCERPT_CHARS characters of `text` around the first match of `term`."""
    first = term.split()[0]
    # The match is found on the folded text; the excerpt is cut from the original one
    # around the first word, so accents and capitals stay as written. A word written
    # with accents ("universita" for "università") is only found on the decomposed
    # text, whose offsets are close enough for an excerpt.
    m = (re.search(re.escape(first), text, re.IGNORECASE)
         or re.search(re.escape(first), unicodedata.normalize("NFKD", text).lower()))
    centre = m.start() if m else 0
    start = max(0, centre - MENTION_EXCERPT_CHARS // 3)
    end = min(len(text), start + MENTION_EXCERPT_CHARS)
    piece = " ".join(text[start:end].split())
    return ("…" if start else "") + piece + ("…" if end < len(text) else "")


def mentions_report(reg: Registers, terms: list[str], d_from: date | None,
                    d_to: date | None) -> tuple[str, str]:
    title = "Registro citazioni nei verbali (calcolato)"
    sessions = reg.in_period(d_from, d_to)
    shown = ", ".join(f"«{t.strip()}»" for t in terms)
    found: list[tuple[Session, list[tuple[str, str]]]] = []
    for s in sessions:
        hits = []
        for p in s.passages:
            folded = _fold(p["text"])
            term = next((t for t in terms if _contains(folded, t)), None)
            if term:
                hits.append((p["path"], _excerpt(p["text"], term)))
        if hits:
            found.append((s, hits))

    lines = [f"Ricerca nel testo dei verbali del Consiglio di: {shown} — {_period_text(d_from, d_to)}",
             _SCOPE,
             f"Sedute il cui verbale contiene il termine: {len(found)} su {len(sessions)} del periodo."]
    if found:
        lines.append("I passaggi servono a stabilire se la seduta riguarda davvero quanto chiesto: "
                     "cita solo le sedute pertinenti, ciascuna con il suo verbale.")
    used = sum(len(x) for x in lines)
    for s, hits in found:
        block = [f"- [{s.label()}]: {len(hits)} "
                 + ("passaggio" if len(hits) == 1 else "passaggi")]
        for path, excerpt in hits[:MENTION_PASSAGES_PER_SESSION]:
            block.append(f"  · {path + ': ' if path else ''}{excerpt}")
        if len(hits) > MENTION_PASSAGES_PER_SESSION:
            block.append(f"  (altri {len(hits) - MENTION_PASSAGES_PER_SESSION} passaggi con il "
                         "termine non riportati)")
        size = sum(len(x) for x in block)
        if used + size > MENTION_MAX_CHARS:
            # Every meeting stays listed, without excerpts once the budget is spent.
            block = block[:1]
            size = len(block[0])
        lines.extend(block)
        used += size
    if not sessions and (span := reg.date_range()):
        lines.append(f"L'archivio contiene verbali del Consiglio dal {_it_date(span[0])} "
                     f"al {_it_date(span[1])}.")
    return title, "\n".join(lines)


# --------------------------------------------------------------------------
# TOOLS
# --------------------------------------------------------------------------

_DATE_PROPS = {
    "date_from": {"type": "string",
                  "description": "Start of the period, YYYY-MM-DD. Empty if the question states no period."},
    "date_to": {"type": "string",
                "description": "End of the period, YYYY-MM-DD. Empty if the question states no period."},
}


@dataclass
class Tool:
    spec: dict
    run: Callable[[Registers, dict], tuple[str, str]]
    describe: Callable[[dict], str] = field(repr=False)
    # Arguments the tool cannot work with are dropped before running it.
    valid: Callable[[dict], bool] = field(default=lambda args: True, repr=False)


def _describe_period(args: dict) -> str:
    return f"{args.get('date_from') or '…'} → {args.get('date_to') or '…'}"


TOOLS: dict[str, Tool] = {
    "attendance": Tool(
        spec={"type": "function", "function": {
            "name": "attendance",
            "description": (
                "Attendance record of ONE person at the meetings (sedute) of the Teaching "
                "Council, computed over ALL the minutes in the archive. Use it for questions "
                "such as: on which dates was a person present or absent, how many meetings "
                "did a person attend or miss in a period, was a person present at the "
                "meeting of a given date."),
            "parameters": {"type": "object", "properties": {
                "person": {"type": "string",
                           "description": "Name of the person as written in the question, "
                                          "without titles (e.g. 'Auletta', 'Vincenzo Auletta')."},
                **_DATE_PROPS}, "required": ["person"]}}},
        run=lambda reg, a: attendance_report(reg, str(a.get("person") or ""),
                                             _parse_date(a.get("date_from")),
                                             _parse_date(a.get("date_to"))),
        describe=lambda a: f"attendance(person={a.get('person')!r}, {_describe_period(a)})",
        valid=lambda a: bool(name_tokens(str(a.get("person") or ""))),
    ),
    "sessions": Tool(
        spec={"type": "function", "function": {
            "name": "sessions",
            "description": (
                "List of the meetings (sedute) of the Teaching Council in a period, with "
                "date, minutes number, times and number of people present, computed over "
                "ALL the minutes in the archive. Use it for questions such as: how many "
                "meetings were held in a year, when did the council meet, which meetings "
                "fall in a period."),
            "parameters": {"type": "object", "properties": dict(_DATE_PROPS)}}},
        run=lambda reg, a: sessions_report(reg, _parse_date(a.get("date_from")),
                                           _parse_date(a.get("date_to"))),
        describe=lambda a: f"sessions({_describe_period(a)})",
    ),
    "mentions": Tool(
        spec={"type": "function", "function": {
            "name": "mentions",
            "description": (
                "Finds ALL the meetings (sedute) of the Teaching Council whose minutes mention "
                "a given subject, by searching the full text of ALL the minutes in the archive. "
                "Use it for questions such as: in which meetings was a course, a person, a "
                "student, a degree programme or a document discussed; when did the council "
                "deal with it; how many times was it mentioned. Not for attendance (use "
                "attendance) and not for what was decided in one given meeting."),
            "parameters": {"type": "object", "properties": {
                "terms": {"type": "array", "items": {"type": "string"},
                          "description": "The distinctive name of the subject as it would be "
                                         "written in the minutes (e.g. ['Quantum Computing'], "
                                         "['Rossi Mario'], ['tirocinio', 'tirocini']). Several "
                                         "terms are alternative spellings of the same subject. "
                                         "Never generic words such as 'sostituzione', "
                                         "'modifica', 'approvazione'."},
                **_DATE_PROPS}, "required": ["terms"]}}},
        run=lambda reg, a: mentions_report(reg, mention_terms(a.get("terms")),
                                           _parse_date(a.get("date_from")),
                                           _parse_date(a.get("date_to"))),
        describe=lambda a: (f"mentions(terms={[t.strip() for t in mention_terms(a.get('terms'))]}, "
                            f"{_describe_period(a)})"),
        valid=lambda a: bool(mention_terms(a.get("terms"))),
    ),
}


# --------------------------------------------------------------------------
# GATE AND PLANNER
# --------------------------------------------------------------------------

# Deliberately permissive: a false positive costs one short model call, a false
# negative sends a counting question to retrieval, which cannot answer it.
_PRESENCE_RE = re.compile(
    r"\b(present[ei]|presenz[ae]|assent[ei]|assenz[ae]|partecipat\w*|partecipazion\w*|"
    r"intervenut\w*)\b", re.IGNORECASE)
_MEETING_RE = re.compile(r"\b(sedut[ae]|riunion[ei]|adunanz[ae]|riunit[oai])\b", re.IGNORECASE)
_AGGREGATE_RE = re.compile(
    r"\b(quant[eiao]|numero|tutt[ei]|elenc\w*|in quali|quali sedut\w*|sempre|mai|ultim[ieao]|"
    r"period\w*|dal|tra il|fra il|nel(?:l')?\s*(?:anno|20\d\d)|frequen\w*|percentual\w*)\b",
    re.IGNORECASE)


_TOPIC_RE = re.compile(
    r"\b(parlat[oaie]|discuss\w*|trattat[oaie]|menzionat\w*|citat[oaie]|"
    r"affrontat[oaie]|esaminat[oaie]|deliberat[oaie])\b", re.IGNORECASE)


def needs_registers(question: str) -> bool:
    """True when the question may need a register (the model takes the final decision)."""
    return bool(_PRESENCE_RE.search(question)
                or ((_MEETING_RE.search(question) or _TOPIC_RE.search(question))
                    and _AGGREGATE_RE.search(question)))


PLANNER_PROMPT = """You decide whether a question about the archive of the Teaching Council of Computer Engineering (University of Salerno) needs one of the available tools.

Call a tool only when the answer requires counting, listing or comparing data across several council meetings or over a period of time, such as the attendance of a person or the number of meetings. If the answer is a passage of a document (a decision, a rule, a request, a single fact), do not call any tool and reply with an empty message.

Dates are YYYY-MM-DD. Today is {today}.
- A calendar year Y means Y-01-01 to Y-12-31.
- An academic year Y/Y+1 ("a.a. 2023/24", "2023/2024") means Y-09-01 to (Y+1)-08-31.
- A single meeting date D means D to D.
- Relative periods ("this year", "last year", "in the last six months") are computed from today.
- If the question states no period, leave both dates empty."""


@dataclass
class RegisterAnswer:
    title: str
    text: str
    calls: list[str]

    @property
    def key(self) -> str:
        return hashlib.md5("|".join(self.calls).encode("utf-8")).hexdigest()[:16]


class PlannerError(RuntimeError):
    """The planner call did not produce a usable response."""


# Models that answered "does not support tools": not asked again in this process.
_NO_TOOLS: set[str] = set()

_TEXT_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)


def _chat(cfg, payload: dict) -> dict:
    """POST /api/chat without streaming; drops "think" for models that reject it."""
    import httpx

    url = f"{cfg.base_url.rstrip('/')}/api/chat"
    with httpx.Client(timeout=cfg.timeout, trust_env=False) as client:
        for attempt in range(2):
            r = client.post(url, json=payload)
            if r.status_code == 200:
                return r.json()
            if r.status_code == 400 and "think" in payload and attempt == 0:
                payload.pop("think")
                continue
            raise PlannerError(f"HTTP {r.status_code}: {r.text.strip()[:200]}")
    raise PlannerError("no response")


def _arguments(raw) -> dict:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            value = json.loads(raw)
        except ValueError:
            return {}
        return value if isinstance(value, dict) else {}
    return {}


def _tool_calls(message: dict) -> list[tuple[str, dict]]:
    """Tool calls from Ollama's structured field, or from <tool_call> tags in the text."""
    calls = [((c.get("function") or {}).get("name") or "",
              _arguments((c.get("function") or {}).get("arguments")))
             for c in message.get("tool_calls") or []]
    if not calls:
        for m in _TEXT_CALL_RE.finditer(message.get("content") or ""):
            try:
                obj = json.loads(m.group(1))
            except ValueError:
                continue
            calls.append((obj.get("name") or "", _arguments(obj.get("arguments"))))
    return calls


def plan_and_run(cfg, reg: Registers, question: str) -> RegisterAnswer | None:
    """Ask the model which register tool to call, run it, return the text for the generator.

    `cfg` needs model, base_url, timeout and num_ctx (ConfigRisposta has them). Any
    failure returns None, so the question falls back to retrieval only.
    """
    calls = plan(cfg, question)
    return run_calls(reg, calls) if calls else None


def plan(cfg, question: str) -> list[tuple[str, dict]] | None:
    """Tool calls the model chooses for the question: [] = none, None = planner unavailable."""
    if cfg.model in _NO_TOOLS:
        return None
    payload = {
        "model": cfg.model,
        "messages": [
            {"role": "system", "content": PLANNER_PROMPT.format(today=date.today().isoformat())},
            {"role": "user", "content": question},
        ],
        "tools": [t.spec for t in TOOLS.values()],
        "stream": False,
        "think": False,
        "keep_alive": "2h",
        # Same num_ctx as the generator: a different value makes Ollama reload the model.
        "options": {"temperature": 0, "num_predict": 256, "num_ctx": cfg.num_ctx},
    }
    try:
        data = _chat(cfg, payload)
    except Exception as exc:
        if "support tools" in str(exc).lower():
            _NO_TOOLS.add(cfg.model)
            print(f"[registers] {cfg.model} does not support tool calling: registers disabled "
                  "for this model", file=sys.stderr)
        else:
            print(f"[registers] planner failed: {exc}", file=sys.stderr)
        return None
    return _tool_calls(data.get("message") or {})


def run_calls(reg: Registers, calls: list[tuple[str, dict]]) -> RegisterAnswer | None:
    """Run the known tool calls (duplicates and unknown names skipped) and join their text."""
    results: list[tuple[str, str, str]] = []
    seen: set[str] = set()
    for name, args in calls[:MAX_TOOL_CALLS]:
        tool = TOOLS.get(name)
        if tool is None or not tool.valid(args):
            continue
        description = tool.describe(args)
        if description in seen:
            continue
        seen.add(description)
        try:
            title, text = tool.run(reg, args)
        except Exception as exc:
            print(f"[registers] {description} failed: {exc}", file=sys.stderr)
            continue
        results.append((description, title, text))
    if not results:
        return None
    print(f"[registers] {'; '.join(r[0] for r in results)}", file=sys.stderr)
    title = results[0][1] if len(results) == 1 else "Registri dell'archivio (calcolati)"
    return RegisterAnswer(title=title, text="\n\n".join(r[2] for r in results),
                          calls=[r[0] for r in results])


# --------------------------------------------------------------------------
# DIAGNOSIS
# --------------------------------------------------------------------------


def _ranges(numbers: list[int]) -> str:
    """[3, 4, 5, 9] -> "3–5, 9"."""
    out, start = [], None
    for i, n in enumerate(numbers):
        if start is None:
            start = n
        if i + 1 == len(numbers) or numbers[i + 1] != n + 1:
            out.append(f"{start}–{n}" if n != start else f"{n}")
            start = None
    return ", ".join(out)
