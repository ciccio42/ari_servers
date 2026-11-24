

from __future__ import annotations

import re
from dataclasses import dataclass, field

# --------------------------------------------------------------------------
# SERIALIZZAZIONE NEL MARKDOWN
# --------------------------------------------------------------------------

TABLE_OPEN_RE = re.compile(r"<!--\s*TABLE\s+(?P<attrs>[^>]*?)\s*-->")
TABLE_BLOCK_RE = re.compile(
    r"<!--\s*TABLE\s+(?P<attrs>[^>]*?)\s*-->\n(?P<body>.*?)\n<!--\s*ENDTABLE\s*-->",
    re.DOTALL,
)
_ATTR_RE = re.compile(r"(\w+)=(?:\"([^\"]*)\"|(\S+))")


def _esc(cell: str) -> str:
    """Rende una cella innocua dentro una riga di pipe."""
    return (
        str(cell)
        .replace("|", "\\|")
        .replace("\r", " ")
        .replace("\n", "<br>")
        .strip()
    )


def _unesc(cell: str) -> str:
    return cell.replace("<br>", "\n").replace("\\|", "|").strip()


def render_table_block(
    header: list[str], rows: list[list[str]], **attrs: object
) -> str:
    """Scrive una tabella come blocco markdown delimitato e auto-descrittivo."""
    width = len(header)
    attrs.setdefault("rows", len(rows))
    attrs.setdefault("cols", width)
    attr_str = " ".join(f'{k}="{v}"' for k, v in attrs.items() if v not in ("", None))

    lines = [f"<!-- TABLE {attr_str} -->"]
    lines.append("| " + " | ".join(_esc(h) for h in header) + " |")
    lines.append("| " + " | ".join("---" for _ in range(width)) + " |")
    for row in rows:
        padded = (list(row) + [""] * width)[:width]
        lines.append("| " + " | ".join(_esc(c) for c in padded) + " |")
    lines.append("<!-- ENDTABLE -->")
    return "\n".join(lines)


def parse_attrs(attr_str: str) -> dict[str, str]:
    return {m.group(1): (m.group(2) if m.group(2) is not None else m.group(3))
            for m in _ATTR_RE.finditer(attr_str)}


def parse_table_body(body: str) -> tuple[list[str], list[list[str]]]:
    """Ricostruisce (intestazione, righe) dal corpo di un blocco tabella."""
    rows: list[list[str]] = []
    for line in body.splitlines():
        line = line.strip()
        if not line.startswith("|"):
            continue
        if re.fullmatch(r"\|(?:\s*:?-{2,}:?\s*\|)+", line):
            continue  # riga separatore
        # Toglie esattamente una pipe per lato: strip("|") ne mangerebbe di
        # piu' quando una riga finisce con "||" per l'ultima colonna vuota,
        # disallineando tutte le celle rispetto all'intestazione.
        body_line = line[1:] if line.startswith("|") else line
        if body_line.endswith("|"):
            body_line = body_line[:-1]
        # Split sulle pipe non protette da backslash.
        cells = [_unesc(c) for c in re.split(r"(?<!\\)\|", body_line)]
        rows.append(cells)

    if not rows:
        return [], []
    header, data = rows[0], rows[1:]
    width = len(header)
    data = [(r + [""] * width)[:width] for r in data]
    return header, data


# --------------------------------------------------------------------------
# CLASSIFICAZIONE
# --------------------------------------------------------------------------


def classify_table(header: list[str], rows: list[list[str]]) -> str:
    """Assegna un tipo alla tabella per scegliere come testualizzarla."""
    norm = [h.strip().lower() for h in header]
    if {"nominativo", "stato"} <= set(norm):
        return "presence"
    if len(header) == 2:
        return "kv"
    # Molte celle lunghe: e' una tabella descrittiva (requisiti di accesso,
    # note di convalida), meglio tenerla leggibile riga per riga.
    cells = [c for r in rows for c in r if c]
    if cells and sum(len(c) for c in cells) / len(cells) > 60:
        return "records_long"
    return "records"


# --------------------------------------------------------------------------
# TESTUALIZZAZIONE
# --------------------------------------------------------------------------


@dataclass
class TableChunk:
    content: str                      # testo destinato all'embedding
    display: str                      # markdown da passare al generatore
    row_span: tuple[int, int]         # righe coperte (0-based, estremo escluso)
    kind: str
    extra: dict = field(default_factory=dict)


def _presence_sentence(row: list[str], context: str) -> str:
    idx, nome, gruppo, stato, nota = (row + [""] * 5)[:5]
    parts = [f"{nome}"]
    if gruppo:
        parts.append(f"({gruppo})")
    frase = f"{' '.join(parts)}: {stato.lower()}"
    if nota:
        frase += f" — {nota}"
    return frase


def textualize(
    header: list[str],
    rows: list[list[str]],
    kind: str,
    context: str,
    rows_per_chunk: int,
) -> list[TableChunk]:
    """Trasforma una tabella in chunk pronti per l'indicizzazione."""
    if not rows:
        return []

    out: list[TableChunk] = []
    sep = "| " + " | ".join("---" for _ in header) + " |"
    head_md = "| " + " | ".join(_esc(h) for h in header) + " |"

    for start in range(0, len(rows), rows_per_chunk):
        block = rows[start : start + rows_per_chunk]

        if kind == "presence":
            body = " ".join(_presence_sentence(r, context) + "." for r in block)
            text = f"Presenze registrate — {context}. {body}"
        elif kind == "kv":
            body = "; ".join(f"{r[0]}: {r[1]}" for r in block if any(r))
            text = f"{context}. {body}."
        else:
            frasi = []
            for r in block:
                campi = [
                    f"{h}: {v}"
                    for h, v in zip(header, r)
                    if v and h and not h.startswith("col_")
                ]
                if campi:
                    frasi.append("; ".join(campi))
            body = ". ".join(frasi)
            text = f"{context}. {body}."

        display = "\n".join(
            [head_md, sep]
            + ["| " + " | ".join(_esc(c) for c in r) + " |" for r in block]
        )
        out.append(
            TableChunk(
                content=re.sub(r"\s+", " ", text).strip(),
                display=display,
                row_span=(start, start + len(block)),
                kind=kind,
            )
        )
    return out
