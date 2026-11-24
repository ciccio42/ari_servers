# Copied unchanged from versione_finale/percorso_persona.py of the RAG repository high-level-unisa (ari_servers: rag/high_level_unisa).
# percorso_persona.py - versione finale (14/10/2026): era anagrafe_lookup.py (74860344). Cambia solo il
# percorso predefinito dell'anagrafe, ora dati/anagrafe.json (la pipeline gli passa comunque il suo).
# -*- coding: utf-8 -*-
"""Classificatore deterministico e ricerca nell'anagrafe.

Due funzioni, nessuna chiamata a un modello:
  - `classifica(domanda)` dice se la domanda riguarda una persona e ne estrae il nome;
  - `cerca(nome)` risolve il nome sull'anagrafe, con tre livelli: chiave esatta, sottoinsieme
    delle parole (la domanda dice «Rita Aquino», l'anagrafe ha «AQUINO Rita Patrizia»), e infine
    il solo cognome quando e' inequivocabile.

Il nome e' indicizzato come insieme ordinato di parole, quindi «PINTO Carmine» e «Carmine Pinto»
sono la stessa chiave.
"""
import json
import re
import unicodedata
from collections import defaultdict
from pathlib import Path

ANAGRAFE = Path(__file__).resolve().parent / "dati" / "anagrafe.json"

TITOLI = {"prof", "profssa", "dott", "dottssa", "ing", "sig", "sigra", "avv", "arch", "dr"}
# parole con l'iniziale maiuscola che NON sono nomi di persona
NON_NOMI = {
    "universita", "universita degli", "studi", "salerno", "ateneo", "amministrazione", "centrale",
    "consiglio", "amministrazione consiglio", "senato", "accademico", "nucleo", "valutazione",
    "dipartimento", "ufficio", "area", "scuola", "medica", "salernitana", "piano", "strategico",
    "codice", "etico", "comitato", "unico", "garanzia", "collegio", "direttori", "presidio",
    "qualita", "rettore", "prorettore", "direttore", "generale", "corso", "laurea", "facolta",
    "istituto", "superiore", "magistero", "real", "liceo", "regno", "italia", "napoli", "campus",
    "fisciano", "baronissi", "biblioteche", "risorse", "umane", "economiche", "finanziarie",
    "didattica", "ricerca", "terza", "missione", "statuto", "regolamento", "relazione", "delibera",
    "convenzione", "decreto", "legge", "prima", "quale", "quali", "quando", "come", "dove", "chi",
    "cosa", "qual", "quanti", "quante", "erasmus", "link", "sannio", "university",
}
# se una di queste parole compare in un candidato, quel candidato non e' una persona ma una
# struttura: «Ufficio Relazioni Internazionali», «Direzione Generale», «Capo Ufficio».
PAROLE_STRUTTURA = {
    "ufficio", "uffici", "direzione", "area", "aree", "consiglio", "senato", "nucleo",
    "dipartimento", "collegio", "comitato", "presidio", "settore", "centro", "servizio",
    "biblioteca", "segreteria", "capo", "coordinamento", "relazioni", "internazionalizzazione",
    "diritto", "studio", "bilancio", "contabilita", "finanza", "patrimonio", "economato",
    "rettorato", "amministrazione", "generale", "scuola", "facolta", "corso", "campus",
}
CONTATTO = re.compile(r"\b(email|e-mail|posta elettronica|telefono|numero|interno|matricola|"
                      r"contatt\w+|indirizzo|ufficio|stanza)\b", re.I)
RUOLO = re.compile(r"\b(chi è|chi e'|che ruolo|quale ruolo|ricopre|ricoprono|incaric\w+|"
                   r"delegat\w+|referent\w+|capo|responsabile|afferisce|dirige|qualifica|"
                   r"professore|professoressa|direttore|direttrice)\b", re.I)
NOME = re.compile(r"\b((?:(?:Prof|Prof\.ssa|Dott|Dott\.ssa|Ing|Sig|Sig\.ra|Avv)\.?\s+)?"
                  r"[A-ZÀ-Ú][a-zà-ú'’]{2,}(?:\s+(?:[A-ZÀ-Ú][a-zà-ú'’]{1,}|D'[A-ZÀ-Ú][a-zà-ú'’]+)){1,2})")


def sfoglia(s):
    s = unicodedata.normalize("NFD", (s or "").lower())
    return "".join(c for c in s if unicodedata.category(c) != "Mn")


def chiave(nome):
    parti = [sfoglia(p).strip(".'’") for p in re.split(r"[\s.]+", nome or "") if p.strip()]
    parti = [p for p in parti if p and p not in TITOLI and len(p) > 1]
    return " ".join(sorted(parti))


class Anagrafe:
    """Ricerca per nome nell'anagrafe (seconda versione, 28/09/2026).

    L'anagrafe ha una scheda per pagina della rubrica: gli OMONIMI sono schede distinte con la
    stessa chiave di nome (prima venivano fusi in una scheda che mescolava due persone). La ricerca
    per nome puo' quindi trovare piu' persone: si sceglie quella che la domanda descrive (ufficio,
    dipartimento, qualifica, incarichi); se la domanda non basta, si restituisce una scheda che le
    elenca tutte e lo dice. Mai fondere due persone. Per un nome con una sola scheda il
    comportamento e' quello di prima, livello per livello."""

    def __init__(self, percorso=ANAGRAFE):
        voci = json.loads(Path(percorso).read_text(encoding="utf-8"))
        self.per_chiave = defaultdict(list)
        for r in voci:
            self.per_chiave[r["chiave"]].append(r)
        self.persone = {r.get("id", r["chiave"]): r for r in voci}
        self.per_parola = defaultdict(set)
        for k in self.per_chiave:
            for p in k.split():
                self.per_parola[p].add(k)

    def cerca_tutti(self, nomi, domanda=None):
        """prova i candidati in ordine e restituisce il primo che si risolve"""
        if isinstance(nomi, str):
            nomi = [nomi]
        ultimo = "nessun nome"
        for n in nomi or []:
            rec, motivo = self.cerca(n, domanda)
            if rec is not None:
                return rec, motivo, n
            ultimo = motivo
        return None, ultimo, (nomi[0] if nomi else None)

    def cerca(self, nome, domanda=None):
        """(record, motivo) oppure (None, motivo). Mai indovinare: se e' ambiguo, si arrende."""
        k = chiave(nome)
        if not k:
            return None, "nessun nome"
        if k in self.per_chiave:
            return self._una(k, "chiave esatta", domanda)
        parole = set(k.split())
        cand = [c for c in set().union(*[self.per_parola.get(p, set()) for p in parole])
                if parole <= set(c.split())] if parole else []
        if len(cand) == 1:
            return self._una(cand[0], "sottoinsieme delle parole", domanda)
        if len(cand) > 1:
            return None, "ambiguo (%d omonimi)" % len(cand)
        rare = [p for p in parole if len(self.per_parola.get(p, ())) == 1]
        if len(rare) == 1:
            # (08/10/2026) un cognome da solo basta solo se le altre parole del candidato sono parole di nome di
            # qualcuno in anagrafe: «Protezione Dati», nel nome di un ufficio, non e' una persona anche se «Dati»
            # e' il cognome di una sola persona (nessuno si chiama «Protezione»)
            if all(p in self.per_parola for p in parole if p != rare[0]):
                return self._una(next(iter(self.per_parola[rare[0]])), "cognome inequivocabile", domanda)
            return None, "cognome inequivocabile, ma le altre parole non sono nomi"
        return None, "non trovato"

    def _una(self, k, motivo, domanda):
        cand = self.per_chiave[k]
        if len(cand) == 1:
            return cand[0], motivo
        return self._distingui(cand, domanda)

    @staticmethod
    def _descrizione(r):
        testo = " ".join([r.get("qualifica") or "", r.get("struttura") or ""]
                         + ["%s %s" % (x.get("organo", ""), x.get("ruolo", "")) for x in r.get("ruoli", [])])
        return set(re.findall(r"[a-z0-9]{3,}", sfoglia(testo))) - GENERICHE

    def _distingui(self, cand, domanda):
        """piu' schede con lo stesso nome: si sceglie quella che la domanda descrive meglio, se ce
        n'e' una sola; altrimenti una scheda che le contiene tutte"""
        parole = set(re.findall(r"[a-z0-9]{3,}", sfoglia(domanda or ""))) - set(cand[0]["chiave"].split())
        punti = sorted(((len(parole & self._descrizione(r)), n) for n, r in enumerate(cand)), reverse=True)
        if punti[0][0] > 0 and punti[0][0] > punti[1][0]:
            return cand[punti[0][1]], "omonimi: %d schede, scelta quella descritta dalla domanda" % len(cand)
        return ({"omonimi": cand, "chiave": cand[0]["chiave"], "nome_visualizzato": cand[0]["nome_visualizzato"],
                 "fonti": [f for r in cand for f in r.get("fonti", [])]},
                "omonimi: %d schede, la domanda non basta a distinguerle" % len(cand))


# parole troppo comuni per distinguere due omonimi: compaiono nelle descrizioni di quasi tutti
GENERICHE = {"della", "dell", "delle", "degli", "del", "dei", "con", "per", "nel", "nella", "sul",
             "universita", "studi", "salerno", "ateneo", "dipartimento", "ufficio", "area", "personale"}


def classifica(domanda, strutture=()):
    """('persona', nome) se la domanda riguarda una persona identificabile, altrimenti
    ('generale', None). Nessuna chiamata a un modello.
    strutture: i nomi delle strutture che la domanda nomina (dal registro delle strutture). Dal 08/10/2026 un
    candidato le cui parole stanno tutte nel nome di una di esse non e' una persona: «Adolfo Zambelli» nel
    Dipartimento di Chimica e Biologia «Adolfo Zambelli», «Protezione Dati» nell'Ufficio Sicurezza Informatica e
    Protezione Dati."""
    parole_strutture = [set(re.findall(r"[a-z0-9]+", sfoglia(s))) for s in strutture or () if s]
    nomi = []
    for m in NOME.findall(domanda):
        pulito = re.sub(r"^(?:Prof|Prof\.ssa|Dott|Dott\.ssa|Ing|Sig|Sig\.ra|Avv)\.?\s+", "", m).strip()
        k = chiave(pulito)
        if len(k.split()) < 2:
            continue
        if any(p in NON_NOMI for p in k.split()[:1]) or k in NON_NOMI:
            continue
        if all(p in NON_NOMI for p in k.split()):
            continue
        if any(p in PAROLE_STRUTTURA for p in k.split()):
            continue          # e' il nome di un ufficio o di un organo, non di una persona
        if any(set(k.split()) <= s for s in parole_strutture):
            continue          # fa parte del nome di una struttura che la domanda nomina
        nomi.append(pulito)
    if not nomi:
        return "generale", None
    if not (CONTATTO.search(domanda) or RUOLO.search(domanda)):
        return "generale", None
    return "persona", nomi


# La rubrica scrive le qualifiche al maschile per convenzione, anche per le donne. Quando il
# genere e' noto (marcatore della foto o nome proprio inequivocabile) i titoli vengono declinati;
# quando non lo e', il testo resta quello della fonte. Le forme invariabili - dirigente,
# referente, rappresentante, componente, presidente, responsabile, docente - non compaiono qui
# proprio perche' non vanno toccate.
FEMMINILI = [(r"Professore\b", "Professoressa"), (r"Ordinario\b", "Ordinaria"),
             (r"Associato\b", "Associata"), (r"Emerito\b", "Emerita"),
             (r"Onorario\b", "Onoraria"), (r"Ricercatore\b", "Ricercatrice"),
             (r"Direttore\b", "Direttrice"), (r"Vicedirettore\b", "Vicedirettrice"),
             (r"Prorettore\b", "Prorettrice"), (r"Delegato\b", "Delegata"),
             (r"Collaboratore\b", "Collaboratrice"), (r"Coordinatore\b", "Coordinatrice"),
             (r"Consigliere\b", "Consigliera"), (r"nominato\b", "nominata"),
             (r"eletto\b", "eletta"), (r"incaricato\b", "incaricata")]


def declina(testo, genere):
    """declina al femminile i titoli di `testo`; se il genere non e' noto non tocca niente"""
    if genere != "f" or not testo:
        return testo
    for masch, femm in FEMMINILI:
        testo = re.sub(masch, femm, testo)
    return testo


def _scheda_una(rec):
    """il testo che la pipeline persone passerebbe al modello"""
    g = rec.get("genere")
    righe = ["%s." % rec["nome_visualizzato"]]
    if rec.get("qualifica"):
        righe.append("Qualifica: %s%s." % (declina(rec["qualifica"], g),
                                           (" - %s" % rec["struttura"]) if rec.get("struttura") else ""))
    for r in rec.get("ruoli", []):
        righe.append("Incarico: %s (%s)." % (declina(r["ruolo"], g), r["organo"]))
    if rec.get("matricola"):
        righe.append("Matricola: %s." % rec["matricola"])
    for campo, etichetta in (("email", "Email"), ("interni", "Interno"), ("telefoni", "Telefono")):
        if rec.get(campo):
            righe.append("%s: %s." % (etichetta, ", ".join(rec[campo])))
    if rec.get("ufficio"):
        righe.append("Ufficio: %s." % rec["ufficio"])
    if rec.get("sito"):
        righe.append("Sito web: %s." % rec["sito"])
    return "\n".join(righe)


# varianti di genere: il riferimento scrive «Professoressa Emerita», l'anagrafe «Professore Emerito»
GENERE = [("professoressa", "professore"), ("emerita", "emerito"), ("onoraria", "onorario"),
          ("delegata", "delegato"), ("nominata", "nominato"), ("referente", "referente"),
          ("direttrice", "direttore"), ("dott.ssa", "dott"), ("prof.ssa", "prof"),
          ("consigliera", "consigliere"), ("presidente", "presidente")]


def neutro(s):
    s = sfoglia(s)
    for f, m in GENERE:
        s = s.replace(f, m)
    return s


# ------------------------------------------------------- quando fidarsi della sola scheda
# La scheda della rubrica contiene contatti, qualifica e afferenza; NON contiene gli incarichi
# che vivono su altre pagine (chi dirige un dipartimento, le pubblicazioni, la storia personale).
# Saltare il recupero e' quindi lecito solo se la domanda chiede un campo che quella specifica
# scheda contiene davvero. Misurato sulle 630 domande: sostituire il recupero anche per le
# domande sulle direzioni costava 17 risposte corrette.
CAMPI_VELOCI = [
    (re.compile(r"\b(e-?mail|posta elettronica|indirizzo di posta)\b", re.I), ("email",)),
    (re.compile(r"\b(interno|numero interno)\b", re.I), ("interni",)),
    (re.compile(r"\b(telefono|telefonico|recapito telefonico)\b", re.I), ("telefoni", "interni")),
    (re.compile(r"\bmatricola\b", re.I), ("matricola",)),
    (re.compile(r"\b(stanza|edificio|dove si trova|dove ha sede)\b", re.I), ("ufficio",)),
    (re.compile(r"\b(sito web|pagina personale|sito personale)\b", re.I), ("sito",)),
]

# argomenti che la rubrica non copre mai: se compaiono, il recupero serve comunque, anche se la
# domanda nomina pure un campo di contatto
FUORI_SCHEDA = re.compile(
    r"\b(pubblicazion\w+|citazion\w+|h-?index|indice h|scopus|web of science|"
    r"dirig\w+|direttor\w+|direzione|vicari\w+|presiede|coordina\w*|"
    r"orari\w*|ricevimento|insegna\w*|curriculum|biografia|nato|nata|nomina\w*)\b", re.I)


def _percorso_veloce_una(domanda, rec):
    """True se la domanda chiede un campo che questa scheda contiene davvero: solo in quel caso
    ha senso saltare del tutto il recupero. Nessuna chiamata a un modello."""
    d = domanda or ""
    if FUORI_SCHEDA.search(d):
        return False
    for rx, campi in CAMPI_VELOCI:
        if rx.search(d):
            return any(rec.get(c) for c in campi)
    return False


def scheda(rec):
    """il testo che la pipeline persone passa al modello. Per gli omonimi che la domanda non
    distingue, le schede di tutti, dichiarandolo: il modello deve sapere che le persone sono due"""
    if rec.get("omonimi"):
        righe = ["Nella rubrica di ateneo ci sono %d persone con questo nome e la domanda non basta a "
                 "distinguerle:" % len(rec["omonimi"])]
        for k, r in enumerate(rec["omonimi"], 1):
            righe.append("Persona %d. %s" % (k, _scheda_una(r)))
        return "\n".join(righe)
    return _scheda_una(rec)


def percorso_veloce(domanda, rec):
    """come prima; per piu' omonimi il recupero si salta solo se il campo chiesto c'e' per tutti"""
    if rec.get("omonimi"):
        return all(_percorso_veloce_una(domanda, r) for r in rec["omonimi"])
    return _percorso_veloce_una(domanda, rec)
