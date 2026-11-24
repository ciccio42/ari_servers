"""Retrieval, context and generation of the department_office RAG.

From rag_pipeline/rispondi.py of the RAG repository (rag_unisa), unchanged but for: Assistente is in
../rag.py (DepartmentOfficeRag), the CLI is gone, the prompt is built at every request (today's date).
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Iterator

from .index import NOME_COLLEZIONE, _filtro, _id, interroga
from .registers import load_registers, needs_registers, plan_and_run

# A one-name answer ("Mario Rossi") does not say what it answers: read on its own, in a
# mail or in the history, it means nothing. It is also scored about 0 by RAGAS
# answer_relevancy, which rebuilds the question from the answer.
REGOLA_FRASE_COMPLETA = """- Rispondi con una frase completa che riprenda l'oggetto della domanda, anche quando la risposta e' un solo nome, una data o un numero: non "Mario Rossi", ma "Il relatore della tesi e' il prof. Mario Rossi". La risposta deve capirsi anche letta da sola, senza la domanda.
"""

ISTRUZIONI = """Sei l'assistente documentale del Consiglio Didattico di Ingegneria Informatica dell'Universita' di Salerno.

Rispondi alla domanda usando ESCLUSIVAMENTE i documenti riportati sotto.

COSA SCRIVERE
- Comincia dalla risposta, non da cosa hai consultato o da cosa manca.
{frase_completa}- Se la domanda chiede un elenco (chi, quali, quanti), riporta TUTTE le voci che trovi nei documenti, una per riga, con nome e matricola quando ci sono. Non fermarti alla prima.
- Riporta nomi, matricole, date, numeri di protocollo e voti esattamente come sono scritti. Non arrotondare, non correggere, non uniformare.
- Cita la fonte fra parentesi quadre subito dopo l'affermazione: [Verbale 11/2025 del 29/09/2025].
- Rispondi in italiano, in modo diretto e conciso.

I CORSI DI STUDIO
L'archivio e' del Consiglio Didattico, che governa piu' di un corso: Ingegneria Informatica triennale (L-8), Ingegneria Informatica magistrale (LM-32), Information Engineering for Digital Medicine (LM-32), Electrical Engineering for Digital Energy (LM-28), Ingegneria per la Salute.
- Rispondi su tutti, non solo su Ingegneria Informatica. Se una persona, una pratica o un insegnamento riguarda un altro corso, quella E' la risposta: riportala dicendo a quale corso appartiene.
- La distinzione conta solo per regolamenti, requisiti di accesso, piani di studio e offerta formativa, che sono diversi da corso a corso. Li' non rispondere con il documento di un altro corso: se hai trovato solo quello, dillo e indica a quale corso si riferisce.

I CONTEGGI
- I documenti qui sotto sono quelli recuperati per questa domanda, NON tutti quelli dell'archivio, che ne contiene migliaia. Non contarli e non presentarli come un totale.
- Se la domanda chiede "quanti documenti", "quante volte", "in quante sedute", "quanti studenti in tutto", non dare un numero come se fosse definitivo. Scrivi che fra i documenti recuperati ne risultano N, e che l'archivio puo' contenerne altri che non sono stati recuperati.
- Non elencare i documenti recuperati come se fossero la risposta a un conteggio.

IL REGISTRO CALCOLATO
- Un DOCUMENTO intitolato "Registro presenze (calcolato)", "Registro sedute (calcolato)", "Registro citazioni nei verbali (calcolato)" o "Registri dell'archivio (calcolati)" non e' un documento recuperato: e' un conteggio o una ricerca fatti su TUTTI i verbali del Consiglio presenti nell'archivio.
- Quando il registro risponde alla domanda, la risposta viene SOLO dal registro: non aggiungere informazioni prese dagli altri documenti, a meno che la domanda non chieda anche altro. Che una persona sia nominata in un verbale non dice nulla sulla sua presenza.
- Per le informazioni del registro non valgono le regole de "I CONTEGGI" e di "QUANDO L'INFORMAZIONE MANCA": riporta numeri ed elenchi di sedute come definitivi, senza ricalcolarli, e non aggiungere la riga "Nei documenti recuperati compaiono solo queste voci".
- Le assenze sono di due tipi: "Assente giustificato" e "Assente". Se la domanda chiede le assenze di una persona, o quante volte e' stata assente, senza dire di che tipo, contale entrambe: dai il totale e poi la suddivisione fra giustificate e non giustificate, con le sedute di ciascun tipo. Solo se la domanda chiede espressamente le assenze giustificate, o quelle non giustificate, riporta solo quel tipo.
- Non spiegare come e' fatto il registro e non commentare se i numeri sono definitivi: riportali e basta.
- Le sole incompletezze da segnalare sono quelle nelle righe del registro che cominciano con "Lacune": riportale in una riga in fondo, con le sedute interessate.
- Se il registro dice che un nome corrisponde a piu' persone, riporta i dati di ciascuna e chiudi chiedendo a quale ci si riferisce.
- Cita le singole sedute come le scrive il registro, per esempio [Verbale 1/2026 del 30/01/2026]. Il registro stesso non si cita: non scrivere mai [Registro presenze (calcolato)] ne' altri titoli di registro fra parentesi quadre.

COSA NON SCRIVERE
- Non elencare i documenti che hai scartato e non commentare la completezza dell'archivio.
- Non colmare i vuoti con quello che ti sembra plausibile.

QUANDO L'INFORMAZIONE MANCA
- Se non è specificata data alcuna e non la riesci a ricostruire dal contesto, assumi che si stra trattando della data odierna {oggi}.
- Prima di dire che manca, rileggi i documenti voce per voce: nei verbali la risposta e' spesso una sola riga dentro un elenco di pratiche studenti, sotto il nome della persona.
- Se una sola voce risponde, quella E' la risposta: riportala. Un elenco di un solo elemento non e' una risposta mancante.
- Se trovi una risposta parziale, dalla per intero e aggiungi in fondo una sola riga: "Nei documenti recuperati compaiono solo queste voci; potrebbero non essere tutte."
- Solo se nessun documento e' pertinente scrivi: "Nei documenti non trovo questa informazione".
- Non cominciare MAI con quella frase se poi nella risposta fai un nome, una data o un numero. Se hai qualcosa da dire, comincia da quello; l'incompletezza si segnala in fondo, non in apertura.
- Se i documenti si contraddicono, dillo invece di sceglierne uno."""


# --------------------------------------------------------------------------
# FILTRI RICAVATI DALLA DOMANDA
# --------------------------------------------------------------------------

_ANNO_RE = re.compile(r"\b(20[12]\d)\b")
_DATA_RE = re.compile(r"\b(\d{1,2})[./-](\d{1,2})[./-](20[12]\d|\d{2})\b")
# "a.a. 2023/2024" e "2025/26": l'anno accademico va da settembre ad agosto,
# quindi cade su DUE anni solari. Va distinto da "nel 2023", che ne indica uno.
_AA_RE = re.compile(r"\b(20[12]\d)\s*[/-]\s*(?:20)?(\d{2})\b")
# "4 febbraio 2025", "1° marzo 2024": the date of the minutes or of the attachment
# as people write it, and as the generated questions often do.
_MESI = ("gennaio", "febbraio", "marzo", "aprile", "maggio", "giugno", "luglio", "agosto",
         "settembre", "ottobre", "novembre", "dicembre")
_DATA_LETTERE_RE = re.compile(
    rf"\b(\d{{1,2}})\s*[°º]?\s+({'|'.join(_MESI)})\s+(20[12]\d)\b", re.IGNORECASE)


def _data_nella_domanda(testo: str) -> tuple[int, int, int] | None:
    """(year, month, day) of the first date in the question, numeric or in words."""
    if (m := _DATA_RE.search(testo)) is not None:
        a = m.group(3)
        return (int(a) if len(a) == 4 else 2000 + int(a)), int(m.group(2)), int(m.group(1))
    if (m := _DATA_LETTERE_RE.search(testo)) is not None:
        return int(m.group(3)), _MESI.index(m.group(2).lower()) + 1, int(m.group(1))
    return None

_TIPI = {
    "verbale_consiglio": r"\bverbal[ei]\b|\bsedut[ae]\b|\badunanz[ae]\b",
    "convocazione": r"\bconvocazion",
    "foglio_presenze": r"foglio presenz",
    "regolamento": r"\bregolament",
    "scheda_insegnamento": r"scheda d[ei][l ]|schede? insegnament|\bprogramma\b",
    "circolare": r"\bcircolar",
    "decreto": r"\bdecret|\bd\.?r\.?\b",
    "convenzione": r"\bconvenzion|\btirocin",
    "calendario_didattico": r"\bcalendari|sessione d[ei] esam",
}


@dataclass
class Filtri:
    year: int | None = None
    doc_type: str | None = None
    data: str | None = None      # ISO, quando la domanda porta una data intera
    anni: list[int] = field(default_factory=list)   # finestra di anni solari
    accademico: bool = False     # l'anno della domanda e' accademico, non solare

    def descrizione(self) -> str:
        parti = []
        if self.data:
            a, m, g = self.data.split("-")
            parti.append(f"seduta del {g}/{m}/{a}")
        elif self.accademico and self.year:
            parti.append(f"anno accademico {self.year}/{self.year + 1}")
        elif self.year:
            parti.append(f"anno {self.year}")
        if self.doc_type:
            parti.append(self.doc_type.replace("_", " "))
        return ", ".join(parti) if parti else "nessuno"


def estrai_filtri(domanda: str) -> Filtri:
    """Ricava i filtri dalla domanda.

    Deliberatamente prudente: un filtro sbagliato non riduce il rumore, esclude
    la risposta. Nel dubbio si preferisce cercare su tutto.
    """
    testo = domanda.lower()

    anno, data, anni, accademico = None, None, [], False
    giorno = _data_nella_domanda(testo)
    if (m := _AA_RE.search(testo)) is not None and giorno is None:
        # Un anno accademico copre due anni solari: le richieste per il
        # 2023/24 stanno nei verbali del 2023 e in quelli del 2024. Con la
        # finestra degli anni solari [Y-1, Y] restavano fuori quasi tutte —
        # misurato: 2 voci di corsi singoli invece di 10.
        anno, accademico = int(m.group(1)), True
        anni = [anno, anno + 1]
    elif giorno is not None:
        anno, me, g = giorno
        # Giorno e mese venivano scartati, e con loro l'unica cosa che
        # distingue una seduta dall'altra. "Il prof. Basile era presente alla
        # seduta del 04/02/2025?" restava filtrata sul solo 2025: il recupero
        # trovava le domande indicizzate su Basile in ALTRE sedute — quasi
        # identiche a parte la data, che nel vettore pesa pochissimo — e il
        # verbale giusto non compariva fra i primi dieci.
        try:
            data = date(anno, me, g).isoformat()
        except ValueError:
            data = None
        anni = [anno]
    elif (m := _ANNO_RE.search(testo)) is not None:
        anno = int(m.group(1))
        anni = [anno - 1, anno]

    tipo = next((t for t, pat in _TIPI.items() if re.search(pat, testo)), None)
    return Filtri(year=anno, doc_type=tipo, data=data, anni=anni, accademico=accademico)



# --------------------------------------------------------------------------
# RECUPERO
# --------------------------------------------------------------------------


@dataclass
class Fonte:
    chunk_id: str
    documento: str
    data: str
    percorso: list[str]
    testo: str
    punteggio: float
    via: str  # "testo", "domanda" oppure "sezione"
    domanda_corrisposta: str = ""
    file: str = ""   # source_path: serve a ritrovare le voci vicine
    genere: str = ""  # chunk_type: "table_presence" e simili

    def etichetta(self) -> str:
        giorno = ""
        if self.data:
            a, m, g = self.data.split("-")
            giorno = f" del {g}/{m}/{a}"
        return f"{self.documento}{giorno}"


def _condizione(models, filtri: Filtri):
    """Traduce i filtri in una condizione Qdrant, con l'anno preso a finestra.

    L'anno della domanda quasi mai coincide con la data della seduta. Verificato
    sul corpus: la richiesta di sostituire "Quantum Computing" e' nel Verbale 12
    del **22.10.2025**, protocollato il 23.02.**2026**; chiedendola "nel 2026"
    il filtro `year == 2026` lasciava 113 chunk su 14.632, e quel chunk non era
    fra quelli. Un anno accademico 2025/26 sta in documenti del 2025, e il
    numero di protocollo scivola all'anno dopo: la finestra e' [Y-1, Y].
    """
    condizioni = []
    if filtri.data:
        # Una seduta ha una data sola: se la domanda la nomina, e' il filtro
        # piu' preciso che si possa avere. Se poi la data del documento fosse
        # sbagliata (succede: vedi §8), meta' dei posti restano alla ricerca
        # libera e la risposta arriva comunque.
        condizioni.append(models.FieldCondition(
            key="doc_date", match=models.MatchValue(value=filtri.data)))
    elif filtri.anni:
        condizioni.append(models.FieldCondition(
            key="year", match=models.MatchAny(any=filtri.anni)))
    # With a full date the type is left out: the date alone picks one meeting, and the
    # type guessed from the question would drop its attachments ("il Regolamento del
    # 15 maggio 2023" is filed as an attachment of that meeting, not as a regolamento).
    if filtri.doc_type and not filtri.data:
        condizioni.append(models.FieldCondition(
            key="doc_type", match=models.MatchValue(value=filtri.doc_type)))
    return models.Filter(must=condizioni) if condizioni else None


# --------------------------------------------------------------------------
# VERSIONS OF REISSUED DOCUMENTS
# --------------------------------------------------------------------------

# Year and revision marks in a file name: "RegolamentoL8-IngegneriaInformatica - 25-26",
# "Calendario Sedute di Laurea 2023-24", "SMA2025 - L8-IIMD", "rev 20092023".
_ANNI_NEL_NOME_RE = re.compile(
    r"(?<!\d)(?:20)?\d{2}\s*[_/\-–]\s*(?:20)?\d{2}(?!\d)|20\d{2}(?!\d)|\bv\d+\b|\brev\b|_\d{8}\b")
# Attachment numbering changes from one year to the next: "ALL. 5.1", "Allegato 4.2.b".
_NUMERO_ALLEGATO_RE = re.compile(r"^\s*all(?:egato)?\.?\s*[\d.]+\s*[a-z]?\b\s*[-–_]?\s*")
# Minutes and calls are not versions of each other: each is its own meeting.
_SENZA_VERSIONI = {"verbale_consiglio", "convocazione"}
# Two documents of the same family closer than this are separate documents (two
# credit-recognition requests a week apart), not two editions.
_DISTANZA_VERSIONI = 180
# Short academic years ("del 24-25", "aa 22_23") and questions about earlier
# editions: in both cases the old version may be exactly what is asked for.
_AA_BREVE_RE = re.compile(r"(?<!\d)(\d{2})\s*[_/-]\s*(\d{2})(?!\d)")
_CONFRONTO_RE = re.compile(
    r"\bprecedent|\bvecch|\bin passato|\bnegli anni|\bevoluzion|\bconfront|\bstoric"
    r"|\brispetto a|\bdifferenz|\bprima versione|\bversion[ei]\b")


def _famiglia(nome_file: str) -> str:
    """The file name without attachment number, years, revision and copy marks.

    Separators are dropped as well: the same SUA is "SUACdS - L8-IngegneriaInformatica"
    one year and "SUA-CdS-L8-IngegneriaInformatica" the next.
    """
    k = nome_file.lower()
    for _ in range(2):   # "scheda.docx.pdf"
        k = re.sub(r"\.(pdf|docx?|xlsx?|md)$", "", k)
    k = _NUMERO_ALLEGATO_RE.sub("", k)
    k = re.sub(r"\(\d+\)", " ", k)
    k = _ANNI_NEL_NOME_RE.sub(" ", k)
    k = re.sub(r"\b(?:del|aa|a a)\b", " ", k)
    return re.sub(r"[\W_]+", "", k)


def versioni_superate(client, models) -> dict[str, str]:
    """source_path of each superseded edition -> source_path of the newest one.

    Regolamenti, piani di studio, SMA, schede and bandi are reissued every year
    with the same name and a different year: the archive holds the 2023, 2024 and
    2025 editions of "RegolamentoL8-IngegneriaInformatica" side by side.
    """

    documenti: dict[str, dict] = {}
    condizione = models.Filter(must=[
        models.FieldCondition(key="kind", match=models.MatchValue(value="chunk"))])
    offset = None
    while True:
        punti, offset = client.scroll(
            NOME_COLLEZIONE, scroll_filter=condizione, limit=2000, offset=offset,
            with_payload=["source_path", "source_file", "doc_date", "doc_type"],
            with_vectors=False)
        for p in punti:
            documenti.setdefault(p.payload.get("source_path", ""), p.payload)
        if offset is None:
            break

    famiglie: dict[str, list[tuple[date, str]]] = {}
    for percorso, d in documenti.items():
        if not percorso or d.get("doc_type") in _SENZA_VERSIONI:
            continue
        try:
            giorno = date.fromisoformat(d.get("doc_date") or "")
        except ValueError:
            continue
        famiglie.setdefault(_famiglia(d.get("source_file", "")), []).append((giorno, percorso))

    superate: dict[str, str] = {}
    for membri in famiglie.values():
        if len(membri) < 2:
            continue
        ultima_data, ultima = max(membri)
        for giorno, percorso in membri:
            if (ultima_data - giorno).days >= _DISTANZA_VERSIONI:
                superate[percorso] = ultima
    return superate


def chiede_un_periodo(domanda: str, filtri: Filtri) -> bool:
    """True when the question names a time or asks about earlier editions."""
    if filtri.year or filtri.data:
        return True
    testo = domanda.lower()
    if _CONFRONTO_RE.search(testo):
        return True
    return any(int(b) == (int(a) + 1) % 100 for a, b in _AA_BREVE_RE.findall(testo))


def recupera(client, models, emb, k: int, modo: str, filtri: Filtri,
             condizione=None) -> list[Fonte]:
    """Recupera i chunk, risolvendo le domande al documento d'origine.

    Le domande indicizzate non portano il testo: quando una corrisponde si
    recupera il chunk a cui appartiene, che e' cio' che va letto e passato al
    modello. Si tiene traccia di quale domanda ha fatto scattare la
    corrispondenza, perche' spiega perche' quel documento e' stato scelto.
    """
    f = condizione if condizione is not None else _condizione(models, filtri)

    # Si chiedono piu' risultati del necessario: dopo la deduplica per chunk
    # ne restano meno, perche' piu' domande possono puntare allo stesso testo.
    punti = interroga(client, models, emb, k * 4, modo, f)

    ordine = []
    visti: set[str] = set()
    for p in punti:
        cid = p.payload["chunk_id"]
        if cid in visti:
            continue
        visti.add(cid)
        ordine.append(p)

    # Le domande corrispondenti si risolvono tutte in una chiamata sola. Prima
    # ne partiva una per ciascuna: con k*4 candidati erano fino a ventiquattro
    # andate e ritorno per una singola ricerca, tutte in serie.
    da_domanda = [p.payload["chunk_id"] for p in ordine
                  if p.payload.get("kind") == "question"]
    originali: dict[str, dict] = {}
    if da_domanda:
        trovati = client.retrieve(NOME_COLLEZIONE,
                                  ids=[_id("chunk", c) for c in da_domanda],
                                  with_payload=True)
        originali = {t.payload["chunk_id"]: t.payload for t in trovati}

    fonti: list[Fonte] = []
    for p in ordine:
        cid = p.payload["chunk_id"]
        origine, via, domanda = p.payload, "testo", ""
        if p.payload.get("kind") == "question":
            origine = originali.get(cid)
            if origine is None:
                continue
            via = "domanda"
            domanda = p.payload.get("question", "")

        fonti.append(Fonte(
            chunk_id=cid,
            documento=origine.get("source_file", ""),
            data=origine.get("doc_date", ""),
            percorso=origine.get("heading_path") or [],
            testo=origine.get("display") or origine.get("content") or "",
            punteggio=p.score,
            via=via,
            domanda_corrisposta=domanda,
            file=origine.get("source_path", ""),
            genere=origine.get("chunk_type", ""),
        ))
        if len(fonti) >= k:
            break
    return fonti


def _prefisso(percorso: list[str], punti: list) -> list[str]:
    """Da dove prendere le voci vicine: i figli se ce ne sono, altrimenti i fratelli.

    Una voce foglia — "2 Pratiche studenti > CAMBIO DI CURRICULUM > CUPO
    Sabatino Pio" — non ha figli: le vicine sono i fratelli, cioe' le altre
    voci sotto lo stesso titolo.

    Una testa di sezione no. "3 Pratiche studenti > ISCRIZIONE CORSI SINGOLI"
    contiene **solo** la frase di apertura ("Il Presidente comunica che sono
    pervenute le seguenti richieste:") e i nomi stanno nei suoi figli. E' anche
    il chunk che la ricerca trova per primo, perche' e' l'unico che nomina
    l'argomento. Prendendo i fratelli si finiva altrove: il padre e' "3 Pratiche
    studenti", che copre immatricolazioni, passaggi e riconoscimenti — centinaia
    di voci che con la domanda non c'entrano. Chiesto l'elenco degli iscritti ai
    corsi singoli, il modello riceveva la frase di apertura e nessun nome, e
    rispondeva — correttamente, rispetto a cio' che aveva — che i nomi non
    c'erano.
    """
    proprio = list(percorso)
    for punto in punti:
        strada = list(punto.payload.get("heading_path") or [])
        if len(strada) > len(proprio) and strada[:len(proprio)] == proprio:
            return proprio
    return proprio[:-1]


def _espandi(client, models, fonti: list[Fonte], quanti: int) -> list[Fonte]:
    """Aggiunge le voci vicine: i chunk sotto lo stesso titolo di livello sopra.

    Serve alle domande che chiedono un elenco, ed e' il difetto che si vede
    peggio nelle risposte. Nei verbali il percorso dei titoli scende fino alla
    singola voce — "2 Pratiche studenti > PASSAGGIO DI CORSO > AVALLONE Marco"
    — perche' cosi' le domande generate restano omogenee (vedi `_section_key`).
    La somiglianza pero' recupera la voce che assomiglia di piu' alla domanda,
    e le altre venti restano fuori: a "chi ha chiesto il passaggio a Systems"
    si risponde con un nome solo, e la risposta sembra completa mentre non lo e'.

    Si sale di un livello, non a tutto il documento: "2 Pratiche studenti" per
    intero sono dieci pagine, e riempirebbe il contesto di voci che parlano
    d'altro. `quanti` e' il tetto complessivo, perche' una sezione con venti
    studenti non deve mangiarsi tutta la finestra.
    """
    if quanti <= 0 or not fonti:
        return fonti

    # In ordine di punteggio, non alfabetico. Con `sorted()` la selezione dei
    # documenti da scansionare diventava arbitraria: i percorsi cominciano con
    # l'anno, quindi un 2021 precedeva sempre un 2025 e il verbale che
    # conteneva davvero la risposta restava fuori dai primi quattro.
    files: list[str] = []
    for f in fonti:
        if f.file and f.percorso and f.file not in files:
            files.append(f.file)
    if not files:
        return fonti

    # Uno scroll per documento invece di uno solo con `should`: in locale ogni
    # scroll e' comunque una scansione completa, e cosi' non si dipende da come
    # la modalita' locale interpreta un `should` annidato. Al massimo quattro
    # documenti, altrimenti si pagano scansioni per contesto che verra' tagliato.
    per_file: dict[str, list] = {}
    for percorso in files[:4]:
        condizione = models.Filter(must=[
            models.FieldCondition(key="kind", match=models.MatchValue(value="chunk")),
            models.FieldCondition(key="source_path",
                                  match=models.MatchValue(value=percorso)),
        ])
        try:
            punti, _ = client.scroll(NOME_COLLEZIONE, scroll_filter=condizione,
                                     limit=1000, with_payload=True)
        except Exception:
            return fonti   # l'espansione e' un di piu': non deve far fallire una risposta
        per_file[percorso] = list(punti)
    # Senza la posizione nel payload l'ordine si approssima con la pagina: e'
    # comunque meglio dell'ordine in cui li restituisce la scansione.
    for elenco in per_file.values():
        elenco.sort(key=lambda x: (x.payload.get("page", 0),
                                   " > ".join(x.payload.get("heading_path") or [])))

    visti = {f.chunk_id for f in fonti}
    vicine: dict[str, list[Fonte]] = {}
    rimasti = quanti
    # A giri, non tutto alla prima fonte. Senza, la prima fonte espandibile si
    # prendeva il budget intero: verificato sul campo, il secondo documento
    # recuperato — che con la domanda non c'entrava nulla — si e' portato
    # dietro sei voci di "Tutorato" e "Monitoraggio insegnamenti" mangiandosi
    # meta' del contesto. Coi giri, se le fonti espandibili sono poche il resto
    # del budget torna comunque a loro.
    per_giro = max(3, quanti // 3)
    candidati = [f for f in fonti if f.file and f.percorso]
    while rimasti > 0:
        prima_del_giro = rimasti
        for f in candidati:
            if rimasti <= 0:
                break
            genitore = _prefisso(f.percorso, per_file.get(f.file, []))
            if not genitore:
                continue
            messe = 0
            for punto in per_file.get(f.file, []):
                if rimasti <= 0 or messe >= per_giro:
                    break
                cid = punto.payload.get("chunk_id", "")
                if not cid or cid in visti:
                    continue
                strada = list(punto.payload.get("heading_path") or [])
                if strada[:len(genitore)] != genitore:
                    continue
                visti.add(cid)
                rimasti -= 1
                messe += 1
                vicine.setdefault(f.chunk_id, []).append(Fonte(
                    chunk_id=cid,
                    documento=punto.payload.get("source_file", ""),
                    data=punto.payload.get("doc_date", ""),
                    percorso=strada,
                    testo=punto.payload.get("display") or punto.payload.get("content") or "",
                    punteggio=f.punteggio,
                    via="sezione",
                    file=punto.payload.get("source_path", ""),
                    genere=punto.payload.get("chunk_type", ""),
                ))
        if rimasti == prima_del_giro:
            break        # nessuna fonte ha piu' fratelli da dare

    # Ogni vicina subito dopo la voce che l'ha portata: se il contesto viene
    # tagliato, si taglia in fondo e non in mezzo a un elenco.
    fuse: list[Fonte] = []
    for f in fonti:
        fuse.append(f)
        fuse.extend(vicine.get(f.chunk_id, []))
    return fuse


def costruisci_contesto(fonti: list[Fonte], tetto_caratteri: int) -> str:
    blocchi = []
    speso = 0
    for i, f in enumerate(fonti, 1):
        percorso = " > ".join(f.percorso)
        testa = f"--- DOCUMENTO {i}: {f.etichetta()}"
        if percorso:
            testa += f"\n    sezione: {percorso}"
        blocco = f"{testa}\n{f.testo}"
        if speso + len(blocco) > tetto_caratteri and blocchi:
            break
        blocchi.append(blocco)
        speso += len(blocco)
    return "\n\n".join(blocchi)


# --------------------------------------------------------------------------
# GENERAZIONE
# --------------------------------------------------------------------------


@dataclass
class ConfigRisposta:
    db: Path = Path("./qdrant")
    model: str = "gemma4:12b"
    base_url: str = "http://127.0.0.1:11500"
    emb_model: str = "BAAI/bge-m3"
    device: str = "cuda"
    k: int = 14
    modo: str = "ibrido"
    # Voci vicine aggiunte al contesto (vedi `_espandi`). 0 = disattivata.
    # Dieci e non sei: una sezione di pratiche studenti ha spesso quattro o piu'
    # voci, e il tetto dei caratteri taglia comunque quello che non entra.
    espansione: int = 10
    # 14 + 10 blocchi da ~850 caratteri sono ~20.000: il tetto a 30.000 lascia
    # margine ai blocchi grandi (le tabelle arrivano a 4.400) senza che il
    # taglio scatti sul primo documento un po' lungo.
    contesto_max: int = 30000
    # 30.000 caratteri stanno in 9.000-11.400 token (misurato: fra 2,64 e 3,31
    # caratteri per token secondo il tipo di chunk); con istruzioni e 800 token
    # di risposta il caso peggiore e' ~12.700, che sforerebbe 12.288.
    num_ctx: int = 16384
    max_tokens: int = 800
    temperature: float = 0.1
    timeout: float = 300.0
    # Registers built by `python -m rag_pipeline.registers build`; None = registers.json
    # in the folder that contains the DB (the project root for ./qdrant and qdrant_eval).
    registers_path: Path | None = None
    use_registers: bool = True
    # Without a time in the question, older editions of a reissued document give way
    # to the newest one (see `versioni_superate`).
    ultime_versioni: bool = True
    # Cross-encoder that reorders the candidates (see rerank.py); "" = off. With it on,
    # `candidati` passages are retrieved and the best `k` of them kept. Measured on
    # dataset_eval_v4: reference passage in the prompt 117 -> 127 of 148, MRR 0.46 -> 0.71,
    # about 2 s more per question.
    riordino: str = "BAAI/bge-reranker-v2-m3"
    candidati: int = 40
    # Answers as a complete sentence that restates what is asked (REGOLA_FRASE_COMPLETA).
    risposte_complete: bool = True


class ErroreGenerazione(RuntimeError):
    """Il server non ha generato: il messaggio e' gia' pronto da leggere.

    Serve perche' l'errore piu' frequente — il modello non e' su quel server —
    arriva come un 404 sull'endpoint. Lasciato a `raise_for_status()` diventa
    "404 Not Found for url .../api/chat", che indica l'endpoint mentre il
    difetto e' nel nome del modello: la diagnosi punta dalla parte opposta.
    """


_COME_AVVIARE = """in ari_servers il server Ollama si avvia con:  bash ollama/docker/run.bash start"""


def _indirizzo(base_url: str) -> str:
    """Da "http://127.0.0.1:11500" a "127.0.0.1:11500", forma di OLLAMA_HOST."""
    return base_url.rstrip("/").split("://", 1)[-1]


def modelli_disponibili(base_url: str, timeout: float = 10.0) -> list[str] | None:
    """I modelli presenti sul server. `None` se il server non risponde.

    La distinzione fra lista vuota e `None` e' quella fra "server Ollama senza
    modelli scaricati" e "sulla porta non c'e' Ollama": due rimedi diversi.
    """
    import httpx

    try:
        r = httpx.get(f"{base_url.rstrip('/')}/api/tags", timeout=timeout)
        r.raise_for_status()
        modelli = r.json().get("models", [])
    except Exception:
        return None
    return sorted(m.get("name", "") for m in modelli if m.get("name"))


_NON_CHIESTO = object()


def verifica_modello(cfg: ConfigRisposta, disponibili=_NON_CHIESTO) -> str:
    """Avviso da mostrare prima di accettare domande, "" se e' tutto a posto.

    Un controllo all'avvio costa una chiamata e toglie di mezzo l'unico modo
    in cui l'interfaccia puo' sembrare pronta e non esserlo. `disponibili` si
    passa quando l'elenco e' gia' stato chiesto: `None` e lista vuota sono due
    esiti distinti, quindi non basterebbe come valore di default.
    """
    if disponibili is _NON_CHIESTO:
        disponibili = modelli_disponibili(cfg.base_url)
    if disponibili is None:
        return (f"da {cfg.base_url} non arriva un elenco di modelli: il server "
                f"Ollama non e' avviato, ascolta su un'altra porta, oppure sulla "
                f"porta c'e' un altro servizio.\n{_COME_AVVIARE}")
    if cfg.model in disponibili or f"{cfg.model}:latest" in disponibili:
        return ""
    return _modello_mancante(cfg, disponibili)


def _modello_mancante(cfg: ConfigRisposta, disponibili: list[str]) -> str:
    elenco = ", ".join(disponibili) if disponibili else "(nessuno scaricato)"
    return (f"il modello «{cfg.model}» non e' su {cfg.base_url}.\n"
            f"presenti: {elenco}\n"
            f"per scaricarlo (da ari_servers):  bash ollama/docker/run.bash pull {cfg.model}")


def _spiega(stato: int, dettaglio: str, cfg: ConfigRisposta) -> str:
    """Traduce lo stato HTTP in una causa e in un rimedio.

    Sul 404 la discriminante non e' il testo dell'errore — Ollama risponde
    "404 page not found" sia per l'endpoint sconosciuto sia, in altre versioni,
    per il modello — ma se `/api/tags` funziona: se funziona e' Ollama, e
    allora manca il modello.
    """
    if stato == 404:
        disponibili = modelli_disponibili(cfg.base_url)
        if disponibili is None:
            return (f"{cfg.base_url} risponde ma non serve /api/chat: sulla porta "
                    f"c'e' un altro servizio, o una versione di Ollama troppo "
                    f"vecchia.\nverifica con:  "
                    f"curl -s {cfg.base_url.rstrip('/')}/api/tags")
        return f"{_modello_mancante(cfg, disponibili)}\n(il server ha risposto: {dettaglio})"
    if stato in (401, 403):
        return (f"{cfg.base_url} rifiuta la richiesta ({stato}): probabilmente "
                f"non e' il server proprio ma un servizio che richiede "
                f"autenticazione.\n(risposta: {dettaglio})")
    return f"il server ha risposto {stato} a {cfg.base_url}/api/chat: {dettaglio}"


def _dettaglio(risposta) -> str:
    """Il messaggio del server, che con `stream=True` va letto esplicitamente."""
    try:
        risposta.read()
    except Exception:
        return "<corpo illeggibile>"
    testo = risposta.text.strip()
    try:
        import json as _json

        return str(_json.loads(testo).get("error", testo))[:300]
    except Exception:
        return testo[:300] or "<nessun messaggio>"


def genera(cfg: ConfigRisposta, domanda: str, contesto: str) -> Iterator[str]:
    """Genera la risposta in flusso.

    In flusso perche' su questo hardware la generazione va da 5 a 27 token al
    secondo a seconda del modello: aspettare la risposta intera prima di
    mostrarne una riga rende l'attesa molto piu' pesante di quanto sia.
    """
    import httpx
    import json as _json

    payload = {
        "model": cfg.model,
        "messages": [
            {"role": "system", "content": ISTRUZIONI.replace(
                "{frase_completa}", REGOLA_FRASE_COMPLETA if cfg.risposte_complete else "").replace(
                "{oggi}", date.today().strftime("%d/%m/%Y"))},
            {"role": "user", "content": f"DOCUMENTI RECUPERATI\n\n{contesto}\n\nDOMANDA: {domanda}"},
        ],
        "stream": True,
        "think": False,
        "keep_alive": "2h",
        "options": {
            "temperature": cfg.temperature,
            "num_predict": cfg.max_tokens,
            "num_ctx": cfg.num_ctx,
        },
    }
    url = f"{cfg.base_url.rstrip('/')}/api/chat"

    with httpx.Client(timeout=cfg.timeout) as c:
        # Due giri al massimo: il secondo serve solo a togliere "think".
        for _ in range(2):
            try:
                with c.stream("POST", url, json=payload) as r:
                    if r.status_code == 200:
                        yield from _senza_citazioni_registro(_leggi_flusso(r, _json))
                        return
                    stato, dettaglio = r.status_code, _dettaglio(r)
            except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
                raise ErroreGenerazione(
                    f"non riesco a contattare {url}: {exc}\n{_COME_AVVIARE}") from exc

            # Il modello non accetta "think": si riprova senza.
            if stato == 400 and "think" in payload:
                payload.pop("think")
                continue
            raise ErroreGenerazione(_spiega(stato, dettaglio, cfg))


def riscalda(cfg: ConfigRisposta) -> str:
    """Carica il modello in memoria. Torna un resoconto da stampare.

    Un 35B a q8 sono decine di GB da leggere da disco: e' il grosso dei 60
    secondi della prima domanda. Il caricamento non si puo' evitare, ma si puo'
    spostare — qui, all'avvio, mentre nessuno sta ancora aspettando. Con
    `keep_alive` a 2h le domande successive lo trovano gia' pronto.
    """
    import time

    import httpx

    corpo = {
        "model": cfg.model,
        "messages": [{"role": "user", "content": "ok"}],
        "stream": False,
        "keep_alive": "2h",
        "options": {"num_predict": 1, "num_ctx": cfg.num_ctx},
    }
    t0 = time.time()
    try:
        r = httpx.post(f"{cfg.base_url.rstrip('/')}/api/chat", json=corpo,
                       timeout=cfg.timeout)
    except Exception as exc:
        return f"modello non caricato: {exc}"
    if r.status_code != 200:
        return f"modello non caricato: HTTP {r.status_code}"
    return f"modello «{cfg.model}» pronto in {time.time() - t0:.1f}s"


def _leggi_flusso(risposta, _json) -> Iterator[str]:
    for riga in risposta.iter_lines():
        if not riga:
            continue
        try:
            dato = _json.loads(riga)
        except ValueError:
            continue
        pezzo = (dato.get("message") or {}).get("content", "")
        if pezzo:
            yield pezzo
        if dato.get("done"):
            break


_CITAZIONE_REGISTRO = re.compile(r" ?\[\s*Registr[oi]\b[^\]\n]*\]")


def _senza_citazioni_registro(pezzi: Iterator[str]) -> Iterator[str]:
    """Drops citations of the computed registers, such as [Registro presenze (calcolato)].

    The register is not a document the reader can open: the meetings it lists are
    cited one by one. The prompt says so, but the model still cites it in about a
    third of the answers. Text from an open "[" is held back until the bracket
    closes (or grows past any citation length), then released without the
    register citations.
    """
    sospeso = ""
    for pezzo in pezzi:
        sospeso += pezzo
        taglio = len(sospeso)
        aperta = sospeso.rfind("[")
        if aperta != -1 and "]" not in sospeso[aperta:] and len(sospeso) - aperta < 120:
            taglio = aperta
        # A trailing space waits too: it belongs to a citation that may still come.
        if taglio and sospeso[taglio - 1] == " ":
            taglio -= 1
        pronto, sospeso = sospeso[:taglio], sospeso[taglio:]
        pronto = _CITAZIONE_REGISTRO.sub("", pronto)
        if pronto:
            yield pronto
    if sospeso:
        yield _CITAZIONE_REGISTRO.sub("", sospeso)


_RIGA = re.compile(r"^\s*\|\s*(\d+)\s*\|")


def _righe_tabella(display: str) -> tuple[str, list[str]]:
    """Separa l'intestazione (breadcrumb + testata) dalle righe di dati."""
    testa, dati = [], []
    for r in display.splitlines():
        (dati if _RIGA.match(r) else testa).append(r)
    return "\n".join(testa).rstrip(), dati


def _ordine_riga(display: str) -> int:
    m = _RIGA.search(display)
    return int(m.group(1)) if m else 0


def _ricomponi_presenze(client, models, fonti: list[Fonte]) -> list[Fonte]:
    """Rimette insieme la tabella delle presenze, che il chunker spezza in 15 righe.

    E' il difetto che rendeva sbagliate le risposte sulle presenze, ed e'
    misurato: nel Verbale 04 del 28.03.2025 la tabella sta in **sei** chunk, e
    la domanda generata "Il prof. AULETTA Vincenzo era presente?" e' finita su
    **quattro** di quei sei — tre dei quali non contengono la sua riga. Il
    recupero funzionava benissimo e consegnava al modello tre esche e una
    risposta, occupando quattro posti su dieci. Chiesto di AULETTA, che era
    presente, il modello rispondeva che non risultava.

    Ricomposta, la tabella e' un blocco solo, completo, senza esche, e libera
    tre posti nel contesto. Costa uno `scroll` per verbale: in locale e' una
    scansione, ma su una domanda di presenze e' l'unica cosa che conta.
    """
    if not fonti:
        return fonti
    files = []
    for f in fonti:
        if f.genere == "table_presence" and f.file and f.file not in files:
            files.append(f.file)
    if not files:
        return fonti

    intere: dict[str, Fonte] = {}
    # Una sola tabella, quella della fonte meglio piazzata: sono ~6.000 caratteri
    # l'una, e due riempirebbero mezzo contesto con la seduta sbagliata. Il
    # tetto a due, per ordine di comparsa, aveva fatto proprio questo: ricomposti
    # i due verbali con un frammento solo, lasciati a pezzi quelli con tre.
    for percorso in files[:1]:
        cond = models.Filter(must=[
            models.FieldCondition(key="source_path", match=models.MatchValue(value=percorso)),
            models.FieldCondition(key="chunk_type",
                                  match=models.MatchValue(value="table_presence")),
        ])
        try:
            punti, _ = client.scroll(NOME_COLLEZIONE, scroll_filter=cond,
                                     limit=100, with_payload=True)
        except Exception:
            continue
        pezzi = sorted((p.payload for p in punti),
                       key=lambda x: _ordine_riga(x.get("display") or ""))
        if len(pezzi) < 2:
            continue
        testa, dati = _righe_tabella(pezzi[0].get("display") or "")
        for altro in pezzi[1:]:
            dati += _righe_tabella(altro.get("display") or "")[1]
        modello = next(f for f in fonti if f.file == percorso and f.genere == "table_presence")
        intere[percorso] = Fonte(
            chunk_id=modello.chunk_id, documento=modello.documento, data=modello.data,
            percorso=modello.percorso, testo=testa + "\n" + "\n".join(dati),
            punteggio=modello.punteggio, via=modello.via,
            domanda_corrisposta=modello.domanda_corrisposta, file=percorso,
            genere="table_presence",
        )

    fuse: list[Fonte] = []
    usati: set[str] = set()
    for f in fonti:
        if f.genere == "table_presence" and f.file in intere:
            if f.file in usati:
                continue        # gli altri pezzi della stessa tabella spariscono
            usati.add(f.file)
            fuse.append(intere[f.file])
        else:
            fuse.append(f)
    return fuse


def _unisci(mirate: list[Fonte], libere: list[Fonte], k: int) -> list[Fonte]:
    """Meta' dei posti alla ricerca filtrata, meta' a quella libera.

    Il filtro dedotto dalla domanda e' un'ipotesi, non un dato. Se e' giusta le
    due liste coincidono quasi del tutto e la deduplica fa si' che non si perda
    nulla; se e' sbagliata la risposta arriva comunque dalle libere. Prima il
    ripiego scattava solo a risultati zero: un filtro sbagliato che restituiva
    113 chunk irrilevanti non lo faceva scattare mai, ed e' esattamente il caso
    in cui serviva.
    """
    fonti: list[Fonte] = []
    visti: set[str] = set()

    def aggiungi(elenco: list[Fonte], tetto: int) -> None:
        for f in elenco:
            if len(fonti) >= tetto:
                return
            if f.chunk_id in visti:
                continue
            visti.add(f.chunk_id)
            fonti.append(f)

    aggiungi(mirate, max(1, k // 2))
    aggiungi(libere, k)
    aggiungi(mirate, k)          # i posti avanzati tornano al filtro
    return fonti
