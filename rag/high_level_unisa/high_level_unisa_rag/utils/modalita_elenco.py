# Copied unchanged from modalita_elenco.py of the RAG repository high-level-unisa (ari_servers: rag/high_level_unisa), the module of the final configuration (doc/resoconto_attivita_5.pdf, test_baseline/sistema.sh).
# -*- coding: utf-8 -*-
"""Modalita' elenco (30/09/2026): riconoscere le domande di elenco di entita' e scrivere le ricerche
orientate alla FONTE. Nessuna dipendenza dalla pipeline, che lo usa quando ENABLE_LIST_MODE e' acceso.

Riconoscimento: la stessa regola della stima (stima_multiquery.py) e della mappa degli elenchi:
casistica INSIEME del classificatore e un sostantivo di insieme chiuso. Le regole sono copiate
IDENTICHE da classificatore.py (REGOLE, QUALE) e da mappa_elenchi.py (CHIUSO); l'identita' sulle
1.042 domande dei due banchi la controlla verifica_modalita_elenco.py.
Ricerche: il prompt e' lo stesso della stima, scritto sulle 613 e mai ritoccato guardando le 429.
Il modello ne scrive 3, una per riga; righe numerate o puntate vengono ripulite.
"""
import re

# ------------------------------------------------------------------ classificatore (identico)
# «qual è» con la e accentata: «quale» ha la e piana e non deve corrispondere
QUALE = r"qual\s*['’]?\s*è"

REGOLE = [
    # --- specifici, sopra i generici -----------------------------------------------------
    ("PUNTUALE", "ruolo di una persona", re.compile(
        r"\bquale ruolo (?:ricopre|svolge|ha)\b|\bche ruolo (?:ricopre|svolge|ha)\b", re.I)),

    ("INSIEME", "conteggio esplicito", re.compile(
        r"\bquant[ei]\b"
        r"|\bnumero (?:di|totale|complessivo)\b(?!\s+(?:telefono|fax|cellulare|matricola"
        r"|protocollo|pratica|iscrizione|stanza|interno))", re.I)),

    ("PUNTUALE", "recapito o identificativo", re.compile(
        QUALE + r"\s+(?:l'|il |la |lo )?(?:indirizzo|e-?mail|posta elettronica|pec|numero"
        r"|telefono|interno|sede|ubicazione|codice|matricola|orario|link|sito|url|recapito"
        r"|settore|qualifica|durata|importo|scadenza|data)\b", re.I)),

    ("PUNTUALE", "entita' o attributo singolo", re.compile(
        r"\bchi (?:è|e')\b|\bchi (?:firm|dirige|presiede|coordina|nomina|guida|rappresenta)\w*\b"
        r"|\bdi quale\b|\ba quale\b|\bin quale (?:anno|data|dipartimento|struttura|area|facolt[àa]|sede)\b"
        r"|\bquando (?:è|e'|fu|venne|è stato|è stata)\b"
        r"|\bquale (?:dipartimento|struttura|ufficio|area|facolt[àa]|organo|corso|comitato)\s+"
        r"(?:ha|è|e'|dirige|coordina|afferisce|appartiene|comprende)\b"
        r"|\bquale (?:indirizzo|decreto|documento|legge|regolamento|delibera|modello|norma"
        r"|provvedimento|statuto)\b"
        r"|\b" + QUALE + r"\s+(?:il |la |lo )?(?:livello|grado|percentuale|valore|punteggio"
        r"|tasso|media)\b", re.I)),

    # --- generici ------------------------------------------------------------------------
    ("SPIEGAZIONE", "definizione, scopo o procedura", re.compile(
        r"\b(?:in )?(?:che )?cosa consiste\b|\bche cos'?[èe]\b|\bcos'?[èe]\b"
        + r"|\b" + QUALE + r"\s+(?:il |la |lo |l')?(?:ruolo|scopo|funzione|obiettivo|significato"
        r"|finalit[àa]|procedura|iter|processo|criterio|differenza|missione|visione)\b"
        r"|\bcome (?:funziona|avviene|si svolge|viene|vengono|si accede|si presenta|si ottiene)\b"
        r"|\bin che modo\b|\bperch[ée]\b|\bper quale motivo\b"
        r"|\bquale (?:fu|è stato|è stata) (?:il |l'|la )?"
        r"(?:significato|impatto|contributo|effetto)\b", re.I)),

    ("INSIEME", "elenco esplicito", re.compile(
        r"\belenc\w*\b|\bchi sono\b|\bda chi (?:è|e') (?:composto|composta|formato|formata)\b"
        r"|\bfanno capo\b|\bcompongono\b|\bcomposizione\b|\bquali sono\b|\bquali\b", re.I)),
]


def classifica(domanda):
    for casistica, nome, rx in REGOLE:
        if rx.search(domanda):
            return casistica, nome
    return "NON CLASSIFICATA", "nessuno schema"


# ------------------------------------------------------------------ insieme chiuso (identico)
CHIUSO = re.compile(
    r"\b(?:dipartiment|uffici|aree|are[ea]\b|organi|commissioni|collegi|corsi|centri|"
    r"strutture|componenti|membri|delegati|referenti|professori|docenti|direttori|"
    r"rappresentanti|consiglieri)\w*\b", re.I)


def e_elenco(domanda):
    """la domanda chiede un elenco o un conteggio di entita' (uffici, aree, componenti...)"""
    return classifica(domanda)[0] == "INSIEME" and bool(CHIUSO.search(domanda))


# ------------------------------------------------------------------ ricerche orientate alla fonte
PROMPT_ELENCO = """Devi preparare le ricerche per trovare, in un archivio di pagine web e documenti dell'Università degli Studi di Salerno, il documento che contiene PER INTERO l'elenco richiesto dalla domanda.

Il tipo di documento dipende dall'insieme richiesto: gli uffici e le aree stanno nell'organigramma, in tabelle; i componenti di un organo nella pagina che ne descrive la composizione o nel decreto di nomina; i delegati e i referenti nel decreto o nella pagina delle deleghe; il personale di un ufficio nella pagina della rubrica di quella struttura; i dipartimenti, i professori emeriti o i corsi in una pagina che li elenca tutti.

Scrivi esattamente 3 ricerche:
- la prima a parole chiave: l'insieme richiesto (per esempio uffici, aree, componenti, delegati, dipartimenti), i termini distintivi della domanda (nomi propri, sigle, numeri romani, nomi di uffici o di organi) e il tipo di documento adatto;
- la seconda: una frase breve che descrive la pagina in cui l'elenco è scritto;
- la terza: una frase breve che nomina l'insieme richiesto con le parole che userebbe il sito dell'università.

Regole: usa solo entità presenti nella domanda, non aggiungere nomi di persone, uffici o organi che la domanda non contiene; non rispondere alla domanda; una ricerca per riga, senza numeri, trattini o commenti.

Domanda: {domanda}
Ricerche:"""


def ricerche(testo_modello, n=3):
    """le prime n righe non vuote dell'uscita del modello, senza numeri o puntini iniziali"""
    righe = [re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", r).strip() for r in (testo_modello or "").splitlines()]
    return [r for r in righe if len(r) > 3][:n]


def unici(docs):
    """candidati senza doppioni: lo stesso pezzo trovato da piu' ricerche conta una volta"""
    visti, out = set(), []
    for d in docs:
        k = (d.metadata.get("parent_id"), d.metadata.get("source_url"), d.page_content)
        if k not in visti:
            visti.add(k)
            out.append(d)
    return out
