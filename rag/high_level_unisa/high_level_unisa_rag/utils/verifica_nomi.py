# Copied unchanged from versione_finale/verifica_nomi.py of the RAG repository high-level-unisa (ari_servers: rag/high_level_unisa).
# -*- coding: utf-8 -*-
"""Verifica dei nomi dopo la generazione (02/10/2026).

Il modello a volte ricopia male un nome proprio negli elenchi lunghi (Egizio per EGIZIANO, Piemento per Piemonte,
Lo Nuestro per Lo Nostro, Saggesse per Saggese): la grafia sbagliata non e' in nessuna fonte, quella giusta si'.
Regola, provata a secco sulle 828 risposte salvate dei due banchi (207 per braccio, 4 bracci) (messa a punto sul 613, confermata sul 429):
  - parola sospetta: iniziale maiuscola, almeno 4 lettere, non a inizio frase, assente dal contesto (confronto
    senza accenti: MACRI' delle fonti e' Macri' della risposta);
  - nome spezzato: se la sospetta, unita alla parola maiuscola che la precede o la segue (separate da uno spazio),
    da' un nome del contesto, la coppia prende la forma del contesto (De Stefanis -> Destefanis), purche' le due
    parti non siano entrambe parole comuni;
  - non e' un nome del contesto una parola con una maiuscola subito dopo una minuscola (AziendaliManagement): sono
    due parole incollate dall'estrazione del testo dai PDF (correzione del 02/10, trovata con la misura);
  - altrimenti, se non e' una parola comune (in minuscolo almeno 2 volte nei pezzi dell'indice), prende il nome del
    contesto (tutto maiuscolo, o maiuscolo non a inizio frase) con la stessa iniziale, distanza di modifica <= 2 e
    somiglianza >= 0,7, se e' l'unico alla distanza minima.
Nessuna chiamata al modello: solo confronti di stringhe.
"""
import re
import unicodedata
from collections import Counter
from typing import Dict, Iterable, List, Set, Tuple

PAROLA = re.compile(r"[^\W\d_]+", re.U)
_INIZIO_FRASE = re.compile(r"(^|[.!?]\s?)$")
_INIZIO_FRASE_O_RIGA = re.compile(r"(^|[.!?\n]\s?)$")
# una minuscola seguita da una maiuscola dentro la parola: due parole incollate dall'estrazione dei PDF, non un nome
_INCOLLATA = re.compile(r"[a-zà-ÿ][A-ZÀ-Þ]")


def piatta(w: str) -> str:
    """minuscola e senza accenti"""
    return "".join(c for c in unicodedata.normalize("NFKD", w.lower()) if not unicodedata.combining(c))


def conta_minuscole(testi: Iterable[str]) -> Counter:
    """quante volte ogni parola compare scritta tutta in minuscolo"""
    conta = Counter()
    for t in testi:
        conta.update(w for w in PAROLA.findall(t or "") if w.islower())
    return conta


def parole_comuni(conta: Counter, soglia: int = 2) -> Set[str]:
    return {w for w, n in conta.items() if n >= soglia}


def lessico(contesto: str) -> Tuple[Set[str], Dict[str, str]]:
    """(parole del contesto senza accenti, {forma senza accenti: forma del contesto} delle parole usate come nome:
    tutte maiuscole, o maiuscole non a inizio frase o riga)"""
    tutte, nomi = set(), {}
    for m in PAROLA.finditer(contesto):
        w = m.group(0)
        tutte.add(piatta(w))
        if len(w) >= 3 and not _INCOLLATA.search(w) and (w.isupper() or (w[0].isupper()
                                             and not _INIZIO_FRASE_O_RIGA.search(contesto[max(0, m.start() - 2):m.start()]))):
            nomi.setdefault(piatta(w), w.lower())
    return tutte, nomi


def distanza(a: str, b: str) -> int:
    """distanza di Damerau-Levenshtein ristretta (trasposizioni adiacenti)"""
    d = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(len(a) + 1):
        d[i][0] = i
    for j in range(len(b) + 1):
        d[0][j] = j
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            c = 0 if a[i - 1] == b[j - 1] else 1
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + c)
            if i > 1 and j > 1 and a[i - 1] == b[j - 2] and a[i - 2] == b[j - 1]:
                d[i][j] = min(d[i][j], d[i - 2][j - 2] + 1)
    return d[len(a)][len(b)]


def _come(modello: str, w: str) -> str:
    return w.upper() if modello.isupper() else w[0].upper() + w[1:].lower()


def verifica_nomi(risposta: str, contesto: str, comuni: Set[str]) -> Tuple[str, List[Tuple[str, str]]]:
    """(risposta con i nomi ricopiati male corretti, [(com'era, come diventa)])"""
    tutte, nomi = lessico(contesto)
    parole = list(PAROLA.finditer(risposta))
    modifiche, fatte = [], []
    for k, m in enumerate(parole):
        w = m.group(0)
        if (len(w) < 4 or not w[0].isupper() or piatta(w) in tutte
                or _INIZIO_FRASE.search(risposta[max(0, m.start() - 2):m.start()])):
            continue
        # nome spezzato: la coppia con la parola maiuscola prima o dopo, unita, e' un nome del contesto
        coppia = None
        for i1, i2 in ((k - 1, k), (k, k + 1)):
            if 0 <= i1 and i2 < len(parole):
                p, q = parole[i1], parole[i2]
                if (risposta[p.end():q.start()] == " " and p.group(0)[0].isupper() and q.group(0)[0].isupper()
                        and piatta(p.group(0) + q.group(0)) in nomi
                        and not (p.group(0).lower() in comuni and q.group(0).lower() in comuni)):
                    coppia = (p, q)
                    break
        if coppia:
            p, q = coppia
            if any(s == p.start() and e == q.end() for s, e, _ in modifiche):
                continue          # la stessa coppia, gia' trovata partendo dalla parola che la precede
            nuova =_come(p.group(0), nomi[piatta(p.group(0) + q.group(0))])
            modifiche.append((p.start(), q.end(), nuova))
            fatte.append(("%s %s" % (p.group(0), q.group(0)), nuova))
            continue
        if w.lower() in comuni:
            continue
        wp = piatta(w)
        cand = [(distanza(wp, c), c) for c in nomi if c[0] == wp[0] and abs(len(c) - len(wp)) <= 3]
        cand = [(d, c) for d, c in cand if d <= 2 and 1 - d / max(len(c), len(wp)) >= 0.7]
        if not cand:
            continue
        dmin = min(d for d, _ in cand)
        migliori = [c for d, c in cand if d == dmin]
        if len(migliori) != 1:
            continue
        nuova = _come(w, nomi[migliori[0]])
        modifiche.append((m.start(), m.end(), nuova))
        fatte.append((w, nuova))
    out, ultimo = [], 0
    for s, e, nuova in sorted(modifiche):
        if s < ultimo:
            continue          # sovrapposta a una modifica gia' fatta
        out.append(risposta[ultimo:s])
        out.append(nuova)
        ultimo = e
    out.append(risposta[ultimo:])
    return "".join(out), fatte
