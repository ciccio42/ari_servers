# Copied unchanged from versione_finale/percorso_struttura.py of the RAG repository high-level-unisa (ari_servers: rag/high_level_unisa).
# percorso_struttura.py - versione finale (14/10/2026): era struttura_lookup.py (579da739). Cambiano solo il
# percorso predefinito del registro, ora dati/registro_strutture.json, e i nomi dei moduli citati nei commenti.
# -*- coding: utf-8 -*-
"""Riconoscitore deterministico e schede del percorso «struttura» (25/09/2026).

Dal 07/10/2026 anche le domande su un dato di UNA struttura: capo ed email di un ufficio, di un'area o
della Direzione Generale, distretto di un dipartimento (si veda «attributi di una struttura» piu' sotto).
Con Registro(..., attributi=False) il riconoscitore e' quello del 25/09/2026.

Gemello di percorso_persona.py: nessuna chiamata a un modello. Tre funzioni:
  - `Registro.riconosci(domanda)` dice se la domanda chiede di ELENCARE o CONTARE i componenti di
    una struttura del registro, e quale operazione;
  - `Registro.scheda(esito)` scrive il testo da passare al modello, con i conteggi gia' fatti;
  - `percorso_veloce(domanda)` dice se il recupero si puo' saltare (la domanda chiede solo cio'
    che la scheda contiene) o se la scheda va messa in testa al contesto recuperato.

Servono DUE condizioni insieme: un'entita' del registro nominata nella domanda (area, ufficio,
distretto, Direzione Generale / Amministrazione Centrale) e un'operazione che il registro copre
(elenco o conteggio di uffici, personale, aree, dipartimenti; una proprieta' sull'elenco del
personale). Le domande su un attributo di UNA struttura («Chi e' il Capo dell'Ufficio X»,
«Qual e' l'email dell'Ufficio X») non si intercettano: funzionano gia', la ricerca trova la pagina.
La precedenza del percorso persona si decide nella pipeline, non qui.

28/09/2026, uffici dell'Amministrazione Centrale e della Direzione Generale (operazione uffici_dg):
la scheda dava solo conteggi, anche a chi chiedeva «quali», e un unico «in tutto: 103» che le
domande di conteggio ripetevano. Ora la scheda dipende da cosa si chiede (elenco, conteggio o
entrambi) e da cosa si nomina (Amministrazione Centrale o Direzione Generale). Per l'Amministrazione
Centrale le fonti danno tre conteggi diversi, e la scheda li riporta tutti con la loro definizione:
64 negli uffici delle 7 Aree, 80 nella voce «Uffici Amministrazione Centrale» della rubrica (senza
gli Uffici di Coordinamento), 103 nell'organigramma completo; l'elenco e' quello della rubrica,
raggruppato come nell'organigramma.
"""
import json
import re
import unicodedata
from pathlib import Path

REGISTRO = Path(__file__).resolve().parent / "dati" / "registro_strutture.json"
VUOTE = {"di", "del", "della", "dello", "dei", "degli", "delle", "dell", "e", "ed", "per", "il", "la",
         "lo", "l", "i", "gli", "le", "a", "al", "alla", "all", "in", "d"}
ROMANI = {"i": "I", "ii": "II", "iii": "III", "iv": "IV", "v": "V", "vi": "VI", "vii": "VII"}

CONTEGGIO = re.compile(r"\bquant[ieao]\b|\bnumero (?:complessivo |totale )?(?:di |degli |delle |dei )"
                       r"(?!telefono|matricola|fax|protocollo|interno|pratica|stanza)", re.I)
PERSONALE = re.compile(r"\b(personale|membri|dipendenti|chi lavora|lavorano|addett[io]|"
                       r"chi (?:fa|fanno) parte)\b", re.I)
UFFICI = re.compile(r"\b(uffici|strutture|unit[àa] organizzative|fanno capo|afferiscono|dipendono|"
                    r"compongono|articola\w*)\b", re.I)
COORDINAMENTI = re.compile(r"\buffici di coordinamento\b", re.I)
AREE = re.compile(r"\baree\b", re.I)
DIPARTIMENTI = re.compile(r"\bdipartiment\w*\b", re.I)
ELENCO = re.compile(r"\bquali\b|\belenc\w*\b|\bchi sono\b|\bnomi\b|\bresponsabili\b|"
                    r"\bchi (?:fa|fanno) parte\b|\bchi lavora\b|\bcompost[oa]\b|\bcomposizione\b", re.I)
PROPRIETA = re.compile(r"\bquale membro\b|\bchi (?:non )?ha\b|\bnon (?:ha|hanno)\b", re.I)
# La scheda non riporta i CV: una domanda sui CV non si intercetta affatto, altrimenti il percorso
# veloce toglierebbe dal contesto proprio la pagina che li contiene ([96] delle 613).
ESCLUSI = re.compile(r"\bcv\b|\bcurricul\w*", re.I)
# argomenti che il registro non contiene: se ci sono, la scheda va in testa al contesto recuperato
FUORI_SCHEDA = re.compile(r"\b(compiti|funzion\w+|competenz\w+|attivit[àa]\w*|si occup\w+|di cosa|"
                          r"orari\w*|ricevimento|dove si trova|sede|procedur\w+|servizi offerti|"
                          r"regolament\w+|decret\w+|storia|quando)\b", re.I)

# ------------------------------------------------------------------ attributi di una struttura (06/10/2026)
# Un dato di UNA struttura del registro: il capo (o responsabile) e l'email di un ufficio, di un'area o della Direzione
# Generale; il distretto di un dipartimento. Fino al 05/10 queste domande non si intercettavano («la ricerca trova la
# pagina»); l'analisi dei fallimenti del 06/10 ha trovato pagine della rubrica e la pagina Contatti non recuperate.
# Le parole sono generiche, i nomi delle strutture vengono dal registro: nessuna regola nasce dalle domande del banco.
# La scheda va sempre in testa al contesto recuperato (la pipeline non usa il percorso veloce per queste operazioni).
# Si intercetta solo cio' che il registro sa: se la struttura non ha il dato chiesto la domanda prosegue come prima, e la
# scheda riporta solo dati presenti (mai «non indicato», che spingerebbe il modello a rinunciare). La PEC non e' nel
# registro, quindi non e' fra le parole dell'email.
ATTR_CAPO = re.compile(r"\b(capo|responsabile|dirigente|chi (?:dirige|guida|coordina))\b", re.I)
ATTR_EMAIL = re.compile(r"\be-?mail\b|\bposta elettronica\b", re.I)
ATTR_DISTRETTO = re.compile(r"\bdistrett[oi]\b", re.I)
OPS_ATTRIBUTO = {"attributo_ufficio", "attributo_area", "attributo_dg", "distretto_dipartimento"}
CONTATTI = "https://web.unisa.it/ateneo/contatti"


def piatto(s):
    s = unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9]+", " ", s).split()


def parole(s):
    return [p for p in piatto(s) if p not in VUOTE]


def contiene(seq, sotto):
    n = len(sotto)
    return n > 0 and any(seq[k:k + n] == sotto for k in range(len(seq) - n + 1))


def nome_leggibile(nome_rubrica):
    """«ALBANO Fernanda» -> «Fernanda ALBANO»: la forma della pagina delle aree (Nome COGNOME)"""
    parti = (nome_rubrica or "").split()
    cogn = []
    while parti and parti[0].replace("'", "").replace("’", "").isupper():
        cogn.append(parti.pop(0))
    return " ".join(parti + cogn) if parti and cogn else (nome_rubrica or "")


class Registro:
    def __init__(self, percorso=REGISTRO, attributi=False):
        r = json.loads(Path(percorso).read_text(encoding="utf-8"))
        self.dg, self.aree, self.uffici, self.distretti = (r["direzione_generale"], r["aree"],
                                                           r["uffici"], r["distretti"])
        # voce «Uffici Amministrazione Centrale» della rubrica (registro dal 28/09); senza, le schede
        # uffici_dg dell'Amministrazione Centrale non si possono scrivere e la pipeline prosegue normale
        self.rubrica_amm = r.get("rubrica_amministrazione")
        # alias degli uffici come sequenze di parole piene, dal piu' lungo al piu' corto: vince
        # la corrispondenza piu' specifica («Ufficio Ricerca, Contratti...» prima di «Ufficio Ricerca»)
        self.alias = sorted(((parole(a), i) for i, u in self.uffici.items() for a in u.get("alias", []) if a),
                            key=lambda x: -len(x[0]))
        self.alias = [(a, i) for a, i in self.alias if len(a) >= 2]
        # attributi di una struttura: spenti, nulla cambia rispetto a percorso_struttura.py
        self.attributi = attributi
        # i dipartimenti dei distretti, per nome (senza sigla, parti fra virgolette o parentesi e la parola
        # «dipartimento») e per sigla; i nomi piu' lunghi prima, perche' vinca il piu' specifico
        self.dipartimenti = []
        for d in self.distretti:
            for x in d["dipartimenti"]:
                nome = re.sub(r"\(.*?\)|[\"“”«»].*?[\"“”«»]", " ", x["nome"].split("/")[0])
                self.dipartimenti.append(([w for w in parole(nome) if w != "dipartimento"], x["sigla"], d["numero"], x))
        self.dipartimenti.sort(key=lambda t: -len(t[0]))

    # ------------------------------------------------------------------ riconoscimento
    def entita(self, domanda):
        p = parole(domanda)
        m = re.search(r"\bdistretto\s*(\d)\b", " ".join(piatto(domanda)))
        if m:
            return ("distretto", int(m.group(1)))
        for a, i in self.alias:
            if contiene(p, a):
                return ("ufficio", i, a)
        m = re.search(r"\barea (i|ii|iii|iv|v|vi|vii)\b", " ".join(piatto(domanda)))
        if m:
            return ("area", ROMANI[m.group(1)])
        if re.search(r"direzione generale|amministrazione centrale", " ".join(piatto(domanda))):
            return ("dg", None)
        return None

    def riconosci(self, domanda):
        """dict con entita', operazione e motivo, oppure None con il motivo del no. Prima gli elenchi e i conteggi
        di sempre; solo se non c'e' nulla da fare e gli attributi sono accesi, un dato di una sola struttura."""
        esito, motivo = self._riconosci_elenchi(domanda)
        if esito or not self.attributi:
            return esito, motivo
        a = self.attributo(domanda)
        if a:
            return a, {"attributo_ufficio": "dato di un ufficio", "attributo_area": "dato di un'area",
                       "attributo_dg": "dato della Direzione Generale",
                       "distretto_dipartimento": "distretto di un dipartimento"}[a["op"]]
        return None, motivo

    def strutture_nominate(self, domanda):
        """i nomi delle strutture del registro che la domanda nomina: l'ufficio o l'area (come in entita) e il
        dipartimento (per nome o per sigla). Serve al percorso persona (08/10/2026): una parola che fa parte del
        nome di una di queste strutture non e' il nome di una persona."""
        out = []
        e = self.entita(domanda)
        if e and e[0] == "ufficio":
            out.append(self.uffici[e[1]].get("nome") or "")
        if e and e[0] == "area":
            out += [a["nome"] for a in self.aree if a["numero"] == e[1]]
        dip = self._dipartimento(domanda)
        if dip:
            out.append(dip[0]["nome"])
        return [n for n in out if n]

    def _dipartimento(self, domanda):
        """(dipartimento, numero del distretto) nominato nella domanda, per nome o per sigla, oppure None. Le sigle
        si cercano come parola intera nel testo originale; quelle di due lettere (DI, DF) solo in maiuscolo,
        perche' «di» e' anche una preposizione."""
        p = parole(domanda)
        for a, sigla, n, x in self.dipartimenti:
            if a and contiene(p, a):
                return x, n
        for a, sigla, n, x in self.dipartimenti:
            if re.search(r"(?<![A-Za-z])%s(?![A-Za-z])" % re.escape(sigla), domanda, 0 if len(sigla) <= 2 else re.I):
                return x, n
        return None

    def attributo(self, domanda):
        """operazione su un dato di UNA struttura, oppure None"""
        if ESCLUSI.search(domanda):
            return None
        if ATTR_DISTRETTO.search(domanda):
            dip = self._dipartimento(domanda)
            if dip:
                return {"op": "distretto_dipartimento", "chiave": dip[1], "dipartimento": dip[0]}
        chiede = [n for n, rx in (("capo", ATTR_CAPO), ("email", ATTR_EMAIL)) if rx.search(domanda)]
        e = self.entita(domanda)
        if not e or not chiede:
            return None
        if e[0] == "ufficio":
            u = self.uffici[e[1]]
            chiede = [c for c in chiede if u.get(c)]
            return {"op": "attributo_ufficio", "chiave": e[1], "chiede": chiede} if chiede else None
        if e[0] == "area":
            return {"op": "attributo_area", "chiave": e[1], "chiede": chiede}
        if e[0] == "dg":
            chiede = [c for c in chiede if self.dg.get(c)]
            return {"op": "attributo_dg", "chiave": None, "chiede": chiede} if chiede else None
        return None

    def _riconosci_elenchi(self, domanda):
        """dict con entita', operazione e motivo, oppure None con il motivo del no"""
        e = self.entita(domanda)
        if not e:
            return None, "nessuna struttura del registro nominata"
        if ESCLUSI.search(domanda):
            return None, "chiede i CV, che la scheda non riporta"
        tipo, chiave = e[0], e[1]
        conta = bool(CONTEGGIO.search(domanda))
        elenca = bool(ELENCO.search(domanda))
        if not (conta or elenca or PROPRIETA.search(domanda)):
            return None, "attributo di una sola struttura (non un elenco ne' un conteggio)"
        if tipo == "distretto":
            if DIPARTIMENTI.search(domanda):
                return {"op": "distretto", "chiave": chiave, "conta": conta}, "dipartimenti del distretto"
            return None, "distretto nominato, ma non si chiedono i dipartimenti"
        if tipo == "ufficio":
            u = self.uffici[chiave]
            # Gli indizi dell'operazione si cercano nella domanda SENZA il nome dell'ufficio: in
            # «Quali uffici fanno capo all'Ufficio di Coordinamento Personale Tecnico-Amministrativo»
            # la parola «personale» appartiene al nome, e non vuol dire che si chieda il personale.
            p, a = parole(domanda), e[2]
            k = next(j for j in range(len(p) - len(a) + 1) if p[j:j + len(a)] == a)
            resto = " ".join(p[:k] + p[k + len(a):])
            if PERSONALE.search(resto):
                if not u["personale"]:
                    return None, "ufficio senza personale nel registro: pipeline normale"
                op = "proprieta_personale" if PROPRIETA.search(domanda) else "personale"
                return {"op": op, "chiave": chiave, "conta": conta}, "personale dell'ufficio"
            if UFFICI.search(resto) and u["uffici_dipendenti"]:
                return {"op": "uffici_dipendenti", "chiave": chiave, "conta": conta}, "uffici che ne dipendono"
            return None, "ufficio nominato, ma non si chiede ne' il personale ne' gli uffici dipendenti"
        if tipo == "area":
            if UFFICI.search(domanda):
                return {"op": "uffici_area", "chiave": chiave, "conta": conta}, "uffici dell'area"
            return None, "area nominata, ma non si chiedono i suoi uffici"
        # Direzione Generale / Amministrazione Centrale
        if AREE.search(domanda):
            return {"op": "aree", "chiave": None, "conta": conta}, "aree dell'amministrazione"
        if COORDINAMENTI.search(domanda):
            return {"op": "coordinamenti_dg", "chiave": None, "conta": conta}, "coordinamenti in staff alla DG"
        if UFFICI.search(domanda):
            # cosa si chiede (elenco, conteggio o entrambi; una domanda senza «quanti» vuole i nomi)
            # e cosa si nomina: la Direzione Generale da sola, o l'Amministrazione Centrale
            testo = " ".join(piatto(domanda))
            ambito = ("direzione generale" if "direzione generale" in testo and "amministrazione centrale" not in testo
                      else "amministrazione centrale")
            return ({"op": "uffici_dg", "chiave": None, "conta": conta, "elenca": elenca or not conta,
                     "ambito": ambito}, "uffici della Direzione Generale" if ambito == "direzione generale"
                    else "uffici dell'Amministrazione Centrale")
        return None, "Direzione Generale nominata, ma non si chiede un elenco coperto dal registro"

    # ------------------------------------------------------------------ schede
    def _riga(self, u, rientro="- "):
        if u.get("capo"):
            capo = "%s%s: %s" % (u["ruolo_capo"], " ad interim" if u["ad_interim"] else "", u["capo"])
        else:
            capo = "capo non indicato nella fonte"
        return "%s%s (%s)" % (rientro, u.get("nome") or "ufficio di cui la fonte non riporta il nome", capo)

    def _albero(self, ids):
        righe = []
        for i in ids:
            u = self.uffici[i]
            if u["livello"] == "primo":
                righe.append(self._riga(u))
                for j in u["uffici_dipendenti"]:
                    righe.append(self._riga(self.uffici[j], "   - "))
        return righe

    # ------------------------------------------------------------------ Amministrazione Centrale / DG
    def _n_aree(self):
        return sum(a["conteggi"]["uffici"] for a in self.aree)

    def _conteggio_ac(self):
        """i tre conteggi delle fonti, ciascuno con la sua definizione: nessuno e' «il» numero"""
        c, ra = self.dg["conteggi"], self.rubrica_amm["conteggi"]
        coord_aree = sum(a["conteggi"]["coordinamenti"] for a in self.aree)
        coord_tutti = coord_aree + c["coordinamenti_in_staff"]
        escl = ("non comprende gli Uffici di Coordinamento" if ra["coordinamenti_non_in_rubrica"] == coord_tutti
                else "non comprende %d dei %d Uffici di Coordinamento" % (ra["coordinamenti_non_in_rubrica"], coord_tutti))
        return [
            "Uffici dell'Amministrazione Centrale: il numero dipende da cosa si conta. Le fonti danno tre conteggi:",
            "- uffici delle %d Aree (pagina Aree): %d (%s), compresi %s;"
            % (len(self.aree), self._n_aree(), ", ".join("AREA %s: %d" % (a["numero"], a["conteggi"]["uffici"])
                                                       for a in self.aree), plurale(coord_aree)),
            # «80 uffici, oltre alle 7 Aree» e' stato letto come «80, Aree comprese» (collaudo del
            # 28/09, [429 180]): la frase dice esplicitamente che le Aree non sono contate
            "- rubrica di ateneo, voce «Uffici Amministrazione Centrale»: %d uffici; la stessa voce elenca a parte "
            "anche le %d Aree, che non sono contate fra questi %d; %s;" % (ra["uffici"], ra["aree"], ra["uffici"], escl),
            "- organigramma completo (pagine Aree e Direzione): %d unita' = %d nelle Aree + %d in staff alla "
            "Direzione Generale (di cui %s) + %d uffici distrettuali."
            % (c["uffici_amministrazione_centrale"], self._n_aree(), c["unita_in_staff"],
               plurale(c["coordinamenti_in_staff"]), c["uffici_in_staff_e_dipendenti"] - c["unita_in_staff"])]

    def _elenco_ac(self):
        """gli uffici della voce «Uffici Amministrazione Centrale» della rubrica, raggruppati come
        nell'organigramma; nessuno si perde: chi non trova posto in un gruppo va in «altri»"""
        ra = self.rubrica_amm
        nomi = {v["id"]: v["nome"] for v in ra["uffici"]}
        solo = set(ra["solo_rubrica"])
        t = ["Uffici dell'Amministrazione Centrale secondo la rubrica di ateneo (voce «Uffici Amministrazione "
             "Centrale»): %d uffici; la stessa voce elenca a parte anche le %d Aree, che non sono contate fra questi "
             "%d. Sono raggruppati qui come nell'organigramma (pagine Aree e Direzione)."
             % (ra["conteggi"]["uffici"], ra["conteggi"]["aree"], ra["conteggi"]["uffici"])]
        messi = set()
        for a in self.aree:
            dentro = [i for i in a["uffici"] if i in nomi]
            if dentro:
                t.append("%s:" % a["nome"])
                t += ["- %s" % nomi[i] for i in dentro]
                messi.update(dentro)
        primi = [i for i in self.dg["uffici"] if self.uffici[i]["livello"] == "primo"]
        staff = [i for i in primi if i in nomi]
        if staff:
            t.append("Unita' in staff alla Direzione Generale:")
            t += ["- %s" % nomi[i] for i in staff]
            messi.update(staff)
        gruppi = [(i, [j for j in self.uffici[i]["uffici_dipendenti"] if j in nomi]) for i in primi]
        gruppi = [(i, f) for i, f in gruppi if f]
        if gruppi:
            t.append("Uffici che dipendono dalle unita' in staff alla Direzione Generale (uffici distrettuali):")
            for i, figli in gruppi:
                t.append("  sotto «%s»:" % self.uffici[i]["nome"])
                t += ["- %s" % nomi[j] for j in figli]
                messi.update(figli)
        altri = [i for i in nomi if i not in messi and i not in solo]
        if altri:
            t.append("Altri uffici dell'organigramma:")
            t += ["- %s" % nomi[i] for i in altri]
        if solo:
            t.append("Nella rubrica ma non nell'organigramma delle pagine Aree e Direzione:")
            t += ["- %s" % nomi[i] for i in ra["solo_rubrica"]]
        fuori = [u for u in self.uffici.values() if not u.get("in_rubrica_amministrazione")]
        if fuori:
            coord = [u for u in fuori if u["coordinamento"]]
            resto = [u for u in fuori if not u["coordinamento"]]
            parti = []
            if coord:
                n_dg = sum(1 for u in coord if u["contenitore"] == "Direzione Generale")
                parti.append("%s (%d nelle Aree e %d in staff alla Direzione Generale)"
                             % (plurale(len(coord)), len(coord) - n_dg, n_dg))
            if resto:
                parti.append("%s: %s" % ("1 ufficio" if len(resto) == 1 else "%d uffici" % len(resto),
                                         ", ".join(u.get("nome") or "ufficio senza nome" for u in resto)))
            t.append("L'organigramma comprende anche %d unita' che la rubrica non elenca fra gli Uffici "
                     "dell'Amministrazione Centrale: %s." % (len(fuori), " e ".join(parti)))
        return t

    def _conteggio_dg(self):
        c = self.dg["conteggi"]
        return [
            "Secondo la pagina Direzione, fanno capo alla Direzione Generale:",
            "- direttamente, %d unita' in staff, di cui %s;" % (c["unita_in_staff"], plurale(c["coordinamenti_in_staff"])),
            "- attraverso le unita' in staff, %d uffici che ne dipendono (gli uffici distrettuali): con le unita' "
            "in staff, %d;" % (c["uffici_in_staff_e_dipendenti"] - c["unita_in_staff"], c["uffici_in_staff_e_dipendenti"]),
            "- inoltre le %d Aree, con %d uffici: in tutto, nell'organigramma, %d."
            % (len(self.aree), self._n_aree(), c["uffici_amministrazione_centrale"])]

    def _elenco_dg(self):
        t = ["Unita' in staff alla Direzione Generale (pagina Direzione), con gli uffici che ne dipendono:"]
        for i in self.dg["uffici"]:
            u = self.uffici[i]
            if u["livello"] != "primo":
                continue
            t.append("- %s%s" % (u.get("nome") or "ufficio di cui la fonte non riporta il nome",
                                 " (Ufficio di Coordinamento)" if u["coordinamento"] else ""))
            t += ["   - %s" % (self.uffici[j].get("nome") or "ufficio di cui la fonte non riporta il nome")
                  for j in u["uffici_dipendenti"]]
        t.append("Fanno capo alla Direzione Generale anche le %d Aree: %s." % (len(self.aree), "; ".join(
            "%s (%d uffici)" % (a["nome"], a["conteggi"]["uffici"]) for a in self.aree)))
        return t

    def _scheda_attributo(self, esito):
        """scheda di un dato di UNA struttura: nome, capo, email, appartenenza; o il distretto di un dipartimento"""
        op, k = esito["op"], esito["chiave"]
        t = ["SCHEDA DELLA STRUTTURA (registro dell'organigramma di ateneo)"]
        if op == "attributo_ufficio":
            u = self.uffici[k]
            riga = u.get("nome") or "ufficio di cui la fonte non riporta il nome"
            if u.get("capo"):
                riga += " (%s%s: %s)" % (u["ruolo_capo"], " ad interim" if u["ad_interim"] else "", u["capo"])
            t.append(riga + ". Fa parte di: %s." % u["contenitore"])
            if u.get("email"):
                t.append("Email dell'ufficio: %s." % u["email"])
            fonte = u["fonte"] + ("; https://rubrica.unisa.it/strutture?struttura=%s" % u["id"] if u.get("email") else "")
        elif op == "attributo_area":
            a = next(x for x in self.aree if x["numero"] == k)
            t.append("%s. Capo Area%s: %s. Email: %s." % (a["nome"], " ad interim" if a["ad_interim"] else "",
                                                          a["capo"], a["email"]))
            fonte = "%s; %s" % (a["fonte"], CONTATTI)
        elif op == "attributo_dg":
            t.append("Direzione Generale. %s: %s.%s" % (self.dg.get("ruolo_capo") or "Direttore Generale", self.dg["capo"],
                                                        " Email: %s." % self.dg["email"] if self.dg.get("email") else ""))
            fonte = "%s; %s" % (self.dg["fonte"], CONTATTI)
        else:
            d = next(x for x in self.distretti if x["numero"] == k)
            dip = esito["dipartimento"]
            altri = [x["nome"] for x in d["dipartimenti"] if x["sigla"] != dip["sigla"]]
            t.append("Il %s fa parte del DISTRETTO%d%s" % (dip["nome"], k, (", insieme a: %s." % "; ".join(altri))
                                                          if altri else "."))
            fonte = d["fonte"]
        t.append("Fonte: %s" % fonte)
        t.append("FINE DELLA SCHEDA")
        return "\n".join(t)

    def scheda(self, esito):
        op, k = esito["op"], esito["chiave"]
        if op in OPS_ATTRIBUTO:
            return self._scheda_attributo(esito)
        t = ["SCHEDA DELLA STRUTTURA (registro dell'organigramma di ateneo)"]
        if op == "uffici_area":
            a = next(x for x in self.aree if x["numero"] == k)
            c = a["conteggi"]
            t.append("%s. Capo Area%s: %s. Email: %s." % (a["nome"], " ad interim" if a["ad_interim"] else "",
                                                          a["capo"], a["email"]))
            t.append("Uffici che fanno capo all'Area: %d, di cui %s. Gli uffici elencati sotto un Ufficio "
                     "di Coordinamento dipendono da quest'ultimo." % (c["uffici"], plurale(c["coordinamenti"])))
            t += self._albero(a["uffici"])
            fonte = a["fonte"]
        elif op == "uffici_dipendenti":
            u = self.uffici[k]
            t.append(self._riga(u, "").lstrip() + ". Fa parte di: %s." % u["contenitore"])
            t.append("Uffici che ne dipendono: %d." % len(u["uffici_dipendenti"]))
            t += [self._riga(self.uffici[j]) for j in u["uffici_dipendenti"]]
            fonte = u["fonte"]
        elif op in ("personale", "proprieta_personale"):
            u = self.uffici[k]
            t.append(self._riga(u, "").lstrip() + ". Fa parte di: %s." % u["contenitore"])
            t.append("Personale dell'ufficio secondo la rubrica di ateneo: %d persone." % len(u["personale"]))
            for p in u["personale"]:
                if p["fonte"] != "anagrafe":
                    t.append("- %s (recapiti non disponibili nel registro)" % nome_leggibile(p["nome"]))
                    continue
                pezzi = [nome_leggibile(p["nome"])]
                if p.get("qualifica"):
                    pezzi.append(p["qualifica"])
                pezzi.append("interno %s" % ", ".join(p["interni"]) if p["interni"] else "nessun interno in rubrica")
                if p.get("email"):
                    pezzi.append("email %s" % ", ".join(p["email"]))
                t.append("- " + "; ".join(pezzi))
            fonte = "https://rubrica.unisa.it/strutture?struttura=%s" % k
        elif op == "aree":
            t.append("La Direzione Generale (Direttore Generale: %s) e' articolata in %d Aree:"
                     % (self.dg["capo"], len(self.aree)))
            for a in self.aree:
                t.append("- %s. Capo Area%s: %s. Email: %s. Uffici: %d."
                         % (a["nome"], " ad interim" if a["ad_interim"] else "", a["capo"], a["email"],
                            a["conteggi"]["uffici"]))
            fonte = self.aree[0]["fonte"]
        elif op == "coordinamenti_dg":
            staff = [self.uffici[i] for i in self.dg["uffici"] if self.uffici[i]["livello"] == "primo"]
            coord = [u for u in staff if u["coordinamento"]]
            altre = [u for u in staff if not u["coordinamento"]]
            t.append("Alla Direzione Generale (Direttore Generale: %s) fanno capo %d unita' in staff, di cui "
                     "%s:" % (self.dg["capo"], len(staff), plurale(len(coord))))
            for u in coord:
                n = len(u["uffici_dipendenti"])
                t.append(self._riga(u) + (" — ne dipendono %d uffici distrettuali" % n if n else ""))
            t.append("Le altre unita' in staff, che non sono Uffici di Coordinamento:")
            t += [self._riga(u) for u in altre]
            fonte = self.dg["fonte"]
        elif op == "uffici_dg":
            # esiti di prima del 28/09 non hanno «elenca» e «ambito»: conteggio o elenco dall'unico
            # indizio che avevano, Amministrazione Centrale
            conta = esito.get("conta")
            elenca = esito.get("elenca", not conta)
            if esito.get("ambito") == "direzione generale":
                t.append("Direzione Generale (Direttore Generale: %s)." % self.dg["capo"])
                if conta:
                    t += self._conteggio_dg()
                if elenca:
                    t += self._elenco_dg()
                fonte = "%s ; %s" % (self.dg["fonte"], self.aree[0]["fonte"])
            else:
                if conta:
                    t += self._conteggio_ac()
                if elenca:
                    t += self._elenco_ac()
                fonte = "%s ; %s ; %s" % (self.rubrica_amm["url"], self.aree[0]["fonte"], self.dg["fonte"])
        elif op == "distretto":
            d = next(x for x in self.distretti if x["numero"] == k)
            n = d["conteggi"]["dipartimenti"]
            t.append("Il DISTRETTO%d comprende %s:" % (k, "1 Dipartimento" if n == 1 else "%d Dipartimenti" % n))
            t += ["- %s" % x["nome"] for x in d["dipartimenti"]]
            fonte = d["fonte"]
        t.append("Fonte: %s" % fonte)
        t.append("FINE DELLA SCHEDA")
        return "\n".join(t)


def fonti_della_scheda(testo):
    """gli URL della riga «Fonte:» di una scheda (uno o due, separati da «;»)"""
    riga = next((r for r in testo.splitlines() if r.startswith("Fonte: ")), "")
    return [u.strip() for u in riga[len("Fonte: "):].split(";") if u.strip()] or ["registro_strutture"]


def plurale(n):
    return "1 Ufficio di Coordinamento" if n == 1 else "%d Uffici di Coordinamento" % n


def percorso_veloce(domanda):
    """True se la domanda chiede solo cio' che la scheda contiene: si salta il recupero"""
    return not FUORI_SCHEDA.search(domanda or "")
