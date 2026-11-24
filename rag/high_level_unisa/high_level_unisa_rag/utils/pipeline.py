# Copied from test_baseline/diem_RAG_routered_prova_elenchi_ottimizzazioni_reranker_cronologia.py of the RAG
# repository high-level-unisa: the pipeline of the final configuration (doc/resoconto_attivita_5.pdf, sistema.sh),
# for ari_servers: relative imports, no .env (paths, models and switches from the environment set by ../rag.py
# from config/servers.yaml), the Ollama clients with base_url and num_ctx (generator_backend ollama only), the
# pieces of the LCEL chain kept on the instance (_build_pipeline, end) so that ../rag.py runs the retrieval
# without the generation, which is done by the LLM server. Everything else is unchanged.
# diem_RAG_routered_prova_elenchi_ottimizzazioni_reranker_cronologia.py (10/11/2026): COPIA di
# diem_RAG_routered_prova_elenchi_ottimizzazioni_reranker.py (invariato) con una sola differenza: la riscrittura
# della domanda avviene PRIMA dei percorsi persona e struttura, che ricevono la domanda riscritta. Senza
# cronologia la riscrittura e' saltata e il comportamento e' identico all'originale. Creata da
# crea_pipeline_cronologia_1011.py.
# diem_RAG_routered_prova_elenchi_ottimizzazioni_reranker.py (08/10/2026): COPIA DI PROVA di diem_RAG_routered_prova_elenchi.py con le varianti piu' leggere del recupero
# (ottimizzazioni_recupero.py, da tenere accanto): RERANKER_FP16, RERANKER_MAX_LEN, RERANKER_BACKEND=vllm,
# BM25_MOTORE=sparso, VECTOR_K, RERANK_CANDIDATI (13/10/2026), RERANK_CANDIDATI_ELENCHI (14/10/2026). Senza variabili si comporta come diem_RAG_routered_prova_elenchi.py. Il file originale non e' toccato.
from datetime import datetime
import json
import os
import re
import time
from pathlib import Path
import logging
from typing import Tuple, List, Dict, Any
import threading

import numpy as np
from langchain_chroma import Chroma
from sentence_transformers import CrossEncoder
from langchain_core.runnables import RunnableBranch, RunnableLambda, RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline, ChatHuggingFace
from langchain_ollama import ChatOllama
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
import chromadb
from collections import Counter, OrderedDict
from .verifica_nomi import conta_minuscole, parole_comuni, verifica_nomi
# Varianti piu' leggere di reranker e BM25 (08/10/2026, ottimizzazioni_recupero.py), tutte spente per difetto. Se il
# modulo non c'e' (copia della pipeline senza il file accanto) si resta al comportamento di prima.
try:
    from .ottimizzazioni_recupero import BM25_MOTORE, crea_bm25, crea_reranker, descrizione as _descrizione_recupero
except ImportError:
    BM25_MOTORE, crea_bm25, crea_reranker, _descrizione_recupero = "langchain", None, None, None
from .modalita_elenco import classifica

# Configure the system logger for event tracking 
logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Suppress verbose logging from third-party libraries to maintain console clarity
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("chromadb").setLevel(logging.WARNING)

# Define absolute paths for the persistent vector storage infrastructure
BASE_DIR = Path(__file__).resolve().parent
# VECTORSTORE_DIR (prova del 01/10/2026): cartella dell'indice, relativa a BASE_DIR o assoluta;
# senza la variabile e' quella in uso. Serve a misurare su una COPIA dell'indice.
PERSIST_DIRECTORY = BASE_DIR / os.getenv("VECTORSTORE_DIR", "diem_vectorstore")

# Defines the semantic domain for each vector collection to condition the routing logic
ROUTER_COLLECTION_DESCRIPTIONS = """
- collection_bandi : Contiene informazioni relative ai bandi per incarichi di insegnamento, collaborazioni con il dipartimento, personale tecnico-amministrativo, dottorati di ricerca, assegni di ricerca, borse di ricerca ed altro. 
- collection_didattica : Contiene informazioni relative all'offerta formativa del dipartimento (UNICAMENTE I CORSI DI LAUREA EROGATI DAL DIPARTIMENTO, PER I DETTAGLI SUI SINGOLI INSEGNAMENTI CONSULTARE LA COLLEZIONE PERSONALE NELLA SEZIONE DIDATTICA DEL DOCENTE), organizzazione del consiglio didattico, alternanza scuola-lavoro, contatti della segreteria per la didattica ed altri approfondimenti sulla didattica. NOTA: Informazioni dettagliate sui singoli corsi di studio (es. regolamenti, piani di studio, docenti, orari) sono invece distribuite nelle collezioni specifiche dei corsi magistrali e triennali, così come l'elenco delle aule didattiche.
- collection_personale : Contiene informazioni relative al personale docente e tecnico-amministrativo del dipartimento, inclusi corsi in cui insegnano (SCHEDA CORSO, PROGRAMMA, METODOLOGIA D'ESAME), contatti, curriculim, aree di ricerca, pubblicazioni, orari di ricevimento e collaborazioni.
- collection_ricerca : Contiene informazioni relative alle aree di ricerca del dipartimento con i relativi docenti responsabili, progetti di ricerca finanziati nel tempo e premi riconosciuti al dipartimento o ai suoi membri per attività di ricerca.
- collection_international : Contiene informazioni relative al progetto europeo erasmus+, con dettagli su convenzioni con università estere, procedure di candidatura, scadenze, borse di mobilità e contatti per l'ufficio di internazionalizzazione del DIEM.
- collection_dipartimento : Contiene informazioni generali relative al dipartimento a partire da una sua presentazione per proseguire con la descrizione dei vari organi collegiali, le commissioni, i delegati, i laboratori ed i propri responsabili. Inoltre, sempre in questa collezione sono presenti le informazioni relative alla terza missione (trasferimento tecnologico, rapporti con imprese, startup, incubatori, eventi e divulgazione scientifica) e le news più recenti che interessano il dipartimento.
- collection_corso_digital_energy_magistrale : Contiene informazioni riguardanti il corso di laurea magistrale in Electrical Engineering for Digital Energy  (ingegneria elettrica per l'energia digitale). Questa collezione include dettagli come il regolamento, programma del corso, le modalità di accesso,l'elenco degli insegnamenti (PER INFORMAZIONI COME SCHEDA, PROGRAMMA, OBIETTIVI, MODALITÀ D'ESAME CONSULTARE LA COLLEZIONE PERSONALE), le attività di ricerca correlate, i professori coinvolti e altre informazioni pertinenti al corso magistrale di Electrical Engineering for Digital Energy.
- collection_corso_digital_medicine_magistrale : Contiene informazioni riguardanti il corso di laurea magistrale in Information Engineering for Digital Medicine (ingegneria informatica per la medicina digitale). Questa collezione include dettagli come il regolamento, programma del corso, le modalità di accesso,l'elenco degli insegnamenti (PER INFORMAZIONI COME SCHEDA, PROGRAMMA, OBIETTIVI, MODALITÀ D'ESAME CONSULTARE LA COLLEZIONE PERSONALE), le attività di ricerca correlate, i professori coinvolti e altre informazioni pertinenti al corso magistrale di Information Engineering for Digital Medicine. 
- collection_corso_ingegneria_informatica_magistrale : Contiene informazioni riguardanti il corso di laurea magistrale in Ingegneria Informatica. Questa collezione include dettagli come il regolamento, programma del corso, le modalità di accesso,l'elenco degli insegnamenti (PER INFORMAZIONI COME SCHEDA, PROGRAMMA, OBIETTIVI, MODALITÀ D'ESAME CONSULTARE LA COLLEZIONE PERSONALE), le attività di ricerca correlate, i professori coinvolti e altre informazioni pertinenti al corso magistrale di Ingegneria Informatica. 
- collection_corso_ingegneria_informatica_triennale : Contiene informazioni riguardanti il corso di laurea triennale in Ingegneria Informatica. Questa collezione include dettagli come il regolamento, programma del corso, le modalità di accesso,l'elenco degli insegnamenti (PER INFORMAZIONI COME SCHEDA, PROGRAMMA, OBIETTIVI, MODALITÀ D'ESAME CONSULTARE LA COLLEZIONE PERSONALE), le attività di ricerca correlate, i professori coinvolti e altre informazioni pertinenti al corso triennale di Ingegneria Informatica.
- collection_corso_ingegneria_informazione_dottorato : Contiene informazioni riguardanti il corso di dottorato in Ingegneria dell'Informazione. Questa collezione include dettagli come il regolamento, programma del corso, le modalità di accesso,l'elenco degli insegnamenti (PER INFORMAZIONI COME SCHEDA, PROGRAMMA, OBIETTIVI, MODALITÀ D'ESAME CONSULTARE LA COLLEZIONE PERSONALE), le attività di ricerca correlate, i professori coinvolti e altre    informazioni pertinenti al corso di dottorato in Ingegneria dell'Informazione.
- collection_corso_medicina_digitale_triennale : Contiene informazioni riguardanti il corso di laurea triennale in Ingegneria dell'Informazione per la Medicina Digitale. Questa collezione include dettagli come il regolamento, programma del corso, le modalità di accesso,l'elenco degli insegnamenti (PER INFORMAZIONI COME SCHEDA, PROGRAMMA, OBIETTIVI, MODALITÀ D'ESAME CONSULTARE LA COLLEZIONE PERSONALE), le attività di ricerca correlate, i professori coinvolti e altre informazioni pertinenti al corso triennale di Ingegneria dell'Informazione per la Medicina Digitale.
- collection_corso_photovoltaics_dottorato : Contiene informazioni riguardanti il corso di dottorato in photovoltaics. Questa collezione include dettagli come il regolamento, programma del corso, le modalità di accesso,l'elenco degli insegnamenti (PER INFORMAZIONI COME SCHEDA, PROGRAMMA, OBIETTIVI, MODALITÀ D'ESAME CONSULTARE LA COLLEZIONE PERSONALE), le attività di ricerca correlate, i professori coinvolti e altre informazioni pertinenti al corso di dottorato in photovoltaics.
- collection_statuto_e_normativa : Contiene lo statuto e la normativa ufficiale dell'Università degli Studi di Salerno (regolamenti, decreti, atti normativi, delibere), a livello di intero Ateneo. USA QUESTA COLLEZIONE per domande sull'Ateneo che riguardano norme, regolamenti o atti ufficiali.
- collection_terza_missione : Contiene informazioni sulla terza missione a livello di intero Ateneo (non specifica del dipartimento DIEM): trasferimento tecnologico, rapporti con imprese, startup, incubatori, eventi e divulgazione scientifica.
- collection_percorsi_di_studio : Contiene informazioni su immatricolazioni, dottorati, master e scuole di specializzazione a livello di intero Ateneo (requisiti di accesso, scadenze, bandi, modalità).
- collection_ateneo_generale : Contiene informazioni generali sull'Università di Salerno a livello di intero Ateneo (non specifiche del dipartimento DIEM). Copre: organi di governo, contatti istituzionali, amministrazione centrale, storia dell'ateneo, sedi e campus, biblioteche, trasporti, residenze e mensa, assistenza per studenti con disabilità o DSA, centri di ricerca e numeri della ricerca a livello di ateneo, orientamento in ingresso, elenco generale dei corsi di studio di tutto l'ateneo, servizio di job placement, programmi international/Erasmus a livello di ateneo, tasse e contributi, domanda di laurea, diritto allo studio (borse di studio, ADISURC) e servizi online per gli studenti (es. Esse3). USA QUESTA COLLEZIONE per domande generali sull'Ateneo (incluse persone, ruoli, organi di governo) che non riguardano specificamente il dipartimento DIEM.
IMPORTANTE: LE INFORMAZIONE RELATIVE A POSIZIONE E CAPIENZA DELLE AULE DIDATTICHE SONO PRESENTI NELLE COLLEZIONI DEI SINGOLI CORSI DI LAUREA.
IMPORTANTE 2: CONSIDERA I DOCUMENTI PIÙ RECENTI COME PIÙ RILEVANTI IN CASO DI INFORMAZIONI CONTRADDITORIE (es. REQUISITI DI IMMATRICOLAZIONE, VOTO DI LAUREA).
"""


HF_TOKEN = os.getenv("HF_TOKEN")
HF_MODEL_ID = "Qwen/Qwen3-30B-A3B-Instruct-2507"

# Spike solution (sezione 16, scadenza venerdì): LLM_BACKEND="ollama" usa un modello già
# disponibile in locale sulla DGX Spark invece di scaricare Qwen3-30B-A3B (il modello della
# tesi). Passare a LLM_BACKEND=hf per tornare al modello della tesi.
#
# Default cambiato da llama3.1:8b a qwen3.6:35b-a3b il 2026-07-22, dopo un confronto empirico
# mirato su 9 domande problematiche: Qwen ha eliminato entrambi i casi di risposta sicura ma
# non ancorata (llama3.1:8b rispondeva con sicurezza anche quando le fonti recuperate erano
# palesemente irrilevanti), grazie soprattutto a una riscrittura della query più efficace che
# migliora tutta la catena di retrieval a valle. ATTENZIONE: qwen3.6:35b-a3b non è stato
# scaricato da questo progetto - appartiene a un altro utente del laboratorio (operatore_rag_2)
# sulla Spark condivisa. Verificare con `ollama list` che sia ancora presente prima di ogni
# esecuzione lunga; se non c'è più, tornare a llama3.1:8b (validato e comunque disponibile).
LLM_BACKEND = os.getenv("LLM_BACKEND", "ollama")
OLLAMA_MODEL_ID = os.getenv("OLLAMA_MODEL_ID", "qwen3.6:35b-a3b")
# ari_servers: the Ollama server of ollama/docker/run.bash (generator_backend ollama)
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11500")
# Il modello supporta il "thinking" (ragionamento nascosto prima della risposta finale),
# non toccato finora - resta al default di Ollama per questo tag (verosimilmente attivo).
# Configurabile per testare l'impatto sui tempi di risposta senza cambiare il default
# usato per tutti i test già documentati.
OLLAMA_ENABLE_THINKING = os.getenv("OLLAMA_ENABLE_THINKING", "true").lower() == "true" # AV: Ho notato che ho dovuto mettere a false il valore letto di default se no partiva il reasoning

# Backend vLLM (dal 07/10/2026): LLM_BACKEND=vllm manda le richieste al server vLLM nel container
# Docker "vllm-rag" (vedi avvia_vllm.sh), con API compatibile OpenAI. Stesso modello di partenza
# (Qwen3.6-35B-A3B, quantizzazione NVFP4 invece della Q4 di Ollama). Il thinking segue la stessa
# variabile OLLAMA_ENABLE_THINKING, cosi' i due backend si confrontano a parita' di impostazioni.
# Con vLLM num_ctx/num_batch non servono: il contesto massimo e' quello del server (--max-model-len)
# e le richieste piu' lunghe vengono rifiutate con un errore, non tagliate in silenzio.
VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://127.0.0.1:8001/v1")
VLLM_MODEL_ID = os.getenv("VLLM_MODEL_ID", "qwen3.6-nvfp4")

# Contextual compression post-rerank (una delle 5 aggiunte di qualità dell'architettura v2,
# anticipata qui dietro flag, disattivata di default per non introdurre rischio prima della
# consegna di venerdì). Filtra i chunk già rerankati per similarità coseno con la query,
# implementata a mano perché EmbeddingsFilter non esiste più nella versione di langchain
# installata su questa macchina (langchain 1.3.0 / langchain-community 0.4.1).
# Router semantico. Disattivato di default dal 2026-08-28: con 4 collezioni misurato in perdita
# netta (su 36 domande fallite il chunk-verita' arriva nel top-7 in 21 casi interrogando tutte le
# collezioni, contro 13 con router e riscrittura attivi). Interrogare tutte costa 2,44s medi contro
# i ~3s di router+retrieval+rerank, quindi non peggiora la latenza. Rimettere a "true" quando le
# collezioni saranno molte (deployment per-dipartimento): la logica del router resta intatta.
ENABLE_SEMANTIC_ROUTER = os.getenv("ENABLE_SEMANTIC_ROUTER", "false").lower() in ("1", "true", "yes")

# Taglio del re-ranking. Alzato da 7 a 10 insieme alla disattivazione del router: senza router il
# pool di candidati passa da ~40 a ~135 documenti, e con un taglio fisso a 7 la diluizione faceva
# perdere 3 domande che prima arrivavano al modello.
RERANK_TOP_N = int(os.getenv("RERANK_TOP_N", "10"))

# Filtro di pertinenza sui documenti gia' rerankati, nella versione che non costa nulla: il
# re-ranker un punteggio lo calcola comunque, e chi sta sotto la soglia viene scartato invece di
# finire nel contesto. E' l'alternativa gratuita al filtro proposto dai relatori, che chiederebbe
# al modello un giudizio per ogni pezzo, cioe' dieci chiamate in piu' per domanda.
# Misura preliminare su 6.300 pezzi (630 domande): i pezzi che contengono davvero il dato hanno
# punteggio mediano 0,979 contro 0,784 di quelli che non lo contengono, ma le code si sovrappongono
# molto. A 0,5 si toglierebbe il 27% del contesto mettendo a rischio 5 risposte oggi corrette.
# Zero, il valore predefinito, disattiva il filtro. Se tutti i documenti stanno sotto la soglia se
# ne tiene comunque uno: il contesto vuoto deve restare il segnale di un recupero fallito davvero.
RERANK_MIN_SCORE = float(os.getenv("RERANK_MIN_SCORE", "0"))

# Filtro di pertinenza fatto dal modello, nella forma proposta dai relatori: una domanda per ogni
# pezzo recuperato, tutte inviate insieme. Costo misurato: 1,42 secondi per dieci pezzi contro i
# 4,71 che costerebbero una dopo l'altra. Spento di default finche' non e' misurato in accuratezza.
ENABLE_CHUNK_FILTER = os.getenv("ENABLE_CHUNK_FILTER", "false").lower() in ("1", "true", "yes")
CHUNK_FILTER_CONCURRENCY = int(os.getenv("CHUNK_FILTER_CONCURRENCY", "10"))
CHUNK_FILTER_MAX_CHARS = int(os.getenv("CHUNK_FILTER_MAX_CHARS", "6000"))

PROMPT_FILTRO_PEZZI = (
    "DOMANDA: {domanda}\n\n"
    "TESTO:\n{testo}\n\n"
    "Il TESTO contiene informazioni utili a rispondere alla DOMANDA, anche solo in parte? "
    "Rispondi con una sola parola: SI oppure NO."
)
_PERTINENTE = re.compile(r"\bs[i\u00ec]\b", re.I)
_NON_PERTINENTE = re.compile(r"\bno\b", re.I)

# Come si decide cosa tenere. «selettivo» tiene solo i pezzi per cui il modello dice si', ed e' il
# comportamento misurato nella sezione 3.31, che si e' rivelato neutro sull'accuratezza e costoso
# in completezza. «prudente» tiene tutto tranne cio' che il modello dichiara chiaramente
# irrilevante: stesso costo, ma il dubbio va a favore della conservazione.
CHUNK_FILTER_MODE = os.getenv("CHUNK_FILTER_MODE", "selettivo").strip().lower()

# Ordine dei pezzi nel contesto. Il prompt mette la domanda DOPO il contesto, quindi l'ultimo pezzo
# e' quello piu' vicino alla domanda. «punteggio» ordina dal piu' al meno pertinente e mette quindi
# il peggiore in quella posizione, «coda» mette il migliore per ultimo, «estremi» distribuisce i
# migliori alle due estremita' lasciando i peggiori nel mezzo. Nessuna delle tre toglie
# informazione: cambia solo la posizione.
#
# Default portato a «coda» il 2026-09-15, su richiesta dei relatori. Il saldo complessivo NON e' un
# miglioramento: piu' 2 domande su 613 con p = 0,888, cioe' indistinguibile da zero. Cio' che la
# misura mostra e' una ridistribuzione, e su quella si e' deciso: «coda» recupera 26 delle 91
# domande oggi fallite (il 29%, l'effetto su categoria piu' grande misurato in tutta la serie) e ne
# rovina 24 delle 522 che oggi funzionano (il 4,6%). Adottarlo significa preferire il primo piatto
# al secondo, non aspettarsi un guadagno netto.
#
# Il valore si replica: il 07/09 sul banco da 630 dava +26/-22, il 15/09 sul banco da 613 da'
# +26/-24. Ventisei recuperate in entrambe le misure, a otto giorni e due banchi di distanza.
# «estremi», rigiudicato nella stessa sessione, da' +20/-20, cioe' esattamente zero.
CONTEXT_ORDER = os.getenv("CONTEXT_ORDER", "coda").strip().lower()
# Pavimento del filtro: quanti documenti tenere comunque, anche se sotto soglia. La diagnostica
# dice che il pezzo che contiene il dato sta entro la terza posizione in 350 casi su 389, quindi
# un pavimento a 3 costa quasi nulla e attenua il crollo da dieci pezzi a uno osservato sulle
# domande in cui il recupero va male.
RERANK_MIN_KEEP = max(1, int(os.getenv("RERANK_MIN_KEEP", "1")))

# Query rewriting: saltata quando la cronologia di conversazione e' vuota. Il suo scopo primario e'
# risolvere i riferimenti anaforici usando i turni precedenti; al primo turno non ha nulla da
# risolvere e resta solo l'arricchimento semantico, misurato in perdita netta (-3 domande su 36).
SKIP_REWRITE_WHEN_NO_HISTORY = os.getenv("SKIP_REWRITE_WHEN_NO_HISTORY", "true").lower() in ("1", "true", "yes")

# Tokenizzazione del ramo lessicale. Il default di BM25Retriever e' text.split(): non abbassa
# le maiuscole e non stacca la punteggiatura, quindi "Pinto" (domanda) e "PINTO" (pagina dei
# delegati) sono token diversi, e "Salerno?" non corrisponde a "Salerno". Rimettere a "false"
# per tornare al comportamento precedente.
BM25_NORMALIZE_TOKENS = os.getenv("BM25_NORMALIZE_TOKENS", "true").lower() in ("1", "true", "yes")

# Profondita' del solo ramo lessicale (quello semantico resta a 20). BM25 gira su CPU, quindi il
# costo non e' la ricerca ma il gruppo di candidati piu' grande da re-rankare. Confronto misurato
# sulle stesse 630 domande: con 50 il successo e' 80,8% e la latenza media 10,29 s, con 30 sono
# 79,7% e 9,22 s (il punto di partenza era 78,3% e 8,91 s). La differenza fra 50 e 30 non e'
# significativa (McNemar p=0,30) e il guadagno sulla categoria dei delegati regge in entrambi i
# casi; con 30 sparisce anche la regressione sui professori emeriti. Da qui la scelta di 30.
BM25_K = int(os.getenv("BM25_K", "30"))
# Profondita' del ramo vettoriale per collezione (08/10/2026: prima fisso a 20 nel codice, ora regolabile per provare
# meno candidati al reranker; il default resta 20).
VECTOR_K = int(os.getenv("VECTOR_K", "20"))
# Candidati passati al reranker (13/10/2026): per ogni collezione si tengono solo i primi RERANK_CANDIDATI della lista
# fusa (ricerca vettoriale + BM25, nell'ordine della reciprocal rank fusion), invece di tutti (~46 per collezione,
# ~185 a domanda). 0 (predefinito) = tutti, come prima.
# 14/10/2026: per le domande di ELENCO (modalita' elenco) vale invece RERANK_CANDIDATI_ELENCHI, predefinito 0 = tutti:
# un elenco sparso su molti documenti ha bisogno anche dei candidati "di coda", e alla risposta ne passano 20, non 10.
RERANK_CANDIDATI = int(os.getenv("RERANK_CANDIDATI", "0") or 0)
RERANK_CANDIDATI_ELENCHI = int(os.getenv("RERANK_CANDIDATI_ELENCHI", "0") or 0)


def _candidati(retriever, testo, limite=None):
    """i documenti di una collezione per una ricerca, limitati ai primi 'limite' (RERANK_CANDIDATI se non indicato;
    0 = tutti)"""
    limite = RERANK_CANDIDATI if limite is None else limite
    docs = retriever.invoke(testo)
    return docs[:limite] if limite > 0 else docs

# Modalita' del filtro di dominio.
#   "permissivo" (predefinita): resta attivo il solo pre-filtro a regole sui saluti; tutto il
#     resto prosegue verso il recupero, che ha gia' un blocco deterministico quando non trova
#     documenti. Scelta sulla base di una misura: delle 7 domande legittime che il filtro LLM
#     respingeva, 5 diventano corrette; e su 15 sonde palesemente fuori tema il sistema rifiuta
#     onestamente in tutti e 15 i casi, senza inventare nulla. Il prezzo e' la latenza sui
#     messaggi fuori tema, 8,95 secondi contro 1,91, perche' attraversano l'intera pipeline.
#   "llm": il comportamento precedente, con la classificazione PERTINENTE/CONVENEVOLI/
#     FUORI_DOMINIO fatta dal modello. Da rimettere se il bot viene esposto a un pubblico
#     ampio, dove il costo di elaborare messaggi fuori tema conta piu' della loro rarita'.
#   "off": nessun filtro, nemmeno sui saluti.
DOMAIN_GUARDRAIL_MODE = os.getenv("DOMAIN_GUARDRAIL_MODE", "permissivo").strip().lower()

# Espansione a livello di pagina. Quando almeno PAGE_EXPANSION_MIN_CHUNKS fra i chunk
# selezionati vengono dalla stessa pagina, al modello si passa l'intera pagina invece dei
# singoli pezzi: serve agli elenchi lunghi, dove il re-ranking pesca voci sparse dello stesso
# elenco e la risposta esce incompleta. I due tetti evitano che una pagina enorme saturi il
# contesto: oltre PAGE_EXPANSION_MAX_CHARS la pagina non viene espansa, e oltre
# PAGE_EXPANSION_MAX_CONTEXT si smette di espandere per quella domanda.
# Disattivata: la prova sulle 68 risposte parziali (vedi sezione 3.23) mostra che rende le
# risposte piu' decise ma non piu' giuste - corrette piene da 14 a 20, errate da 6 a 10,
# nessuna delle due differenze significativa - a fronte di 3 secondi in piu' per domanda.
# Il meccanismo resta disponibile per riprovarlo dopo aver sistemato le risposte di
# riferimento del benchmark, che oggi non distinguono in modo stabile CORRETTA da PARZIALE.
ENABLE_PAGE_EXPANSION = os.getenv("ENABLE_PAGE_EXPANSION", "false").lower() in ("1", "true", "yes")

# Pipeline dedicata alle domande su persone: classificatore a regole piu' ricerca esatta
# sull'anagrafe costruita in fase di ingestione (crea con costruisci_anagrafe.py).
#
# La prima versione sostituiva il recupero con la scheda ogni volta che il nome si risolveva, e
# sulle 630 domande e' costata 2,9 punti (79,0% contro 81,9%): la rubrica dice a quale
# dipartimento una persona appartiene, non quale dirige, e senza recupero il modello negava
# fatti veri («Capacchione non dirige alcun dipartimento»). Le 10 domande guadagnate erano tutte
# del tipo «quale ruolo ricopre X», dove l'anagrafe aggrega incarichi sparsi su pagine diverse.
#
# Da qui la divisione in due meccanismi indipendenti:
#   PERSON_FAST_PATH     salta il recupero, ma solo se la domanda chiede un campo che quella
#                        scheda contiene davvero (email, interno, telefono, matricola, stanza);
#   PERSON_CONTEXT_BOOST mette la scheda in testa al contesto recuperato in tutti gli altri casi,
#                        senza togliere niente.
#
# Misura della versione divisa (fase 3b, 630 domande, 03/09/2026, trascrizioni giudicate una di
# seguito all'altra): 83,5% contro 81,9%. Il totale non e' significativo da solo (McNemar
# p=0,12) e va letto scomposto, perche' meta' del guadagno non dipende da questa modifica:
#   - 183 domande con risposta identica nei due giri: 183 verdetti uguali su 183, il giudice
#     consecutivo non introduce rumore;
#   - 145 domande toccate dalla pipeline: +5 (8 migliorate, 3 peggiorate);
#   - 302 domande NON toccate ma con testo diverso lo stesso: +5, cioe' la non riproducibilita'
#     del generatore (modello a esperti selezionati su GPU, gia' a temperatura zero).
# Il risultato solido e' sulla categoria bersaglio, delegati_referenti: da 64,0% a 82,0%, nove
# domande migliorate e nessuna peggiorata, McNemar p=0,0077. Restano tre casi noti: due schede
# che impongono il titolo al maschile («Professore Emerito» invece di «Professoressa Emerita»)
# e una che elenca due interni dove la risposta di riferimento ne indica uno solo.
# La regola 6 del prompt di generazione («Oggi e' il ...») inserisce la data odierna. In esercizio
# serve, ma rende non confrontabili due giri di prova eseguiti in giorni diversi: il prompt cambia,
# e con esso una parte delle risposte. Misurato: su 466 domande non toccate da una modifica, 302
# cambiano testo fra un giorno e l'altro pur avendo recuperato gli stessi identici documenti,
# mentre il modello ripete se stesso alla perfezione all'interno della stessa giornata (20 su 20,
# anche dopo averlo scaricato e ricaricato in memoria). Impostando BENCHMARK_DATE, nel formato
# gg/mm/aaaa, la data viene congelata e due giri qualsiasi ricevono prompt identici. Lasciata
# vuota - il caso normale - si usa la data reale.
BENCHMARK_DATE = os.getenv("BENCHMARK_DATE", "").strip()

# Parametri di Ollama che il client LangChain non espone (num_batch) o che conviene fissare
# esplicitamente (num_ctx). Zero, il valore predefinito, lascia decidere il server: oggi sceglie
# una finestra da 262.144 token, mentre il nostro prompt piu' lungo mai osservato ne usa 17.953.
OLLAMA_NUM_BATCH = int(os.getenv("OLLAMA_NUM_BATCH", "0"))
OLLAMA_NUM_CTX = int(os.getenv("OLLAMA_NUM_CTX", "0"))
ENABLE_PERSON_PIPELINE = os.getenv("ENABLE_PERSON_PIPELINE", "true").lower() in ("1", "true", "yes")
PERSON_FAST_PATH = os.getenv("PERSON_FAST_PATH", "true").lower() in ("1", "true", "yes")
PERSON_CONTEXT_BOOST = os.getenv("PERSON_CONTEXT_BOOST", "true").lower() in ("1", "true", "yes")
ANAGRAFE_PATH = Path(os.getenv("ANAGRAFE_PATH", str(Path(__file__).resolve().parent / "anagrafe.json")))
# Percorso «struttura» (25/09/2026): domande che chiedono di ELENCARE o CONTARE i componenti di
# una struttura amministrativa - aree e loro uffici, uffici che dipendono da un coordinamento,
# personale di un ufficio, unita' in staff alla Direzione Generale, dipartimenti di un distretto -
# servite da un registro costruito in ingestione (costruisci_registro_strutture.py, solo dalle
# fonti) invece che dalla ricerca, che quegli elenchi non li riporta interi. Stesso meccanismo del
# percorso persona, che ha la precedenza: riconoscimento a regole (struttura_lookup.py), veloce
# solo se la domanda chiede cio' che la scheda contiene, altrimenti scheda in testa al contesto.
# ACCESO per difetto dal 25/09/2026, dopo la misura sulle 429 domande mai viste: sulle 34
# intercettate riuscite da 16 a 29 (corrette da 8 a 25), saldo passa/non passa +13, p = 0,001,
# tre letture del giudice a bracci alternati; tutte le altre domande identiche a interruttore
# spento (verificato su 30 domande: fonti e contesto identici). ENABLE_STRUCT_PATH=false lo spegne.
# Modalita' elenco (30/09/2026, modalita_elenco.py): per le domande di elenco di entita' il recupero
# usa, oltre alla domanda, 3 ricerche orientate alla FONTE scritte dal modello (il documento in cui
# l'elenco e' scritto per intero); i candidati di tutte le ricerche, senza doppioni, si riordinano
# rispetto alla domanda e se ne passano LIST_MODE_TOP_N. Stima del 30/09 (stima_multiquery.py, solo
# recupero): voci degli elenchi nel contesto 86,2% -> 92,9% sulle 613, 87,0% -> 91,2% sulle 429.
# Spento per difetto: a interruttore spento il recupero e' identico a prima.
ENABLE_LIST_MODE = os.getenv("ENABLE_LIST_MODE", "false").lower() in ("1", "true", "yes")
LIST_MODE_TOP_N = int(os.getenv("LIST_MODE_TOP_N", "20"))
# Verifica dei nomi (02/10/2026, verifica_nomi.py): dopo la generazione, un nome della risposta che non compare
# nel contesto e ha nel contesto un solo nome quasi identico (Egizio -> EGIZIANO, Piemento -> Piemonte) prende la
# grafia del contesto; un nome spezzato (De Stefanis) prende la forma del contesto (DESTEFANIS). Deterministica,
# nessuna chiamata al modello. Misura del 02/10 sulle 207 domande della misura degli elenchi, risposta grezza e
# corretta dalla stessa generazione: riuscite +7/-0 e corrette +9/-0 sulle risposte cambiate, tempi invariati.
# ACCESA per difetto dal 05/10/2026; ENABLE_NAME_CHECK=false la spegne.
ENABLE_NAME_CHECK = os.getenv("ENABLE_NAME_CHECK", "true").lower() in ("1", "true", "yes")
# Ragionamento mirato (05/10/2026): il ragionamento del modello, spento dal luglio 2026 per la latenza (sezione
# 3.5: 76 s contro 7,6 di media, ma -8,6 punti), si accende solo per le domande dei tipi in RAGIONAMENTO_TIPI
# secondo classifica() di modalita_elenco.py (PUNTUALE, INSIEME, SPIEGAZIONE, NON CLASSIFICATA), con un modello
# dedicato e un tetto di token piu' alto, perche' il ragionamento non consumi la risposta. Il testo del
# ragionamento non entra nella risposta. Spento per difetto.
ENABLE_RAGIONAMENTO_MIRATO = os.getenv("ENABLE_RAGIONAMENTO_MIRATO", "false").lower() in ("1", "true", "yes")
RAGIONAMENTO_TIPI = {s.strip().upper() for s in os.getenv("RAGIONAMENTO_TIPI", "INSIEME").split(",") if s.strip()}
RAGIONAMENTO_NUM_PREDICT = int(os.getenv("RAGIONAMENTO_NUM_PREDICT", "16384"))
ENABLE_STRUCT_PATH = os.getenv("ENABLE_STRUCT_PATH", "true").lower() in ("1", "true", "yes")
# Dati di UNA struttura (07/10/2026, struttura_lookup.py): capo ed email di uffici, aree e Direzione Generale,
# distretto di un dipartimento, con la scheda del registro in testa al contesto recuperato (mai il percorso
# veloce); si intercetta solo cio' che il registro sa. ACCESO per difetto dal 07/10/2026, dopo la misura del
# 06-07/10 (ibrido, tre letture a bracci alternati): sulle 148 domande dei banchi intercettate riuscite da 132 a
# 145 (+13/-0, p < 0,001; sulle 429 +9/-0, p = 0,004), 30 domande di controllo con risposte identiche, 44 domande
# nuove su strutture mai nominate nei banchi con il dato giusto da 42 a 44. ENABLE_STRUCT_ATTRIBUTES=false lo spegne.
ENABLE_STRUCT_ATTRIBUTES = os.getenv("ENABLE_STRUCT_ATTRIBUTES", "true").lower() in ("1", "true", "yes")
REGISTRO_STRUTTURE_PATH = Path(os.getenv("REGISTRO_STRUTTURE_PATH",
                                         str(Path(__file__).resolve().parent / "registro_strutture.json")))
PAGE_EXPANSION_MIN_CHUNKS = int(os.getenv("PAGE_EXPANSION_MIN_CHUNKS", "2"))
PAGE_EXPANSION_MAX_CHARS = int(os.getenv("PAGE_EXPANSION_MAX_CHARS", "24000"))
PAGE_EXPANSION_MAX_CONTEXT = int(os.getenv("PAGE_EXPANSION_MAX_CONTEXT", "60000"))

# Espansione per vicinato. Quando il documento intero supera PAGE_EXPANSION_MAX_CHARS, invece di
# rinunciare si passano i genitori adiacenti a quelli scelti dal re-ranker. Serve la mappa delle
# posizioni (posizioni_parent.json, prodotta da costruisci_posizioni.py): `parent_id` e' un hash
# del contenuto e non dice l'ordine. Misura del 22/09: senza questo, il 54% delle pagine candidate
# non si espande, e sui conteggi due su tre. Raggio 1 perche' tre genitori stanno sotto il tetto
# anche al nono decile (14.859 caratteri) mentre cinque lo sforerebbero (24.765).
ENABLE_VICINATO = os.getenv("ENABLE_VICINATO", "false").lower() in ("1", "true", "yes")
VICINATO_RAGGIO = int(os.getenv("VICINATO_RAGGIO", "1"))
POSIZIONI_PATH = Path(os.getenv("POSIZIONI_PATH",
                                str(Path(__file__).resolve().parent / "posizioni_parent.json")))


# Parole grammaticali italiane, tolte da query e documenti prima del calcolo BM25. Senza questo
# filtro il punteggio lessicale e' dominato dalle parole comuni della domanda («quale», «ruolo»,
# «presso», «di»), che compaiono ovunque nel corpus: il chunk giusto della pagina dei delegati
# passa dalla posizione 127 alla 18 semplicemente togliendole. La lista contiene solo parole
# grammaticali, non termini di dominio: «Salerno» e «Ateneo» restano, perche' distinguono
# davvero in domande che citano altri atenei.
_BM25_STOPWORD = set("""a ad ai al alla alle allo agli anche c che chi ci coi col come con cui da
dal dalla dalle dallo dagli degli dei del della delle dello di e ed gli ha hanno ho i il in la le
lo loro ma me mi ne negli nei nel nella nelle nello non o per presso qual quale quali quanta
quante quanti quanto quella quelle quelli quello questa queste questi questo si sono sta stata
stato su sui sul sulla sulle sullo tra fra un una uno vi essere viene vengono era erano fu furono
sua suo sue suoi piu solo quando dove cosa perche""".split())


def _bm25_tokenizer(text: str) -> list:
    """Tokenizzazione lessicale normalizzata: minuscolo, sole sequenze di caratteri di parola,
    senza parole grammaticali. Sostituisce il default di LangChain (text.split()), che lascia
    maiuscole e punteggiatura attaccate — rendendo invisibili a BM25 i cognomi scritti in
    maiuscolo e l'ultima parola di qualunque domanda che finisca con un punto interrogativo — e
    che lascia pesare le parole comuni piu' dei termini distintivi."""
    return [w for w in re.findall(r"\w+", text.lower()) if w not in _BM25_STOPWORD]

ENABLE_CONTEXTUAL_COMPRESSION = os.getenv("ENABLE_CONTEXTUAL_COMPRESSION", "false").lower() in ("1", "true", "yes")

# Le fonti dell'ateneo chiamano la stessa carica con parole diverse: la pagina delle aree scrive
# «Dirigente» dove le domande dicono «Capo Area», e la rubrica scrive «Direttore vicario di
# dipartimento» dove le domande dicono «Vicario». Questa regola dichiara le equivalenze al
# generatore. La misura del tetto (tetto_sinonimi2.py) prevede che non serva: le domande che
# nominano una carica stanno gia' sopra la media, e il rischio e' di confondere il «Capo» di un
# ufficio con il «Responsabile del procedimento», che e' un ruolo diverso. Resta spenta finche'
# una misura non dice il contrario.
PROMPT_SINONIMI_CARICA = os.getenv("PROMPT_SINONIMI_CARICA", "false").lower() in ("1", "true", "yes")

# Apertura e regole 2 e 4 del prompt di generazione, riscritte perche' oggi si contraddicono:
# l'apertura chiede risposte «esaustive», la 2 vieta la verbosita', la 4 chiede di essere
# «completo ma coinciso», «discorsive», e di «sintetizzare il contenuto se troppo esteso» -
# che e' anche l'istruzione che autorizza a troncare un elenco lungo. Misura del 18/09 sulle
# 613: le PARZIALI coprono il 64% del contenuto atteso contro l'86% delle corrette, sono per
# il 71% materiale estraneo alla risposta attesa e lunghe 2,3 volte. Il problema non e' il
# troncamento ma la selezione. Spento per difetto: da accendere solo per la misura.
PROMPT_COPERTURA = os.getenv("PROMPT_COPERTURA", "false").lower() in ("1", "true", "yes")

APERTURA_RISPOSTA = (
    "Fornisci risposte chiare, professionali e pertinenti alle domande degli utenti su tutti gli aspetti dell'Ateneo (dipartimenti come il DIEM, organi di governo, corsi di studio, servizi agli studenti, ricerca, regolamenti e vita universitaria).\n\n"
    if PROMPT_COPERTURA else
    "Fornisci risposte chiare, professionali ed esaustive alle domande degli utenti su tutti gli aspetti dell'Ateneo (dipartimenti come il DIEM, organi di governo, corsi di studio, servizi agli studenti, ricerca, regolamenti e vita universitaria).\n\n")

REGOLA_FOCUS = (
    "2. FOCUS ASSOLUTO: Rispondi ESCLUSIVAMENTE a cio' che la DOMANDA UTENTE chiede. E' severamente vietato aggiungere paragrafi di approfondimento non richiesti, divagare o riprendere argomenti passati. Il divieto vale ANCHE per materiale che riguarda l'argomento giusto: se il CONTESTO contiene informazioni correlate che la domanda non ha chiesto, lasciale fuori. Ogni frase della risposta deve rispondere a una parte della domanda; se una frase non risponde a nulla di cio' che e' stato chiesto, non scriverla.\n"
    if PROMPT_COPERTURA else
    "2. FOCUS ASSOLUTO (ANTI-VERBOSITÀ): Rispondi ESCLUSIVAMENTE alla domanda posta nell'ultimo turno. È severamente vietato aggiungere paragrafi di approfondimento non richiesti, divagare o riprendere argomenti passati.\n")

REGOLA_COPERTURA = (
    "4. COPERTURA DELLA DOMANDA: Prima di rispondere, individua tutti gli elementi che la DOMANDA UTENTE chiede - possono essere piu' di uno (un nome e un recapito, un ruolo e la struttura, una definizione e il suo scopo, un atto e chi lo emana) - e rispondi a ciascuno di essi. Se la domanda chiede un insieme («quali», «elenca», «chi sono», «da chi e' composto», «quanti»), riporta TUTTE le voci presenti nel CONTESTO, non un campione e non solo le principali: un elenco incompleto e' una risposta sbagliata. Non accorciare e non riassumere un elenco perche' e' lungo. Se invece la domanda chiede un dato singolo, fornisci quel dato e fermati.\n"
    if PROMPT_COPERTURA else
    "4. COMPLETEZZA: Fornisci risposte discorsive. Devi essere completo ma coinciso, riportando ove strettamente necessario elenchi, procedure, orari o scadenze utilizzando formattazioni chiare, ma sintetizzando il contenuto se troppo esteso, in maniera funzionale.\n")

REGOLA_INSUFFICIENTE = (
    "7. GESTIONE DEL CONTESTO INSUFFICIENTE: Se il CONTESTO fornito è vuoto o non contiene alcuna informazione pertinente alla DOMANDA UTENTE, NON INVENTARE MAI una risposta plausibile per colmare il vuoto. Ammetti onestamente la mancanza di informazioni, ad esempio: 'Al momento non dispongo di informazioni su questo argomento specifico.' Questa regola vale ANCHE quando il CONTESTO riguarda il giusto argomento generale ma non riporta esplicitamente il dato puntuale richiesto (un nome, una data, un numero, un indirizzo, un ruolo specifico): in quel caso NON dedurre, stimare o generalizzare a partire da informazioni simili o correlate solo perché vertono sull'argomento giusto - dichiara che il dato specifico richiesto non è disponibile nel materiale a tua disposizione, anche se il CONTESTO contiene informazioni correlate sull'argomento generale. Questa regola ha priorità assoluta sulla regola 4 (COPERTURA DELLA DOMANDA): un'ammissione di incertezza è sempre preferibile a un'informazione inventata o dedotta.\n"
    if PROMPT_COPERTURA else
    "7. GESTIONE DEL CONTESTO INSUFFICIENTE: Se il CONTESTO fornito è vuoto o non contiene alcuna informazione pertinente alla DOMANDA UTENTE, NON INVENTARE MAI una risposta plausibile per colmare il vuoto. Ammetti onestamente la mancanza di informazioni, ad esempio: 'Al momento non dispongo di informazioni su questo argomento specifico.' Questa regola vale ANCHE quando il CONTESTO riguarda il giusto argomento generale ma non riporta esplicitamente il dato puntuale richiesto (un nome, una data, un numero, un indirizzo, un ruolo specifico): in quel caso NON dedurre, stimare o generalizzare a partire da informazioni simili o correlate solo perché vertono sull'argomento giusto - dichiara che il dato specifico richiesto non è disponibile nel materiale a tua disposizione, anche se il CONTESTO contiene informazioni correlate sull'argomento generale. Questa regola ha priorità assoluta sulla regola 4 (COMPLETEZZA): un'ammissione di incertezza è sempre preferibile a un'informazione inventata o dedotta.\n")

# Riparazione del recupero quando la risposta e' un rifiuto (si veda la sezione 3.10 per la
# versione originale del meccanismo, e il commento in testa a patch_riparazione.py per cosa
# cambia). Spenta: va misurata prima di essere adottata.
ENABLE_RIPARAZIONE = os.getenv("ENABLE_RIPARAZIONE", "false").lower() in ("1", "true", "yes")
RIPARAZIONE_MODO = os.getenv("RIPARAZIONE_MODO", "verifica").strip().lower()
RIPARAZIONE_INNESCO = os.getenv("RIPARAZIONE_INNESCO", "rifiuto").strip().lower()

# Riassunto dei pezzi (proposta dei relatori dell'11 settembre). Spento: va misurato.
#   «tutti»    raggruppa i pezzi per documento e ne estrae il contenuto utile; al generatore
#              arrivano solo gli estratti;
#   «scartati» tiene interi i pezzi che il filtro promuove e riassume solo quelli che
#              butterebbe.
RIASSUNTO_MODO = os.getenv("RIASSUNTO_MODO", "off").strip().lower()

# Potatura per frase con XProvence (Chirkova et al., ICLR 2025), variante multilingue
# costruita sullo stesso bge-reranker-v2-m3 che usiamo come re-ranker. A differenza del
# filtro, che decide sul pezzo intero, qui il modello assegna un punteggio a ogni token, lo
# media per frase e tiene le frasi sopra soglia: la riga con il dato sopravvive anche se il
# resto del pezzo viene scartato.
#
# Spento di default. La misura del tetto del 15/09 (sezione 3.43) dice che alla soglia 0,1
# conserva il 98,3% delle ancore togliendo pero' solo il 9% del contesto, e che alzando la
# soglia il danno supera quello del filtro gia' rifiutato. Resta acceso da variabile
# d'ambiente per il giro di verifica end-to-end.
ENABLE_XPROVENCE = os.getenv("ENABLE_XPROVENCE", "false").lower() in ("1", "true", "yes")
XPROVENCE_SOGLIA = float(os.getenv("XPROVENCE_SOGLIA", "0.1"))
XPROVENCE_MODELLO = os.getenv("XPROVENCE_MODELLO", "naver/xprovence-reranker-bgem3-v1")
RIASSUNTO_CONCORRENZA = int(os.getenv("RIASSUNTO_CONCORRENZA", "10"))
RIASSUNTO_MAX_CHARS = int(os.getenv("RIASSUNTO_MAX_CHARS", "12000"))

# Estrattivo, non discorsivo: la famiglia di domande piu' debole e' quella degli elenchi, dove il
# sistema gia' sottostima perche' elenca cio' che vede. Un riassunto che parafrasa accorcerebbe
# gli elenchi e peggiorerebbe proprio li'.
PROMPT_RIASSUNTO = (
    "DOMANDA: {domanda}\n\n"
    "TESTO:\n{testo}\n\n"
    "Riporta dal TESTO, testualmente e senza riformulare, tutte e sole le informazioni che "
    "servono a rispondere alla DOMANDA: nomi, numeri, date, indirizzi ed elenchi vanno copiati "
    "per intero e senza abbreviarli. Non aggiungere commenti n\u00e9 spiegazioni. Se il TESTO non "
    "contiene nulla di utile, rispondi esattamente: NULLA."
)
RIPARAZIONE_K = int(os.getenv("RIPARAZIONE_K", "100"))
RIPARAZIONE_ANCORAGGIO = float(os.getenv("RIPARAZIONE_ANCORAGGIO", "0.5"))

# Le sette formule con cui il prompt di generazione istruisce il modello a dichiarare di non
# sapere (regole 7 e 9). Sono le stesse di refusal_detector.py, scelte a suo tempo leggendo le
# risposte reali e non ipotizzate.
_RIFIUTO_RE = re.compile("|".join([
    r"non dispongo", r"non sono disponibili informazioni", r"non ho informazioni",
    r"non risulta[a-z]*\s+(un|il|la|alcun|nessun)",
    r"non e[\u0027\u2019] possibile (elencare|fornire|determinare|indicare|specificare)",
    r"non sono in grado di", r"non sono presenti informazioni",
]), re.IGNORECASE)


def _e_rifiuto(testo: str) -> bool:
    return not (testo or "").strip() or bool(_RIFIUTO_RE.search(testo))


def _quota_ancorata(risposta: str, contesto: str) -> float:
    """Quanta parte dei termini distintivi della risposta compare nel contesto.

    Si guardano solo nomi propri, sigle e numeri: sono cio' che una risposta afferma e che puo'
    essere inventato. Le parole comuni non dicono niente sull'ancoraggio. Se non ce ne sono, si
    restituisce 1.0: una risposta che non afferma nessun dato puntuale non puo' essere accusata
    di averlo inventato."""
    termini = set(re.findall(r"\b(?:[A-Z][A-Za-z\u00c0-\u00fa\u0027]{2,}|[A-Z]{2,}|\d{2,})\b", risposta or ""))
    termini = {t for t in termini if t.lower() not in (
        "universita", "ateneo", "salerno", "studi", "degli", "della", "dell", "non", "email")}
    if not termini:
        return 1.0
    c = (contesto or "").lower()
    return sum(1 for t in termini if t.lower() in c) / len(termini)
REGOLA_SINONIMI = (
    "\n10. SINONIMI DI CARICA: le fonti dell'Ateneo indicano la stessa carica con parole diverse. "
    "Considera equivalenti «Capo Area» e «Dirigente dell'Area»; «Capo Ufficio» e «Responsabile "
    "dell'Ufficio»; «Vicario» e «Direttore vicario». Se il CONTESTO indica chi ricopre una carica "
    "usando una di queste parole e la DOMANDA UTENTE ne usa un'altra, rispondi con quella persona "
    "invece di dichiarare che il dato non e' disponibile. ATTENZIONE: «Responsabile del "
    "procedimento» NON e' il Capo o il Dirigente di un ufficio, e' chi segue una singola pratica: "
    "non usarlo mai per rispondere a una domanda su chi dirige un ufficio o un'area."
) if PROMPT_SINONIMI_CARICA else ""
CONTEXTUAL_COMPRESSION_THRESHOLD = float(os.getenv("CONTEXTUAL_COMPRESSION_THRESHOLD", "0.76"))

# Pre-filtro rule-based prima del guardrail LLM: intercetta saluti/ringraziamenti puri senza
# chiamare il modello. Deliberatamente conservativo (matcha solo l'intero messaggio ripulito,
# non una parola al suo interno) per non classificare erroneamente una domanda vera che inizia
# per caso con "ciao". Il caso FUORI_DOMINIO non è coperto qui: distinguerlo con regole affidabili
# richiederebbe una lista aperta di argomenti non pertinenti, troppo rischiosa di falsi positivi.
_GREETING_PREFILTER_PATTERN = re.compile(
    r"^\s*("
    r"ciao|salve|buongiorno|buonasera|buonanotte|"
    r"grazie( mille| infinite)?|ok grazie|va bene[,]? grazie|"
    r"chi sei\??|come stai\??|"
    r"a presto|arrivederci"
    r")[\s!.,?]*$",
    re.IGNORECASE,
)


def _rule_based_domain_prefilter(question: str) -> str | None:
    """Restituisce 'CONVENEVOLI' se l'intero messaggio è un saluto/ringraziamento riconosciuto,
    altrimenti None (nel qual caso si ricade sul guardrail LLM)."""
    if _GREETING_PREFILTER_PATTERN.match(question.strip()):
        return "CONVENEVOLI"
    return None

def data_odierna() -> str:
    """Data da mettere nel prompt: quella reale, oppure quella congelata per i giri di prova.
    Un valore malformato viene ignorato con un avviso, invece di finire nel prompt."""
    if BENCHMARK_DATE:
        try:
            datetime.strptime(BENCHMARK_DATE, "%d/%m/%Y")
            return BENCHMARK_DATE
        except ValueError:
            logger.warning(f"[DATA] BENCHMARK_DATE=«{BENCHMARK_DATE}» non e' nel formato "
                           f"gg/mm/aaaa: uso la data reale")
    return datetime.now().strftime("%d/%m/%Y")


class _CachedQueryEmbeddings:
    """Avvolge l'embedder (HuggingFaceEmbeddings) per non ricalcolare l'embedding della stessa
    query piu' volte di seguito.

    Perche' serve: ogni collezione ha il proprio VectorStoreRetriever, e ciascuno richiama
    internamente `embedding_function.embed_query(q)` quando interrogato - con 4 collezioni,
    _retrieve_and_format embeddava la STESSA domanda 4 volte, sulla stessa GPU, per lo stesso
    identico risultato. Misurato (02/10/2026): tentare di parallelizzare le 4 chiamate a
    `retriever.invoke()` con un ThreadPoolExecutor non ha aiutato - anzi, il retrieval e'
    risultato PIU' lento (rapporto a 4 utenti 2,16x contro 2,07x della baseline sequenziale) -
    perche' il costo non era I/O parallelizzabile: la ricerca lessicale BM25 e' Python puro
    (vincolato dal GIL, i thread non la accelerano) e le 4 chiamate GPU ridondanti, lanciate
    "insieme" da piu' thread, si sono probabilmente intralciate a vicenda sulla GPU invece di
    sovrapporsi in modo utile. La vera redundanza da eliminare non e' la sequenzialita' delle
    chiamate, ma il fatto che siano 4 chiamate identiche quando basterebbe farne 1.

    Cache LRU di piccola taglia (non un singolo slot): piu' utenti concorrenti possono avere
    domande diverse in corso nello stesso momento (il test di concorrenza usa thread paralleli
    per utente), quindi la cache deve poter tenere piu' query distinte senza che si scaccino a
    vicenda prima di essere riusate dalle 4 collezioni della stessa domanda. Protetta da lock
    perche' e' condivisa fra i thread degli utenti simulati.
    """

    def __init__(self, embedder, maxsize: int = 32):
        self._embedder = embedder
        self._maxsize = maxsize
        self._cache = OrderedDict()
        self._lock = threading.Lock()

    def embed_query(self, text: str):
        with self._lock:
            vettore = self._cache.get(text)
            if vettore is not None:
                self._cache.move_to_end(text)
                return vettore
        vettore = self._embedder.embed_query(text)
        with self._lock:
            self._cache[text] = vettore
            self._cache.move_to_end(text)
            while len(self._cache) > self._maxsize:
                self._cache.popitem(last=False)
        return vettore

    def embed_documents(self, texts):
        # Usato solo in fase di costruzione degli indici (un batch enorme e irripetuto, mai la
        # stessa query due volte): nessun bisogno di cache, si delega e basta.
        return self._embedder.embed_documents(texts)

    def __getattr__(self, name):
        # Delega qualunque altro attributo o metodo (es. usato da altre librerie che
        # ispezionano l'embedder) all'oggetto originale, cosi' questo wrapper resta
        # trasparente ovunque non ci sia una query da mettere in cache.
        return getattr(self._embedder, name)


class DiemRAG:
    """
    DIEM RAG engine implementing an LCEL-based architecture.
    Integrates semantic routing, few-shot query expansion, targeted retrieval, 
    and cross-encoder re-ranking.
    """
    
    def __init__(self):
        """
        Initializes the RAG infrastructure.
            
        Raises:
            Exception: If API connections, model loading, or local vectorstore instantiations fail.
        """
        try:
            logger.info("DIEM RAG initialization...")
            
            # Embedding model intialization
            self.embedding_model = HuggingFaceEmbeddings(
                model_name="BAAI/bge-m3",
                model_kwargs={"device": "cuda"},
                encode_kwargs={"normalize_embeddings": True},
            )
            # Cache dell'embedding di query (vedi _CachedQueryEmbeddings sopra): evita di
            # ricalcolare 4 volte lo stesso embedding quando _retrieve_and_format interroga le
            # 4 collezioni con la stessa domanda. Va fatto qui, prima che _retrieve_vectorstore
            # costruisca i Chroma con questo embedder: tutte le collezioni condividono la stessa
            # istanza (e quindi la stessa cache).
            self.embedding_model = _CachedQueryEmbeddings(self.embedding_model)
            
            # Binding to the persistent ChromaDB instance and dynamically loading all collections for retrieval
            self.client = chromadb.PersistentClient(path=str(PERSIST_DIRECTORY))
            self._cache_pagine = {}   # indirizzo -> testo completo della pagina (o None)
            self._cache_parenti = {}  # indirizzo -> [(parent_id, testo)] in ordine di documento
            self._posizioni = {}      # indirizzo -> {parent_id: posizione nel documento}
            if ENABLE_VICINATO:
                try:
                    grezzo = json.loads(POSIZIONI_PATH.read_text(encoding="utf-8"))
                    self._posizioni = {u: {p: d["posizione"] for p, d in v.items()}
                                       for u, v in grezzo.items()}
                    logger.info("[VICINATO] mappa delle posizioni: %d documenti, %d genitori"
                                % (len(self._posizioni),
                                   sum(len(v) for v in self._posizioni.values())))
                except Exception as e:
                    logger.warning("[VICINATO] mappa non caricata (%s): il vicinato resta inattivo" % e)

            # Anagrafe delle persone: caricata solo se la pipeline dedicata e' attiva. Se il
            # file manca, la pipeline si disattiva da sola e il sistema continua a funzionare
            # come prima.
            self.anagrafe = None
            if ENABLE_PERSON_PIPELINE:
                try:
                    from .anagrafe_lookup import Anagrafe
                    self.anagrafe = Anagrafe(ANAGRAFE_PATH)
                    logger.info(f"[PERSONE] Anagrafe caricata: {len(self.anagrafe.persone)} persone")
                except Exception as e:
                    logger.warning(f"[PERSONE] Anagrafe non caricata ({e}): pipeline dedicata disattivata")

            # Registro delle strutture: caricato solo se il percorso «struttura» e' acceso. Se il
            # file manca, il percorso si disattiva da solo e il sistema funziona come prima.
            self.registro_strutture = None
            if ENABLE_STRUCT_PATH:
                try:
                    from .struttura_lookup import Registro
                    self.registro_strutture = Registro(REGISTRO_STRUTTURE_PATH, attributi=ENABLE_STRUCT_ATTRIBUTES)
                    logger.info(f"[STRUTTURA] Registro caricato: {len(self.registro_strutture.aree)} aree, "
                                f"{len(self.registro_strutture.uffici)} uffici, "
                                f"{len(self.registro_strutture.distretti)} distretti")
                except Exception as e:
                    logger.warning(f"[STRUTTURA] Registro non caricato ({e}): percorso struttura disattivato")
            self._minuscole = Counter()   # verifica dei nomi: parole in minuscolo dei pezzi indicizzati
            collections = self.client.list_collections()
            self.retrievers = {}
            
            for c in collections:
                self.retrievers[c.name] = self._retrieve_vectorstore(
                    vs_dir=PERSIST_DIRECTORY, 
                    collection_name=c.name
                )

            # Verifica dei nomi: parole COMUNI = in minuscolo almeno 2 volte nei pezzi dell'indice (i nomi e i
            # refusi no). Contate sui testi gia' letti per BM25, senza una lettura in piu' dell'indice.
            self.parole_comuni = parole_comuni(self._minuscole) if ENABLE_NAME_CHECK else set()
            del self._minuscole
            if ENABLE_NAME_CHECK:
                logger.info(f"[NOMI] Verifica dei nomi accesa: {len(self.parole_comuni)} parole comuni")

            logger.info("Reranker initialization...")
            
            # Cross-Encoder weights loading for semantic re-ranking
            self.reranker = crea_reranker() if crea_reranker else CrossEncoder("BAAI/bge-reranker-v2-m3")
            if _descrizione_recupero:
                logger.info(f"[RECUPERO] {_descrizione_recupero()}, VECTOR_K {VECTOR_K}, BM25_K {BM25_K}, "
                            f"RERANK_CANDIDATI {RERANK_CANDIDATI or 'tutti'} "
                            f"(elenchi {RERANK_CANDIDATI_ELENCHI or 'tutti'})")

            logger.info(f"LLM initialization (backend={LLM_BACKEND})...")

            # LLM initialization
            if LLM_BACKEND == "ollama":
                self.llm = ChatOllama(
                    base_url=OLLAMA_BASE_URL, num_ctx=OLLAMA_NUM_CTX or None,
                    model=OLLAMA_MODEL_ID, temperature=0, num_predict=4096,
                    reasoning=OLLAMA_ENABLE_THINKING,
                )
                # Modello dedicato al filtro: risposta di una parola sola e ragionamento
                # disattivato, altrimenti ogni giudizio costerebbe come una risposta intera.
                self.llm_filtro = ChatOllama(
                    base_url=OLLAMA_BASE_URL, num_ctx=OLLAMA_NUM_CTX or None,
                    model=OLLAMA_MODEL_ID, temperature=0, num_predict=4, reasoning=False,
                ) if ENABLE_CHUNK_FILTER else None
                # Il riassunto ha bisogno di scrivere, non di rispondere una parola: num_predict
                # piu' alto, ma tenuto basso perche' un estratto lungo quanto l'originale non
                # ridurrebbe niente. Il filtro serve anche al modo «scartati», quindi in quel modo
                # il modello del filtro viene creato comunque.
                self.llm_riassunto = ChatOllama(
                    base_url=OLLAMA_BASE_URL, num_ctx=OLLAMA_NUM_CTX or None,
                    model=OLLAMA_MODEL_ID, temperature=0, num_predict=512, reasoning=False,
                ) if RIASSUNTO_MODO in ("tutti", "scartati") else None
                # Modalita' elenco: scrive 3 righe di ricerca, niente ragionamento (come il filtro).
                self.llm_elenco = ChatOllama(
                    base_url=OLLAMA_BASE_URL, num_ctx=OLLAMA_NUM_CTX or None,
                    model=OLLAMA_MODEL_ID, temperature=0, num_predict=300, reasoning=False,
                ) if ENABLE_LIST_MODE else None
                # Ragionamento mirato: stesso modello (Ollama non lo ricarica), ragionamento acceso e tetto di
                # token alto: ragionamento e risposta stanno nello stesso budget.
                self.llm_ragionamento = ChatOllama(
                    base_url=OLLAMA_BASE_URL, num_ctx=OLLAMA_NUM_CTX or None,
                    model=OLLAMA_MODEL_ID, temperature=0, num_predict=RAGIONAMENTO_NUM_PREDICT, reasoning=True,
                ) if ENABLE_RAGIONAMENTO_MIRATO else None

            elif LLM_BACKEND == "vllm":
                # import qui dentro: chi usa Ollama non ha bisogno del pacchetto langchain-openai
                from langchain_openai import ChatOpenAI

                def _chat_vllm(max_token: int, thinking: bool) -> "ChatOpenAI":
                    return ChatOpenAI(
                        base_url=VLLM_BASE_URL, api_key="EMPTY", model=VLLM_MODEL_ID,
                        temperature=0, max_tokens=max_token, timeout=900, max_retries=0,
                        # nell'ultimo pezzo dello streaming: token del prompt e della risposta
                        stream_usage=True,
                        extra_body={"chat_template_kwargs": {"enable_thinking": thinking}},
                    )

                self.llm = _chat_vllm(4096, OLLAMA_ENABLE_THINKING)
                self.llm_filtro = _chat_vllm(4, False) if ENABLE_CHUNK_FILTER else None
                self.llm_riassunto = _chat_vllm(512, False) if RIASSUNTO_MODO in ("tutti", "scartati") else None
                # Modalita' elenco: come con Ollama, 3 righe di ricerca, niente ragionamento.
                self.llm_elenco = _chat_vllm(300, False) if ENABLE_LIST_MODE else None
                # Ragionamento mirato: NON supportato con vLLM (il server non e' avviato con un reasoning parser, quindi
                # il ragionamento finirebbe dentro la risposta). Provato e non adottato: con vLLM resta spento.
                self.llm_ragionamento = None
                if ENABLE_RAGIONAMENTO_MIRATO:
                    logger.warning("[VLLM] ENABLE_RAGIONAMENTO_MIRATO ignorato: non supportato con il backend vLLM")
                logger.info(f"[VLLM] server {VLLM_BASE_URL}, modello {VLLM_MODEL_ID}, "
                            f"thinking={OLLAMA_ENABLE_THINKING}")

            else:
                hf_pipeline = HuggingFacePipeline.from_model_id(
                    model_id=HF_MODEL_ID,
                    task="text-generation",
                    device_map="auto",
                    pipeline_kwargs={
                        "max_new_tokens": 4096,
                        "do_sample": False,
                        "temperature": None,
                        "top_p": None,
                        "return_full_text": False,
                    },
                    model_kwargs={
                        "torch_dtype": "auto",
                        "token": HF_TOKEN,
                    },
                )
                self.llm = ChatHuggingFace(llm=hf_pipeline, tokenizer=hf_pipeline.pipeline.tokenizer)

            self.xprovence = None
            if ENABLE_XPROVENCE:
                # torch non e' importato a livello di modulo in questo file
                import torch as _torch
                from transformers import AutoModel as _AutoModel
                _dev = "cuda" if _torch.cuda.is_available() else "cpu"
                self.xprovence = _AutoModel.from_pretrained(
                    XPROVENCE_MODELLO, trust_remote_code=True).to(_dev).eval()
                logger.info("[XPROVENCE] modello caricato su %s, soglia %.2f"
                            % (_dev, XPROVENCE_SOGLIA))
                if RIASSUNTO_MODO == "scartati" and self.llm_filtro is None:
                    self.llm_filtro = ChatOllama(
                        base_url=OLLAMA_BASE_URL, num_ctx=OLLAMA_NUM_CTX or None,
                        model=OLLAMA_MODEL_ID, temperature=0, num_predict=4, reasoning=False,
                    )
            # FUORI da `if ENABLE_XPROVENCE:` (corretto il 05/10/2026 in diem_RAG_routered.py, riportato qui il
            # 07/10): con il blocco indentato dentro l'if, con ENABLE_XPROVENCE=false (default) num_ctx/num_batch
            # non venivano mai applicati. Il modello della modalita' elenco riceve lo stesso num_ctx: con un
            # contesto diverso fra due richieste Ollama ricaricherebbe il modello a ogni domanda di elenco.
            if LLM_BACKEND == "ollama" and (OLLAMA_NUM_BATCH or OLLAMA_NUM_CTX):
                # il client costruisce da se' il dizionario delle opzioni, ma se gliene si
                # passa uno lo usa al posto del suo: va quindi ricostruito per intero,
                # altrimenti si perdono temperatura e lunghezza massima della risposta
                opzioni = {"temperature": 0, "num_predict": 4096}
                if OLLAMA_NUM_CTX:
                    opzioni["num_ctx"] = OLLAMA_NUM_CTX
                if OLLAMA_NUM_BATCH:
                    opzioni["num_batch"] = OLLAMA_NUM_BATCH
                self.llm = self.llm.bind(options=opzioni)
                if self.llm_ragionamento is not None:
                    self.llm_ragionamento = self.llm_ragionamento.bind(
                        options=dict(opzioni, num_predict=RAGIONAMENTO_NUM_PREDICT))
                if self.llm_elenco is not None:
                    self.llm_elenco = self.llm_elenco.bind(options=dict(opzioni, num_predict=300))
                logger.info(f"[OLLAMA] Opzioni esplicite: {opzioni}")
            
            # In-memory dictionary for conversational state management
            self.chat_histories = {}
            
            self.conversational_pipeline = self._build_pipeline()
            logger.info("[SUCCESS] RAG ready to go...")
        except Exception as e:
            logger.error(f"Rag initialization failed: {str(e)}")
            raise

    def _build_router_collection_descriptions(self) -> str:
        """
        Filtra ROUTER_COLLECTION_DESCRIPTIONS alle sole collezioni realmente caricate in
        self.retrievers, scartando le righe (bullet "- collection_x : ...") relative a
        collezioni non presenti in questo deployment.

        Verificato empiricamente (test manuali del 2026-07-21, spike ateneo_minimo): con tutte
        le descrizioni statiche ma una sola collezione reale caricata, il router selezionava
        'false' per l'unica collezione esistente in ~20% dei casi anche su domande chiaramente
        pertinenti - il mismatch tra "Collezioni e descrizioni" (14 voci) e "Nomi delle
        collezioni" (1 voce reale) confondeva il modello. Le righe non-bullet (es. le note
        "IMPORTANTE") restano invariate: sono avvertenze generiche, non legate a una singola
        collezione specifica.
        """
        active_names = set(self.retrievers.keys())
        kept_lines = []
        for line in ROUTER_COLLECTION_DESCRIPTIONS.strip("\n").split("\n"):
            stripped = line.strip()
            if stripped.startswith("- collection_"):
                collection_name = stripped[2:].split(" :", 1)[0].strip()
                if collection_name not in active_names:
                    continue
            kept_lines.append(line)
        return "\n".join(kept_lines)

    def _retrieve_vectorstore(self, vs_dir: Path, collection_name: str):
        """
        Defines and configures a hybrid LangChain retriever (embedding + BM25) for a specific
        ChromaDB collection.

        Motivazione: la similarità semantica pura (bge-m3) fallisce sistematicamente su query
        che cercano un dato puntuale (email/telefono di un'area, ufficio o persona specifica)
        dentro un chunk breve e poco discorsivo (es. "AREA VI - Risorse Strumentali | ar000006@
        unisa.it") - verificato che il chunk corretto puo' restare fuori anche dai primi 1000
        risultati su ~129k child chunk. BM25 (ricerca lessicale, term-frequency) recupera bene
        proprio questi casi perche' i termini della query (es. "AREA VI") compaiono quasi
        verbatim nel chunk. L'EnsembleRetriever combina i due ranking (reciprocal rank fusion),
        cosi' i casi gia' ben serviti dall'embedding puro non regrediscono.

        Args:
            vs_dir (Path): The absolute path to the persistent vectorstore directory.
            collection_name (str): The unique identifier of the target collection.

        Returns:
            EnsembleRetriever: A retriever combining embedding-based and BM25 lexical search.

        Raises:
            Exception: If an I/O error occurs or the specified collection cannot be loaded.
        """
        try:
            vectorstore = Chroma(
                collection_name=collection_name,
                embedding_function=self.embedding_model,
                persist_directory=str(vs_dir),
            )
            vector_retriever = vectorstore.as_retriever(search_kwargs={"k": VECTOR_K})

            # Carica tutti i child chunk della collezione per costruire l'indice BM25.
            # Paginato: un singolo .get() senza limit su collezioni grandi (>~100k righe)
            # supera il limite di variabili SQL di sqlite lato chromadb.
            raw_collection = vectorstore._collection
            total = raw_collection.count()
            batch_size = 5000
            all_documents, all_metadatas = [], []
            for offset in range(0, total, batch_size):
                batch = raw_collection.get(
                    include=["documents", "metadatas"], limit=batch_size, offset=offset
                )
                all_documents.extend(batch["documents"])
                all_metadatas.extend(batch["metadatas"])

            bm25_docs = [
                Document(page_content=doc, metadata=meta)
                for doc, meta in zip(all_documents, all_metadatas)
            ]
            if ENABLE_NAME_CHECK and getattr(self, "_minuscole", None) is not None:
                self._minuscole.update(conta_minuscole(all_documents))
            if BM25_MOTORE == "sparso" and crea_bm25:
                # stessa formula e stessa tokenizzazione, calcolo su matrice sparsa (ottimizzazioni_recupero.py)
                bm25_retriever = crea_bm25(bm25_docs, _bm25_tokenizer if BM25_NORMALIZE_TOKENS else None, BM25_K)
            elif BM25_NORMALIZE_TOKENS:
                bm25_retriever = BM25Retriever.from_documents(
                    bm25_docs, preprocess_func=_bm25_tokenizer
                )
                bm25_retriever.k = BM25_K
            else:
                bm25_retriever = BM25Retriever.from_documents(bm25_docs)
                bm25_retriever.k = BM25_K

            return EnsembleRetriever(
                retrievers=[vector_retriever, bm25_retriever], weights=[0.5, 0.5]
            )
        except Exception as e:
            logger.error(f" Error loading vectorstore for collection '{collection_name}': {str(e)}")
            raise

    def _cerca_persona(self, domanda: str):
        """Se la domanda riguarda una persona risolvibile, restituisce la sua scheda, le fonti e
        il tipo di percorso da seguire; altrimenti None e la pipeline prosegue normalmente.

        La chiave 'veloce' dice se il recupero puo' essere saltato del tutto: vale True solo
        quando la domanda chiede un campo che quella scheda contiene davvero. Negli altri casi
        la scheda viene comunque restituita, ma per essere aggiunta al contesto recuperato."""
        if not self.anagrafe:
            return None
        t0 = time.perf_counter()
        try:
            from .anagrafe_lookup import classifica, percorso_veloce, scheda
            tipo, nomi = classifica(domanda)
            if tipo != "persona":
                return None
            # la domanda serve a distinguere gli omonimi: l'anagrafe ha una scheda per pagina
            # della rubrica e sceglie quella che la domanda descrive (ufficio, qualifica, incarichi)
            rec, motivo, nome = self.anagrafe.cerca_tutti(nomi, domanda)
            if rec is None:
                logger.info(f"[PERSONE] «{nomi[0] if nomi else '-'}»: {motivo} -> pipeline normale")
                return None
            testo = scheda(rec)
            docs = [
                Document(page_content=testo,
                         metadata={"source_url": u, "relevance_score": 1.0,
                                   "title": "Anagrafe di ateneo"})
                for u in (rec.get("fonti") or ["anagrafe"])[:3]
            ]
            veloce = PERSON_FAST_PATH and percorso_veloce(domanda, rec)
            modo = ("recupero saltato (percorso veloce)" if veloce
                    else "scheda aggiunta al contesto recuperato")
            logger.info(f"[PERSONE] «{nome}» risolto ({motivo}) in {time.perf_counter() - t0:.3f}s: "
                        f"{modo}")
            return {"context": testo, "docs": docs, "veloce": veloce}
        except Exception as e:
            logger.warning(f"[PERSONE] Errore nella ricerca ({e}): pipeline normale")
            return None

    def _cerca_struttura(self, domanda: str):
        """Se la domanda chiede di elencare o contare i componenti di una struttura del registro,
        restituisce la scheda, le fonti e la modalita'; altrimenti None e la pipeline prosegue
        normalmente. Gemello di _cerca_persona, nessuna chiamata a un modello.

        La chiave 'veloce' dice se il recupero si puo' saltare: vale True quando la domanda chiede
        solo cio' che la scheda contiene (elenchi, conteggi, capi, interni, email). Se chiede anche
        altro («...e di cosa si occupano?») la scheda va in testa al contesto recuperato."""
        if not self.registro_strutture:
            return None
        t0 = time.perf_counter()
        try:
            from .struttura_lookup import OPS_ATTRIBUTO, fonti_della_scheda, percorso_veloce
            esito, motivo = self.registro_strutture.riconosci(domanda)
            if not esito:
                return None
            testo = self.registro_strutture.scheda(esito)
            docs = [
                Document(page_content=testo,
                         metadata={"source_url": u, "relevance_score": 1.0,
                                   "title": "Registro delle strutture di ateneo"})
                for u in fonti_della_scheda(testo)
            ]
            veloce = percorso_veloce(domanda) and esito["op"] not in OPS_ATTRIBUTO
            modo = ("recupero saltato (percorso veloce)" if veloce
                    else "scheda aggiunta al contesto recuperato")
            logger.info(f"[STRUTTURA] {esito['op']} ({esito['chiave']}, {motivo}) riconosciuta in "
                        f"{time.perf_counter() - t0:.3f}s: {modo}")
            return {"context": testo, "docs": docs, "veloce": veloce}
        except Exception as e:
            logger.warning(f"[STRUTTURA] Errore nel riconoscimento ({e}): pipeline normale")
            return None

    def _ordina_contesto(self, docs: list) -> list:
        """Ordine con cui i pezzi entrano nel contesto. Non toglie e non aggiunge nulla: sposta
        soltanto le posizioni, perche' un modello legge con piu' attenzione l'inizio e la fine di
        cio' che riceve, e la domanda sta dopo il contesto."""
        if CONTEXT_ORDER == "coda":
            return list(reversed(docs))
        if CONTEXT_ORDER == "estremi":
            return docs[0::2] + docs[1::2][::-1]
        return docs

    def _riassumi_pezzi(self, q: str, docs: list) -> list:
        """Sostituisce i pezzi con il loro estratto, nei due modi previsti.

        Restituisce oggetti Document nuovi, con lo stesso source_url, cosi' che il resto della
        pipeline — la deduplica per genitore, l'attribuzione delle fonti, l'ordinamento — continui
        a funzionare senza saperne niente.
        """
        if RIASSUNTO_MODO not in ("tutti", "scartati") or not getattr(self, "llm_riassunto", None):
            return docs
        if len(docs) < 2:
            return docs
        t0 = time.perf_counter()

        def testo_di(d):
            return (d.metadata.get("parent_content") or d.page_content or "")

        if RIASSUNTO_MODO == "scartati":
            # il filtro decide chi passa intero e chi viene condensato
            prompt = [PROMPT_FILTRO_PEZZI.format(domanda=q, testo=testo_di(d)[:CHUNK_FILTER_MAX_CHARS])
                      for d in docs]
            giudizi = self.llm_filtro.batch(prompt, config={"max_concurrency": RIASSUNTO_CONCORRENZA})
            interi, da_ridurre = [], []
            for d, r in zip(docs, giudizi):
                t = getattr(r, "content", str(r))
                (interi if (_PERTINENTE.search(t) and not _NON_PERTINENTE.search(t))
                 else da_ridurre).append(d)
            unita = [([d], testo_di(d)) for d in da_ridurre]
            tenuti = list(interi)
        else:
            # «tutti»: i pezzi della stessa pagina sono elementi adiacenti dello stesso documento
            # e vanno trattati come uno solo, poi si estrae da ogni unita'
            gruppi = {}
            for d in docs:
                gruppi.setdefault(d.metadata.get("source_url") or id(d), []).append(d)
            unita = [(g, "\n\n".join(testo_di(x) for x in g)) for g in gruppi.values()]
            tenuti = []

        if not unita:
            return tenuti or docs
        prompt = [PROMPT_RIASSUNTO.format(domanda=q, testo=t[:RIASSUNTO_MAX_CHARS])
                  for _, t in unita]
        estratti = self.llm_riassunto.batch(prompt, config={"max_concurrency": RIASSUNTO_CONCORRENZA})

        fuori = list(tenuti)
        vuoti = 0
        for (gruppo, _), r in zip(unita, estratti):
            testo = (getattr(r, "content", str(r)) or "").strip()
            if not testo or testo.upper().startswith("NULLA"):
                vuoti += 1
                continue
            capo = gruppo[0]
            meta = dict(capo.metadata)
            meta["parent_content"] = testo
            meta["riassunto"] = True
            # la deduplica a valle usa parent_id: un estratto che riunisce piu' pezzi deve avere
            # un identificativo proprio, altrimenti verrebbe scambiato per uno dei suoi originali
            meta["parent_id"] = "estratto:" + str(meta.get("source_url", "")) + ":" + str(len(fuori))
            fuori.append(Document(page_content=testo, metadata=meta))

        logger.info("[RIASSUNTO] modo=%s  %d pezzi -> %d unita' (%d interi, %d estratti, %d vuoti) "
                    "in %.2fs" % (RIASSUNTO_MODO, len(docs), len(fuori), len(tenuti),
                                  len(fuori) - len(tenuti), vuoti, time.perf_counter() - t0))
        return fuori or docs

    def _pota_pezzi(self, q: str, docs: list) -> list:
        """Toglie da ogni pezzo le frasi che XProvence giudica non pertinenti alla domanda.

        La corrispondenza e' uno a uno: nessun pezzo viene eliminato, ognuno viene accorciato.
        Un pezzo che resterebbe vuoto viene rimesso intero, per lo stesso motivo per cui il
        filtro tiene sempre almeno un pezzo: il contesto vuoto deve restare il segnale di un
        recupero fallito, non il risultato di un taglio troppo aggressivo.
        """
        if not ENABLE_XPROVENCE or not getattr(self, "xprovence", None) or not docs:
            return docs
        t0 = time.perf_counter()
        testi = [(d.metadata.get("parent_content") or d.page_content or "") for d in docs]
        try:
            out = self.xprovence.process(
                q, [testi], threshold=XPROVENCE_SOGLIA, always_select_title=True,
                reorder=False, enable_warnings=False)
            potati = out["pruned_context"][0]
        except Exception as e:
            logger.warning("[XPROVENCE] errore nella potatura (%s): contesto invariato" % e)
            return docs
        fuori, vuoti, car0, car1 = [], 0, 0, 0
        for d, originale, potato in zip(docs, testi, potati):
            testo = (potato if isinstance(potato, str) else str(potato)).strip()
            car0 += len(originale)
            if not testo:
                vuoti += 1
                fuori.append(d)
                car1 += len(originale)
                continue
            car1 += len(testo)
            meta = dict(d.metadata)
            meta["parent_content"] = testo
            meta["potato"] = True
            fuori.append(Document(page_content=testo, metadata=meta))
        logger.info("[XPROVENCE] soglia=%.2f  %d pezzi, %d caratteri -> %d (-%.0f%%), "
                    "%d rimessi interi, in %.2fs"
                    % (XPROVENCE_SOGLIA, len(docs), car0, car1,
                       (1 - car1 / max(1, car0)) * 100, vuoti, time.perf_counter() - t0))
        return fuori

    def _filtra_pezzi(self, q: str, docs: list) -> list:
        """Chiede al modello, per ogni pezzo recuperato, se contenga informazioni utili alla
        domanda, e tiene solo quelli per cui la risposta e' si'. Le domande partono insieme.
        Se nessun pezzo passa si tiene il primo: il contesto vuoto deve restare il segnale di un
        recupero fallito davvero, non l'effetto del filtro."""
        if not (ENABLE_CHUNK_FILTER and getattr(self, "llm_filtro", None)) or len(docs) < 2:
            return docs
        t0 = time.perf_counter()
        try:
            prompt = [
                # si giudica il chunk figlio, non il genitore: e' l'unita' che il re-ranker ha
                # valutato, porta con se' il percorso degli header ed e' abbastanza corta da non
                # dover essere troncata, mentre il genitore arriva a 5.000 caratteri e verrebbe
                # tagliato proprio dove potrebbe stare il dato
                PROMPT_FILTRO_PEZZI.format(
                    domanda=q, testo=d.page_content[:CHUNK_FILTER_MAX_CHARS])
                for d in docs
            ]
            risposte = self.llm_filtro.batch(
                prompt, config={"max_concurrency": CHUNK_FILTER_CONCURRENCY})
            testi = [getattr(r, "content", "") or "" for r in risposte]
            if CHUNK_FILTER_MODE == "prudente":
                tenuti = [d for d, t in zip(docs, testi)
                          if not (_NON_PERTINENTE.search(t) and not _PERTINENTE.search(t))]
            else:
                tenuti = [d for d, t in zip(docs, testi) if _PERTINENTE.search(t)]
            logger.info(f"[FILTRO] ({CHUNK_FILTER_MODE}) {len(docs)} -> {max(1, len(tenuti))} pezzi "
                        f"tenuti in {time.perf_counter() - t0:.2f}s")
            return tenuti or docs[:1]
        except Exception as e:
            logger.warning(f"[FILTRO] Errore ({e}): i pezzi passano tutti")
            return docs

    def _parenti_ordinati(self, source_url: str):
        """I genitori di una pagina, in ordine di documento secondo la mappa delle posizioni.

        I genitori che la mappa non posiziona finiscono in coda nell'ordine di memorizzazione:
        restano disponibili, ma non possono fare da riferimento per il calcolo dei vicini.
        Il risultato e' in cache perche' le pagine lunghe ricorrono in molte domande."""
        if source_url in self._cache_parenti:
            return self._cache_parenti[source_url]
        fuori = []
        try:
            for nome in self.retrievers.keys():
                res = self.client.get_collection(nome).get(
                    where={"source_url": source_url}, include=["documents", "metadatas"]
                )
                if not res["ids"]:
                    continue
                visti = set()
                for doc, meta in zip(res["documents"], res["metadatas"]):
                    pid = meta.get("parent_id")
                    if pid is not None:
                        if pid in visti:
                            continue
                        visti.add(pid)
                    fuori.append((pid, meta.get("parent_content") or doc))
                break
        except Exception as e:
            logger.warning(f"[PAGE] Lettura della pagina {source_url} fallita: {e}")
        pos = self._posizioni.get(source_url, {})
        if pos:
            fuori.sort(key=lambda x: pos.get(x[0], 10 ** 6))
        self._cache_parenti[source_url] = fuori
        return fuori

    def _vicinato(self, source_url: str, pid_scelti: set, raggio: int):
        """I genitori adiacenti a quelli scelti dal re-ranker, piu' i genitori scelti stessi.

        Ritorna None se nessuno dei genitori scelti ha una posizione nota, oppure se il testo
        risultante supera comunque il tetto per pagina: in entrambi i casi il chiamante prosegue
        con il comportamento di oggi. I genitori scelti sono inclusi anche quando non sono
        posizionati, altrimenti l'espansione perderebbe proprio i pezzi per cui e' scattata."""
        parenti = self._parenti_ordinati(source_url)
        if not parenti:
            return None
        pos = self._posizioni.get(source_url, {})
        indice = {pid: i for i, (pid, _) in enumerate(parenti)}
        voluti = {indice[p] for p in pid_scelti if p in indice}
        ancorati = [p for p in pid_scelti if p in pos]
        if not ancorati:
            return None
        for p in ancorati:
            i = indice.get(p)
            if i is None:
                continue
            for j in range(max(0, i - raggio), min(len(parenti), i + raggio + 1)):
                voluti.add(j)
        testo = "\n\n".join(parenti[j][1] for j in sorted(voluti))
        if len(testo) > PAGE_EXPANSION_MAX_CHARS:
            return None
        return testo

    def _pagina_intera(self, source_url: str):
        """Contenuto completo di una pagina: l'unione dei suoi chunk genitore, deduplicati e
        nell'ordine in cui sono memorizzati. Ritorna None se la pagina non si trova o se supera
        il tetto di lunghezza. Il risultato e' messo in cache: le stesse pagine lunghe (rubrica,
        organi collegiali) ricorrono in molte domande."""
        if source_url in self._cache_pagine:
            return self._cache_pagine[source_url]
        testo = None
        try:
            for nome in self.retrievers.keys():
                res = self.client.get_collection(nome).get(
                    where={"source_url": source_url}, include=["documents", "metadatas"]
                )
                if not res["ids"]:
                    continue
                visti, parti = set(), []
                for doc, meta in zip(res["documents"], res["metadatas"]):
                    pid = meta.get("parent_id")
                    if pid is not None:
                        if pid in visti:
                            continue
                        visti.add(pid)
                    parti.append(meta.get("parent_content") or doc)
                candidato = "\n\n".join(parti)
                if len(candidato) <= PAGE_EXPANSION_MAX_CHARS:
                    testo = candidato
                break
        except Exception as e:
            logger.warning(f"[PAGE] Lettura della pagina {source_url} fallita: {e}")
        self._cache_pagine[source_url] = testo
        return testo

    def _rerank_func(self, query: str, documents: list, top_n: int = None) -> list:
        """
        Applies a cross-encoder model to compute cross-attention scores between the query 
        and the retrieved documents, effectively filtering semantic noise.
        
        Args:
            query (str): The optimized textual query generated by the rewriting segment.
            documents (list): A list of retrieved document chunks.
            top_n (int): The cutoff threshold for the highest-scoring documents to retain.
            
        Returns:
            list: A subset of the most relevant documents enriched with the calculated `relevance_score`.
            
        Raises:
            Exception: If tensor inference or VRAM saturation errors occur.
        """
        try:
            if top_n is None:
                top_n = RERANK_TOP_N
            if not documents: 
                return []
            
            # Construct [Query, Document] tensor pairs for CrossEncoder evaluation
            pairs = [[query, d.page_content] for d in documents]
            scores = self.reranker.predict(pairs)
            
            # Perform descending sort based on the semantic logit scores
            ranked = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)
            out = []
            for d, s in ranked[:top_n]:
                d.metadata["relevance_score"] = float(s)
                out.append(d)
            if RERANK_MIN_SCORE > 0 and out:
                tenuti = [d for d in out if d.metadata["relevance_score"] >= RERANK_MIN_SCORE]
                if len(tenuti) < len(out):
                    logger.info(f"[SOGLIA] {len(out)} -> {max(RERANK_MIN_KEEP, len(tenuti))} documenti "
                                f"(punteggio minimo {RERANK_MIN_SCORE})")
                out = tenuti if len(tenuti) >= RERANK_MIN_KEEP else out[:RERANK_MIN_KEEP]
            return out
        except Exception as e:
            logger.error(f"[ERROR] Error during reranking: {str(e)}")
            raise

    def _contextual_compression_filter(self, query: str, documents: list, threshold: float) -> list:
        """
        Filtra i documenti già rerankati per similarità coseno tra la query e il chunk (contextual
        compression post-rerank). Riusa lo stesso embedder bge-m3 della retrieval; gli embedding
        sono già L2-normalizzati (encode_kwargs sopra), quindi la similarità coseno si riduce a un
        prodotto scalare. Deliberatamente NON un compressore basato su LLM (LLMChainExtractor):
        una chiamata LLM per chunk aggiungerebbe 5-7 chiamate extra a turno, sforando il budget
        di latenza. Se il filtro scarterebbe tutti i documenti, ricade sulla lista originale
        invece di restituire un contesto vuoto.
        """
        if not documents:
            return documents

        query_vec = np.array(self.embedding_model.embed_query(query))
        doc_vecs = np.array(self.embedding_model.embed_documents([d.page_content for d in documents]))
        similarities = doc_vecs @ query_vec

        kept = [d for d, s in zip(documents, similarities) if s >= threshold]
        return kept if kept else documents

    def _recupero_ampio(self, q: str) -> dict:
        """Ripete il recupero scendendo piu' in profondita', per la riparazione.

        Nel meccanismo di luglio il secondo tentativo faceva due cose: allargava k da 20 a 100 e
        interrogava tutte le collezioni invece di quella scelta dal router. Oggi il router e'
        disattivato e la pipeline interroga gia' tutte e quattro le collezioni a ogni domanda,
        quindi di quel meccanismo resta solo la profondita' - ed e' una ragione per aspettarsi
        meno di quanto rese allora, dove due delle quattro domande recuperate erano casi di
        routing sbagliato.

        Le profondita' vengono alzate sui recuperatori esistenti e ripristinate subito dopo:
        ricostruire l'indice BM25 costerebbe il caricamento di tutti i chunk della collezione.
        """
        originali = []
        try:
            for ens in self.retrievers.values():
                for r in getattr(ens, "retrievers", []):
                    if hasattr(r, "search_kwargs") and "k" in r.search_kwargs:
                        originali.append((r, "search_kwargs", r.search_kwargs["k"]))
                        r.search_kwargs["k"] = RIPARAZIONE_K
                    elif hasattr(r, "k"):
                        originali.append((r, "k", r.k))
                        r.k = RIPARAZIONE_K
            return self._retrieve_and_format(q, selected_collections=None)
        finally:
            for r, dove, valore in originali:
                if dove == "search_kwargs":
                    r.search_kwargs["k"] = valore
                else:
                    r.k = valore

    def _retrieve_and_format(self, q: str, selected_collections: List[str] | None = None) -> dict:
        """
        Executes targeted retrieval: invokes specifically routed retrievers in parallel, 
        applies global re-ranking, and concatenates the results for prompt injection.
        
        Args:
            q (str): The vectorially optimized textual query.
            selected_collections (List[str] | None): Optional list of target collection names mapped by the router.
            
        Returns:
            dict: A mapping containing the 'context' (flattened string) and
                  'docs' (list of source objects for attribution).

        Raises:
            Exception: If asynchronous map-reduce extraction fails.
        """
        try:
            # active retrievers filtering based on the router's semantic mapping
            if selected_collections is None:
                active_retrievers = list(self.retrievers.values())
            else:
                active_retrievers = [
                    self.retrievers[name]
                    for name in selected_collections
                    if name in self.retrievers
                ]

            t0 = time.perf_counter()
            # Le collezioni restano interrogate in sequenza (non con un ThreadPoolExecutor:
            # provato il 02/10/2026, il retrieval e' risultato PIU' lento, non piu' veloce - si
            # veda il commento di _CachedQueryEmbeddings sopra per il perche'). Il risparmio
            # reale non viene dal come vengono fatte le chiamate, ma dal non ripeterle: le 4
            # collezioni condividono lo stesso embedder (self.embedding_model, avvolto nella
            # cache), quindi la query viene embeddata una sola volta invece di 4, qualunque sia
            # l'ordine delle chiamate.
            # domanda di elenco? (riconoscimento a regole, senza modello: si sa gia' prima del recupero)
            domanda_elenco = False
            if ENABLE_LIST_MODE and getattr(self, "llm_elenco", None) is not None:
                from .modalita_elenco import e_elenco
                domanda_elenco = e_elenco(q)
            limite = RERANK_CANDIDATI_ELENCHI if domanda_elenco else RERANK_CANDIDATI
            combined_docs = [doc for retriever in active_retrievers for doc in _candidati(retriever, q, limite)]
            logger.info(
                f"Retrieved (Pre-Reranking): {len(combined_docs)} documents (combined from {len(active_retrievers)} target retrievers)"
            )
            logger.info(f"[TIMING] Retrieval ({len(active_retrievers)} collezioni): {time.perf_counter() - t0:.2f}s")

            # Modalita' elenco: per le domande di elenco di entita' si aggiungono i candidati di 3
            # ricerche orientate alla fonte; il re-ranking resta rispetto alla domanda (la fusione
            # «globale»: nella stima le quote per ricerca andavano peggio) e ne tiene LIST_MODE_TOP_N.
            top_n = None
            if ENABLE_LIST_MODE and getattr(self, "llm_elenco", None) is not None:
                from .modalita_elenco import PROMPT_ELENCO, e_elenco, ricerche, unici
                if e_elenco(q):
                    t0 = time.perf_counter()
                    sotto = ricerche(self.llm_elenco.invoke(PROMPT_ELENCO.format(domanda=q)).content)
                    for s in sotto:
                        combined_docs += [doc for retriever in active_retrievers for doc in _candidati(retriever, s, limite)]
                    combined_docs = unici(combined_docs)
                    top_n = LIST_MODE_TOP_N
                    logger.info(f"[ELENCO] ricerche: {sotto}; candidati unici {len(combined_docs)}, "
                                f"ne passano {LIST_MODE_TOP_N}")
                    logger.info(f"[TIMING] Modalita' elenco (ricerche e recupero): {time.perf_counter() - t0:.2f}s")

            t0 = time.perf_counter()
            reranked = self._rerank_func(q, combined_docs, top_n=top_n)
            logger.info(f"[TIMING] Reranking: {time.perf_counter() - t0:.2f}s")

            reranked = self._filtra_pezzi(q, reranked)
            reranked = self._riassumi_pezzi(q, reranked)
            reranked = self._pota_pezzi(q, reranked)

            if ENABLE_CONTEXTUAL_COMPRESSION:
                t0 = time.perf_counter()
                before_compression = len(reranked)
                reranked = self._contextual_compression_filter(
                    q, reranked, CONTEXTUAL_COMPRESSION_THRESHOLD
                )
                logger.info(
                    f"[COMPRESSION] {before_compression} -> {len(reranked)} documenti "
                    f"dopo la contextual compression (soglia={CONTEXTUAL_COMPRESSION_THRESHOLD})"
                )
                logger.info(f"[TIMING] Contextual compression: {time.perf_counter() - t0:.2f}s")

            # Parent-child expansion: reranking scores the precise child text (d.page_content),
            # but generation gets the fuller parent_content (falls back to page_content for
            # collections built before this change, which carry no parent_id/parent_content).
            # Dedup by parent_id so two reranked children sharing one parent don't duplicate it
            # in the context string.
            # Pagine che contribuiscono con piu' di un chunk: sono elenchi spezzati, e vanno
            # passate intere (si veda ENABLE_PAGE_EXPANSION).
            conteggio_pagine = Counter(
                d.metadata.get("source_url") for d in reranked if d.metadata.get("source_url")
            )
            # Le pagine da espandere si scelgono per PERTINENZA, non per ordine di
            # presentazione: con CONTEXT_ORDER=coda il ciclo di montaggio parte dal pezzo meno
            # pertinente, e se il tetto MAX_CONTEXT si esaurisce resterebbe fuori la pagina
            # migliore. Qui si decide prima, scorrendo le pagine candidate nell'ordine del loro
            # pezzo migliore (reranked e' gia' per punteggio decrescente). Il tetto limita la
            # somma delle pagine espanse: e' una lettura deterministica, mentre quella
            # precedente dipendeva da quali genitori fossero gia' stati accodati.
            da_espandere = set()
            pagine_scelte_per_punteggio = {}
            modo_di = {}
            if ENABLE_PAGE_EXPANSION:
                prima_posizione = {}
                for posizione, doc in enumerate(reranked):
                    u = doc.metadata.get("source_url")
                    if u is not None and u not in prima_posizione:
                        prima_posizione[u] = posizione
                candidate = [u for u, n in conteggio_pagine.items()
                             if n >= PAGE_EXPANSION_MIN_CHUNKS]
                speso = 0
                for u in sorted(candidate, key=lambda x: prima_posizione.get(x, 10 ** 6)):
                    testo_pagina = self._pagina_intera(u)
                    modo = "intera"
                    if not testo_pagina and ENABLE_VICINATO:
                        # il documento e' troppo grande: invece di rinunciare si prendono i
                        # genitori adiacenti a quelli che il re-ranker ha scelto
                        scelti = {d.metadata.get("parent_id") for d in reranked
                                  if d.metadata.get("source_url") == u}
                        testo_pagina = self._vicinato(u, scelti, VICINATO_RAGGIO)
                        modo = "vicinato"
                    if not testo_pagina:
                        continue      # pagina assente, troppo grande, o senza posizioni note
                    if speso + len(testo_pagina) > PAGE_EXPANSION_MAX_CONTEXT:
                        logger.info(
                            f"[PAGE] Pagina non espansa, tetto del contesto raggiunto "
                            f"({speso} caratteri gia' spesi): {u}"
                        )
                        continue
                    da_espandere.add(u)
                    pagine_scelte_per_punteggio[u] = testo_pagina
                    modo_di[u] = modo
                    speso += len(testo_pagina)

            seen_parents = set()
            espanse = set()
            context_parts = []
            lunghezza = 0
            ordinati = self._ordina_contesto(reranked)
            if ordinati is not reranked:
                logger.info(f"[ORDINE] contesto in ordine «{CONTEXT_ORDER}»")
            for d in ordinati:
                url = d.metadata.get("source_url")
                if url in da_espandere:
                    if url in espanse:
                        continue          # la pagina e' gia' nel contesto per intero
                    testo_pagina = pagine_scelte_per_punteggio[url]
                    context_parts.append(testo_pagina)
                    lunghezza += len(testo_pagina)
                    espanse.add(url)
                    logger.info(
                        f"[PAGE] Pagina espansa in modo «{modo_di.get(url, 'intera')}» "
                        f"({conteggio_pagine[url]} chunk selezionati, "
                        f"{len(testo_pagina)} caratteri): {url}"
                    )
                    continue
                parent_id = d.metadata.get("parent_id")
                if parent_id is not None:
                    if parent_id in seen_parents:
                        continue
                    seen_parents.add(parent_id)
                testo = d.metadata.get("parent_content", d.page_content)
                context_parts.append(testo)
                lunghezza += len(testo)
            context = "\n\n".join(context_parts)
            logger.info(f"[PAGE] Contesto finale: {len(context)} caratteri, "
                        f"{len(espanse)} pagine espanse")

            return {"context": context, "docs": reranked}
        except Exception as e:
            logger.error(f"Error during retrieval or formatting: {str(e)}")
            raise

    def _verifica_nomi_risposta(self, x: dict) -> dict:
        """Verifica dei nomi sulla risposta generata (si veda ENABLE_NAME_CHECK), con il contesto da cui e' stata
        generata: la risposta corretta prende il posto di quella grezza, che resta in «answer_grezza» con l'elenco
        delle sostituzioni. Dopo una riparazione adottata il contesto della risposta e' un altro: nessuna verifica."""
        risposta, contesto = x.get("answer") or "", x.get("context") or ""
        if not contesto.strip() or x.get("riparazione") == "adottata":
            return x
        nuova, fatte = verifica_nomi(risposta, contesto, self.parole_comuni)
        if fatte:
            logger.info("[NOMI] " + "; ".join("%s -> %s" % f for f in fatte))
        return {**x, "answer": nuova, "answer_grezza": risposta, "nomi_sostituiti": fatte}

    def _build_pipeline(self):
        """
        Designs and compiles the LCEL pipeline.
        Integrates classification, Few-Shot rewriting, Routing, and Context-Augmented Generation.
        
        Args:
            None
            
        Returns:
            RunnableWithMessageHistory: The complete execution chain equipped with state management.
            
        Raises:
            None
        """
        def _timed(stage_name: str, chain):
            """Avvolge un Runnable con un checkpoint di tempo, loggato allo stesso modo delle
            altre sottofasi (guardrail/router/compression). Usato per le sottofasi il cui costo
            e' dominato da una singola chiamata LLM (rewrite, router, generazione) - retrieval/
            reranking/compression sono invece cronometrati inline in _retrieve_and_format perche'
            li' non sono Runnable invocati singolarmente."""
            def _invoke(x):
                t0 = time.perf_counter()
                result = chain.invoke(x)
                logger.info(f"[TIMING] {stage_name}: {time.perf_counter() - t0:.2f}s")
                return result
            return _invoke

        # 1. SEMANTIC GUARDRAIL (Zero-Shot Classification)
        # Bypasses expensive vector operations for out-of-domain or purely conversational intents.
        guardrail_prompt = ChatPromptTemplate.from_messages([
            ( "system",
            "Sei un filtro di classificazione semantica.\n"
            "Analizza la richiesta dell'utente e classificala in una delle seguenti tre categorie:\n\n"
            "- 'PERTINENTE': se la domanda riguarda l'Università degli Studi di Salerno in senso ampio (dipartimenti come il DIEM, organi di governo, corsi di studio, dottorati, master, tasse, diritto allo studio, servizi agli studenti, ricerca, o altri aspetti della vita universitaria). Sono PERTINENTI anche: (a) dati di contatto o identificativi (es. matricola, email, numero di telefono interno) di personale, uffici o dipartimenti dell'ateneo, anche quando la domanda nomina una persona specifica per nome invece di un ufficio; (b) enti, normative e sistemi che regolano la qualità, l'accreditamento e la valutazione del sistema universitario italiano (es. ANVUR, Nucleo di Valutazione, ACCREDIA, ESG, decreti ministeriali sull'accreditamento), anche quando la domanda non nomina esplicitamente l'Università di Salerno, perché sono il quadro normativo a cui l'ateneo stesso è soggetto; (c) la storia dell'ateneo e delle istituzioni da cui discende (lo Studio e la Scuola Medica Salernitana, il Real Liceo di Salerno, l'Istituto Superiore di Magistero), compresi i decreti, le convenzioni, i regolamenti e le delibere che le hanno fondate o trasformate, anche quando la domanda non nomina esplicitamente l'Università di Salerno; (d) i ruoli e gli incarichi delle figure istituzionali dell'ateneo (Rettore, Prorettore, delegati, direttori di dipartimento), anche quando l'incarico è ricoperto presso un altro ente.\n"
            "- 'CONVENEVOLI': se il messaggio NON CONTIENE ALTRO CHE un saluto, un ringraziamento o uno scambio di convenevoli puro (es. 'ciao', 'buongiorno', 'grazie mille', 'chi sei?').\n"
            "- 'FUORI_DOMINIO': se riguarda argomenti generali non accademici (es. ricette, politica, sport, programmazione non contestualizzata).\n\n"
            "IMPORTANTE: se il messaggio contiene un saluto o un ringraziamento SEGUITO da una domanda o richiesta reale, classifica in base al CONTENUTO della domanda (PERTINENTE o FUORI_DOMINIO), NON come CONVENEVOLI. Usa 'CONVENEVOLI' solo quando non c'è nient'altro nel messaggio oltre al saluto/ringraziamento.\n"
            "Esempio 1: 'Ciao, quali sono gli orari della biblioteca?' -> PERTINENTE (contiene un saluto ma anche una domanda reale sull'ateneo, quindi NON è CONVENEVOLI).\n"
            "Esempio 2: 'Qual è la matricola di Mario Rossi dell'Ufficio Diritto allo Studio?' -> PERTINENTE (richiesta di un dato identificativo legato a un ufficio dell'ateneo, non un argomento generico).\n"
            "Esempio 3: 'Quale decreto istituisce l'ANVUR?' -> PERTINENTE (l'ANVUR è l'ente che valuta le università italiane, incluso questo ateneo, anche se la domanda non lo nomina esplicitamente).\n"
            "Esempio 4: 'Per quale motivo il governo sabaudo decise di sopprimere il Real Liceo di Salerno?' -> PERTINENTE (il Real Liceo è una delle istituzioni da cui discende l'ateneo: è storia dell'ateneo, non storia generale).\n"
            "Esempio 5: 'Quale insegnamento tiene il Prof. Virgilio D'Antonio presso la Link Campus University?' -> PERTINENTE (riguarda un incarico del Rettore in carica, quindi una figura istituzionale dell'ateneo).\n\n"
            "Rispondi ESCLUSIVAMENTE con una di queste tre etichette testuali."
            ),
            ( "human",
            "DOMANDA: {question}\n\nCLASSIFICAZIONE:"
            )
        ])
        
        def log_guardrail(label: str, question: str) -> str:
            logger.info(f"[PIPELINE] Initial query: {question}")
            logger.info(f"[GUARDRAIL] Domain classification: {label}")
            return label.strip()

        domain_guardrail_chain = (
            RunnablePassthrough.assign(question=lambda x: x["question"])
            | guardrail_prompt
            | self.llm
            | StrOutputParser()
        )

        def domain_status_with_prefilter(x: dict) -> str:
            """Salta la chiamata LLM del guardrail se il pre-filtro rule-based riconosce
            con certezza un saluto/ringraziamento puro; altrimenti invoca il guardrail com'era."""
            t0 = time.perf_counter()
            if DOMAIN_GUARDRAIL_MODE == "off":
                logger.info("[GUARDRAIL] Disattivato (DOMAIN_GUARDRAIL_MODE=off)")
                return "PERTINENTE"
            rule_based_label = _rule_based_domain_prefilter(x["question"])
            if rule_based_label is not None:
                logger.info(
                    f"[GUARDRAIL] Pre-filtro rule-based: {rule_based_label} "
                    f"(chiamata LLM del guardrail saltata)"
                )
                logger.info(f"[TIMING] Guardrail (prefiltro): {time.perf_counter() - t0:.2f}s")
                return rule_based_label
            if DOMAIN_GUARDRAIL_MODE == "permissivo":
                logger.info(
                    "[GUARDRAIL] Modalita' permissiva: nessuna classificazione LLM, "
                    "la domanda prosegue verso il recupero"
                )
                logger.info(f"[TIMING] Guardrail (permissivo): {time.perf_counter() - t0:.2f}s")
                return "PERTINENTE"
            result = domain_guardrail_chain.invoke(x)
            logger.info(f"[TIMING] Guardrail (LLM): {time.perf_counter() - t0:.2f}s")
            return result
        
        # QUERY OPTIMIZATION
        # Resolves coreferences and enriches semantics for dense vector matching.
        rewrite_prompt = ChatPromptTemplate.from_messages([
            ("system",
             "Il tuo unico compito è estrarre i concetti chiave dalla richiesta dell'utente (o comando impartito) e riformulare la sua richiesta o comando in una domanda testuale autonoma, completa e inequivocabile. "
             "NON rispondere alla domanda dell'utente.\n\n"
             "REGOLE OPERATIVE:\n"
                "1. RISOLUZIONE DEI RIFERIMENTI: Analizza la cronologia della conversazione esclusivamente per sostituire pronomi o identificatori (es. 'questo corso', 'lui') con i referenti espliciti.\n"
                "2. CAMBIO DI ARGOMENTO: Se l'utente pone una domanda non correlata alla cronologia precedente, ignora del tutto le interazioni passate.\n"
                "3. VINCOLO DI FORMATO: Genera esclusivamente testo naturale in lingua italiana. È severamente vietato generare codice.\n"
                "4. DIVIETO DI RISPOSTA: Non fornire la risposta alla domanda o al comando.\n"
                "5. OUTPUT ISOLATO: Restituisci rigorosamente e soltanto il testo della domanda rielaborata senza annotazioni aggiuntive.\n"
                "6. ANCORAGGIO TEMPORALE: Data odierna di riferimento: {today}.\n"
                "7. ARRICCHIMENTO SEMANTICO: Se la richiesta contiene termini molto specifici, integrala in modo naturale con la macro-area accademica di riferimento per migliorare la ricerca nel database. Ad esempio: se la domanda riguarda 'corsi' o 'esami', includi termini come 'didattica' o 'offerta formativa'; se riguarda 'laboratori', includi termini come 'ricerca' o 'strutture'. I termini aggiuntivi devono fondersi fluidamente nella sintassi della domanda.\n"
             "Esempio 1:\n"
             "Input: 'come si calcola il voto di laurea magistrale?'\n"
             "Output: Come si calcola il voto di laurea in riferimento al regolamento rispetto al corso di laurea magistrale?\n\n"
             "Esempio 2:\n"
             "Input: 'chi è il rettore?'\n"
             "Output: Chi è il Rettore dell'Università degli Studi di Salerno?\n\n"
             "Esempio 3:\n"
             "Input: 'Dimmi quali strumenti ci sono nel laboratorio di robotica'\n"
             "Output: Qual è la strumentazione a disposizione del laboratorio di robotica?"
             "Esempio 4:\n"
             "Input: 'Quali corsi insegna il prof Tortorella?'\n"
             "Output: Quali corsi sono riportarti nell'area didattica del professore Tortorella?"
            ),
            ("placeholder", "{chat_history}"),
            ("human", "Input: {question}\nOutput:")
        ])

        def log_rewriter(q: str) -> str:
            q_stripped = q.strip()
            logger.info(f"[PIPELINE] Rewritten Query: {q_stripped}")
            return q_stripped
        
        rewrite_chain = (
            RunnablePassthrough.assign(today=lambda _: data_odierna())
            | rewrite_prompt 
            | self.llm 
            | StrOutputParser() 
            | RunnableLambda(log_rewriter)
        )

        # SEMANTIC ROUTER
        # Maps the optimized query to the most relevant logical Chroma collections.
        router_prompt = ChatPromptTemplate.from_messages([
            ("system",
             "Sei un router per un sistema RAG dell'Università degli Studi di Salerno.\n"
             "Decidi quali collezioni Chroma devono essere interrogate per la domanda data.\n"
             "Restituisci un oggetto JSON che mappa ogni nome di collezione a true o false.\n"
             "Restituisci solo JSON, nessun altro testo.\n\n"
             "Collezioni e descrizioni:\n"
             "{collection_descriptions}\n\n"
             "Nomi delle collezioni: {collection_names}\n"
             "Regole:\n"
             "- Usa solo i nomi di collezione forniti.\n"
             "- Includi sempre ogni nome di collezione nell'output JSON.\n"
             "- L'output deve essere un JSON valido con valori booleani.\n"
             "- IN CASO DI DUBBIO: se non sei certo che una collezione sia pertinente ma la sua "
             "descrizione copre anche solo in parte l'argomento della domanda, includila (true). "
             "Un falso positivo (collezione inclusa ma non necessaria) costa solo qualche "
             "candidato in più scartato dal reranking successivo; un falso negativo (collezione "
             "pertinente esclusa) produce una risposta mancata o sbagliata - è un errore molto "
             "più grave, da evitare anche a costo di includere qualche collezione in più del "
             "necessario.\n\n"
             "Esempi:\n"
             "Query: 'Chi è Pasquale Stanzione e a quale Dipartimento afferisce?' -> la domanda "
             "riguarda una persona e non nomina esplicitamente 'organi di governo' o "
             "'personale': includi comunque (true) qualunque collezione la cui descrizione copra "
             "organi di governo, amministrazione centrale o personale dell'ateneo.\n"
             "Query: 'Qual è l'indirizzo email dell'AREA VI - Risorse Strumentali?' -> riguarda "
             "un contatto istituzionale specifico non menzionato testualmente in nessuna "
             "descrizione: includi comunque (true) la collezione che copre amministrazione "
             "centrale/contatti istituzionali.\n"
             "Query: 'Quali sono i requisiti di immatricolazione per la laurea magistrale in "
             "Ingegneria Informatica?' -> riguarda chiaramente le immatricolazioni: includi "
             "(true) solo la collezione dedicata a immatricolazioni/dottorati/master/scuole di "
             "specializzazione, escludi (false) le altre se le loro descrizioni non sono "
             "pertinenti."
             ),
            ("human", "Query: {optimized_question}\nJSON:")
        ])

        # def parse_router_output(raw: str, collections: List[str]) -> Dict[str, bool]:
        #     """Parses the LLM's constrained output into a strict boolean routing matrix."""
        #     try:
        #         data = json.loads(raw)
        #         if not isinstance(data, dict):
        #             raise ValueError("Router output is not a JSON object.")
        #     except Exception as e:
        #         logger.warning(f"[ROUTER] Invalid JSON output fallback: {str(e)}")
        #         return {name: False for name in collections}

        #     parsed = {}
        #     for name in collections:
        #         val = data.get(name, False)
        #         if isinstance(val, bool):
        #             parsed[name] = val
        #         elif isinstance(val, str):
        #             parsed[name] = val.strip().lower() in {"true", "yes", "1"}
        #         else:
        #             parsed[name] = False
        #     return parsed
        def parse_router_output(raw: str, collections: List[str]) -> Dict[str, bool]:
            """Parses the LLM's constrained output into a strict boolean routing matrix."""
            try:
                # Estrae il primo oggetto JSON trovato, ignorando eventuali backtick markdown
                import re
                match = re.search(r'\{.*\}', raw, re.DOTALL)
                if not match:
                    raise ValueError("No JSON object found in router output.")
                data = json.loads(match.group())
                if not isinstance(data, dict):
                    raise ValueError("Router output is not a JSON object.")
            except Exception as e:
                logger.warning(f"[ROUTER] Invalid JSON output fallback: {str(e)}")
                return {name: False for name in collections}

            parsed = {}
            for name in collections:
                val = data.get(name, False)
                if isinstance(val, bool):
                    parsed[name] = val
                elif isinstance(val, str):
                    parsed[name] = val.strip().lower() in {"true", "yes", "1"}
                else:
                    parsed[name] = False
            return parsed

        def select_collections(route_map: Dict[str, bool], collections: List[str]) -> List[str]:
            return [name for name in collections if route_map.get(name, False)]

        def log_router(raw: str) -> str:
            logger.info(f"[ROUTER] Raw logical matrix: {raw}")
            return raw

        def bypass_router(collection_names: List[str]) -> Dict[str, bool]:
            """
            Con una sola collezione attiva non c'e' nessuna scelta da instradare: verificato
            empiricamente (2026-07-27) che in questo caso il router LLM introduce solo falsi
            negativi (risponde 'false' anche a domande chiaramente pertinenti, es. "Chi e'
            Pasquale Stanzione..." con OLLAMA_ENABLE_THINKING=false - 5/5 casi diagnosticati
            avrebbero avuto retrieval e reranking corretti se solo il router li avesse lasciati
            passare). Riattivare il thinking solo per il router risolveva la misclassificazione
            ma costava 46-102s a chiamata (verificato), incompatibile con l'obiettivo di
            latenza - qui invece si salta del tutto la chiamata LLM quando non c'e' nessuna
            reale ambiguita' da risolvere. Con piu' collezioni attive (deployment futuro
            multi-dipartimento) il router LLM torna ad essere usato normalmente, invariato.
            """
            logger.info(
                f"[ROUTER] Bypass attivo ({len(collection_names)} collezioni: {', '.join(collection_names)}) "
                f"- nessuna chiamata LLM, il filtro e' affidato al re-ranking."
            )
            return {name: True for name in collection_names}

        router_chain = (
            RunnablePassthrough.assign(
                collection_names=lambda _: ", ".join(self.retrievers.keys()),
                collection_list=lambda _: list(self.retrievers.keys()),
                collection_descriptions=lambda _: self._build_router_collection_descriptions(),
            )
            | RunnableBranch(
                (
                    lambda x: (not ENABLE_SEMANTIC_ROUTER) or len(x["collection_list"]) == 1,
                    RunnableLambda(lambda x: bypass_router(x["collection_list"])),
                ),
                (
                    RunnablePassthrough.assign(
                        routing_raw=(
                            router_prompt
                            | self.llm
                            | StrOutputParser()
                            | RunnableLambda(log_router)
                        )
                    )
                    | RunnableLambda(lambda x: parse_router_output(x["routing_raw"], x["collection_list"]))
                ),
            )
        )

        # ANSWER GENERATOR 
        # Synthesizes the final output while strongly conditioning the LLM to avoid hallucinations.
        answer_prompt = ChatPromptTemplate.from_messages([
            ( "system",
            "Sei l'Assistente Virtuale Ufficiale dell'Università degli Studi di Salerno.\n"
            + APERTURA_RISPOSTA
            + "REGOLE DI COMPORTAMENTO RIGIDE:\n"
            + "1. IMMEDESIMAZIONE TOTALE: Tratta le informazioni a tua disposizione come tua conoscenza diretta. È SEVERAMENTE VIETATO usare espressioni come 'in base ai documenti', 'secondo il testo', o 'i pdf dicono'. Rispondi in modo naturale e autorevole.\n"
            + REGOLA_FOCUS
            + "3. UTILIZZO DELLA CRONOLOGIA: Leggi la cronologia della conversazione SOLO per comprendere il contesto dei pronomi. Non farti influenzare dagli argomenti passati per generare la risposta attuale.\n"
            + REGOLA_COPERTURA
            + "5. ADERENZA AI FATTI: Non inventare o dedurre nulla.\n"
            + "6. CONSAPEVOLEZZA TEMPORALE: Oggi è il {today}. Adeguati ai tempi verbali per eventi passati o futuri.\n"
            + REGOLA_INSUFFICIENTE
            + "8. AMBITO CORRETTO: Non dare per scontato che una domanda riguardi il dipartimento DIEM specificamente, a meno che il CONTESTO o la DOMANDA UTENTE non lo indichino esplicitamente. Molte domande riguardano l'Ateneo nel suo complesso (es. tasse, borse di studio, corsi di laurea di tutti i dipartimenti, biblioteche, mensa) e vanno trattate come tali, senza attribuirle al DIEM o rimandare genericamente al 'sito web del DIEM' per informazioni non DIEM-specifiche.\n"
            + "9. PRECISIONE SU ELENCHI E TABELLE CON PIÙ VOCI SIMILI: Se il CONTESTO contiene un elenco o una tabella con più persone, ruoli o voci simili tra loro (es. più direttori di dipartimento, più componenti di un organo, più uffici con capi diversi), individua SEMPRE la riga o voce che corrisponde ESATTAMENTE al nome, dipartimento o identificativo specifico indicato nella DOMANDA UTENTE, prima di rispondere. Non restituire una voce diversa solo perché sembra plausibile, è la più vicina nel testo, o riguarda una persona con un ruolo simile altrove nel CONTESTO. Se non riesci a individuare con certezza la voce corrispondente, applica la regola 7 (ammetti la mancanza di informazioni) invece di scegliere una voce a caso. Inoltre, non contraddire nella stessa risposta un'affermazione fatta in precedenza nella risposta stessa: se non sei certo di un ruolo o di un fatto, fai una sola affermazione coerente, non due in conflitto tra loro."
            + REGOLA_SINONIMI
            ),
            ( "placeholder", "{chat_history}" ),
            ( "human",
            "INFORMAZIONI A TUA DISPOSIZIONE:\n---\n"
            "CONTESTO:\n{context}\n\n"
            "DOMANDA UTENTE:\n{question}\n\n"
            "RISPOSTA:"
            )
        ])

        # Risposta fissa quando il retrieval non restituisce alcun documento: verificato empiricamente
        # (test manuali del 2026-07-21) che affidarsi solo alla regola 7 del prompt sopra NON basta -
        # llama3.1:8b continua a inventare risposte dettagliate e plausibili anche con contesto vuoto.
        # Questo branch bypassa del tutto la chiamata all'LLM in quel caso, invece di sperare che
        # segua l'istruzione.
        NO_CONTEXT_FALLBACK_ANSWER = (
            "Al momento non dispongo di informazioni su questo argomento specifico. "
            "Ti consiglio di contattare gli uffici competenti o consultare il sito ufficiale dell'Ateneo."
        )

        # La generazione e' identica per i due percorsi, quindi vive in un pezzo riusabile.
        #
        # Reintrodotta il 05/10/2026 la misura di TTFT e TPOT (presente nel giro di baseline_latenza_630,
        # persa in una modifica successiva). La risposta viene generata in STREAMING invece che con
        # .invoke(): il testo finale e' identico (concatenazione dei pezzi = quello che restituiva
        # StrOutputParser), ma cosi' si vede quando arriva il primo pezzo.
        #
        # Righe scritte nel log (tutte in secondi, nel formato "[TIMING] <nome>: <valore>s" letto da
        # analyze_timing.py e analizza_stadi_per_livello.py):
        #   - Generazione risposta: tempo totale della chiamata, come prima (confrontabile coi giri A-D).
        #   - TTFT: tempo dall'invio della richiesta al primo pezzo di testo, misurato DAL CLIENT. E'
        #     la latenza percepita dall'utente (e dal robot): include l'eventuale attesa in coda su
        #     Ollama quando piu' utenti chiedono insieme.
        #   - TTFT (server): load_duration + prompt_eval_duration riportati da Ollama. Parte da quando
        #     il server inizia davvero a lavorare sulla richiesta, quindi NON include la coda.
        #   - Attesa in coda (stima): TTFT - TTFT (server) - TPOT (il primo token generato e la rete
        #     sono dentro il TTFT del client ma non in quello del server). E' la misura diretta della
        #     serializzazione: a 1 utente deve stare vicino a 0, se cresce con gli utenti le richieste
        #     si accodano.
        #   - TPOT: eval_duration / eval_count riportati da Ollama (tempo medio per token generato,
        #     senza coda e senza prompt).
        # Con il backend HF (niente metadati di Ollama) vengono scritti solo Generazione risposta e TTFT.
        # Con il backend vLLM (dal 07/10/2026) il server non riporta i propri tempi: si scrivono
        # Generazione risposta, TTFT (client, coda compresa), TPOT misurato dal client
        # ((fine - primo token) / (token generati - 1)) e i contatori [TOKENS]. Mancano TTFT (server)
        # e Attesa in coda, che con vLLM non si possono separare dal client.
        def genera_misurata(x: dict, suffisso: str = "") -> str:
            messaggi = answer_prompt.invoke(x)
            t0 = time.perf_counter()
            t_primo = None
            parti = []
            meta = {}
            for chunk in self.llm.stream(messaggi):
                testo = chunk.content if isinstance(chunk.content, str) else str(chunk.content or "")
                if testo:
                    if t_primo is None:
                        t_primo = time.perf_counter()
                    parti.append(testo)
                # l'ultimo pezzo inviato da Ollama ("done") porta i contatori della richiesta
                rm = getattr(chunk, "response_metadata", None) or {}
                if rm.get("eval_count") is not None:
                    meta = rm
                # vLLM (API OpenAI): i contatori arrivano in usage_metadata dell'ultimo pezzo
                um = getattr(chunk, "usage_metadata", None)
                if um and um.get("output_tokens") is not None:
                    meta = {"usage": um}
            t_fine = time.perf_counter()

            logger.info(f"[TIMING] Generazione risposta{suffisso}: {t_fine - t0:.2f}s")
            ttft = (t_primo - t0) if t_primo is not None else None
            if ttft is not None:
                logger.info(f"[TIMING] TTFT{suffisso}: {ttft:.3f}s")

            n_out = meta.get("eval_count") or 0
            eval_ns = meta.get("eval_duration") or 0
            if n_out > 0 and eval_ns > 0:
                tpot = eval_ns / n_out / 1e9
                ttft_server = ((meta.get("load_duration") or 0) + (meta.get("prompt_eval_duration") or 0)) / 1e9
                logger.info(f"[TIMING] TPOT{suffisso}: {tpot:.4f}s")
                logger.info(f"[TIMING] TTFT (server){suffisso}: {ttft_server:.3f}s")
                if ttft is not None:
                    logger.info(f"[TIMING] Attesa in coda (stima){suffisso}: {max(0.0, ttft - ttft_server - tpot):.3f}s")
                # contatori di token: non sono tempi, quindi fuori dal formato [TIMING]
                logger.info(f"[TOKENS]{suffisso} prompt={meta.get('prompt_eval_count')} output={n_out}")
            elif "usage" in meta:
                n_out = meta["usage"].get("output_tokens") or 0
                if t_primo is not None and n_out > 1:
                    logger.info(f"[TIMING] TPOT{suffisso}: {(t_fine - t_primo) / (n_out - 1):.4f}s")
                logger.info(f"[TOKENS]{suffisso} prompt={meta['usage'].get('input_tokens')} output={n_out}")
            return "".join(parti)

        # Ragionamento mirato (ENABLE_RAGIONAMENTO_MIRATO): per i tipi di domanda in RAGIONAMENTO_TIPI la risposta
        # si genera con il modello che ragiona. Spento, genera_ragionata e' None e tutto resta come prima.
        genera_ragionata = (answer_prompt | self.llm_ragionamento
                            if getattr(self, "llm_ragionamento", None) is not None else None)

        def genera(x: dict, ragiona: bool, suffisso: str = ""):
            """(risposta, informazioni sul ragionamento). Con il ragionamento acceso il suo testo resta fuori dalla
            risposta (langchain-ollama lo mette in additional_kwargs) e se ne registrano lunghezza, motivo di fine
            e tempo: «length» vuol dire che il tetto di token e' finito prima della risposta.
            Senza ragionamento la risposta passa da genera_misurata (07/10/2026: TTFT, TPOT e token nel log)."""
            if not ragiona:
                return genera_misurata(x, suffisso), {"usato": False}
            t0 = time.perf_counter()
            msg = genera_ragionata.invoke(x)
            ragionamento = (msg.additional_kwargs or {}).get("reasoning_content") or ""
            info = {"usato": True, "caratteri": len(ragionamento),
                    "fine": (msg.response_metadata or {}).get("done_reason"),
                    "secondi": round(time.perf_counter() - t0, 1)}
            logger.info(f"[RAGIONAMENTO] {info}")
            # con il ragionamento niente streaming: solo il tempo totale, come prima
            logger.info(f"[TIMING] Generazione risposta{suffisso}: {time.perf_counter() - t0:.2f}s")
            return msg.content, info

        def genera_con_riparazione(x: dict) -> dict:
            ragiona = genera_ragionata is not None and classifica(x["question"])[0] in RAGIONAMENTO_TIPI
            risposta, info = genera(x, ragiona)
            x = {**x, "ragionamento": info}
            innescata = RIPARAZIONE_INNESCO == "sempre" or _e_rifiuto(risposta)
            if not (ENABLE_RIPARAZIONE and innescata):
                return {**x, "answer": risposta}

            logger.info("[RIPARAZIONE] la risposta dichiara di non sapere: riprovo a k=%d" % RIPARAZIONE_K)
            ampio = self._recupero_ampio(x["question"])
            if not ampio["context"].strip() or ampio["context"] == x["context"]:
                logger.info("[RIPARAZIONE] nessun contesto nuovo: tengo la prima risposta")
                return {**x, "answer": risposta, "riparazione": "nessun_contesto_nuovo"}

            seconda, _ = genera({**x, "context": ampio["context"]}, ragiona, suffisso=" (riparazione)")

            if _e_rifiuto(seconda):
                logger.info("[RIPARAZIONE] anche il secondo tentativo e' un rifiuto: tengo il primo")
                return {**x, "answer": risposta, "riparazione": "rifiuto_confermato"}

            if RIPARAZIONE_MODO == "verifica":
                quota = _quota_ancorata(seconda, ampio["context"])
                if quota < RIPARAZIONE_ANCORAGGIO:
                    logger.info("[RIPARAZIONE] seconda risposta ancorata solo al %.0f%%: la scarto "
                                "e tengo il rifiuto" % (quota * 100))
                    return {**x, "answer": risposta, "riparazione": "scartata_non_ancorata"}
                logger.info("[RIPARAZIONE] seconda risposta ancorata al %.0f%%: la adotto" % (quota * 100))

            # "context" aggiornato: la risposta adottata e' stata generata sul contesto ampio
            return {**x, "answer": seconda, "sources": ampio["docs"], "context": ampio["context"],
                    "riparazione": "adottata"}

        genera_chain = RunnableBranch(
            (
                lambda x: not x["context"].strip(),
                RunnableLambda(lambda x: {**x, "answer": NO_CONTEXT_FALLBACK_ANSWER}),
            ),
            RunnableLambda(genera_con_riparazione),
        )
        # Verifica dei nomi (ENABLE_NAME_CHECK) in coda alla generazione, per tutti i percorsi (persona, struttura,
        # catena normale): la cronologia salva la risposta corretta. Spenta, la catena e' quella di prima.
        if ENABLE_NAME_CHECK:
            genera_chain = genera_chain | RunnableLambda(self._verifica_nomi_risposta)

        def con_scheda(persona, contesto):
            """Mette la scheda dell'anagrafe in testa al contesto recuperato, senza togliere
            nulla: i dati certi della rubrica restano disponibili al generatore insieme alle
            pagine che contengono gli incarichi (direzioni, deleghe, storia)."""
            if not (PERSON_CONTEXT_BOOST and persona):
                return contesto
            blocco = ("SCHEDA ANAGRAFICA UFFICIALE (rubrica di ateneo)" + chr(10)
                      + persona["context"] + chr(10) + "FINE DELLA SCHEDA ANAGRAFICA")
            return (blocco + chr(10) + chr(10) + contesto) if contesto.strip() else blocco

        def con_struttura(struttura, contesto):
            """Percorso «struttura» in modalita' rinforzo: la scheda del registro in testa al
            contesto recuperato, senza togliere nulla. Senza struttura riconosciuta - e quindi
            sempre a interruttore spento - il contesto resta quello di prima."""
            if not struttura:
                return contesto
            return (struttura["context"] + chr(10) + chr(10) + contesto) if contesto.strip() else struttura["context"]

        # Percorso veloce: nessun recupero, il contesto e' la sola scheda dell'anagrafe. Ci si
        # arriva solo per le domande su un campo che la scheda contiene con certezza.
        persona_chain = (
            RunnableLambda(
                lambda x: {
                    "context": x["persona"]["context"],
                    "question": x["question"],
                    "sources": x["persona"]["docs"],
                    "chat_history": x["chat_history"],
                    "today": data_odierna()
                }
            )
            | genera_chain
        )

        # Percorso «struttura» veloce: come quello persona, il contesto e' la sola scheda del
        # registro. Ci si arriva solo a interruttore acceso (si veda dopo_persona, piu' sotto).
        struttura_chain = (
            RunnableLambda(
                lambda x: {
                    "context": x["struttura"]["context"],
                    "question": x["question"],
                    "sources": x["struttura"]["docs"],
                    "chat_history": x["chat_history"],
                    "today": data_odierna()
                }
            )
            | genera_chain
        )

        rag_chain = (
            RunnablePassthrough.assign(
                retrieved_data=lambda x: self._retrieve_and_format(
                    x["optimized_question"],
                    selected_collections=x.get("selected_collections")
                )
            )
            | RunnableLambda(
                lambda x: {
                    "context": con_struttura(x.get("struttura"),
                                             con_scheda(x.get("persona"), x["retrieved_data"]["context"])),
                    "question": x["question"],
                    "sources": ((x["struttura"]["docs"] if x.get("struttura") else [])
                                + (x["persona"]["docs"] if (PERSON_CONTEXT_BOOST and x.get("persona"))
                                   else []) + x["retrieved_data"]["docs"]),
                    "chat_history": x["chat_history"],
                    "today": data_odierna()
                }
            )
            | genera_chain
        )

        # Logical subchain to intercept non-document-related intents 
        out_of_domain_chain = RunnableLambda(
            lambda _: {
                "answer": "Mi dispiace, ma posso rispondere solo a domande relative all'Università degli Studi di Salerno, ai suoi dipartimenti, corsi di studio e servizi.",
                "sources": [],
            }
        )

        greeting_chain = RunnableLambda(
            lambda _: {
                "answer": "Ciao! Sono l'Assistente Virtuale dell'Università degli Studi di Salerno. Posso aiutarti a trovare informazioni su corsi, dipartimenti, servizi, regolamenti e vita universitaria. Come posso esserti utile?",
                "sources": [],
            }
        )

        def rewrite_if_history(x: dict) -> str:
            """Salta la riscrittura quando non c'e' cronologia da cui risolvere riferimenti.
            Vedi il commento su SKIP_REWRITE_WHEN_NO_HISTORY in cima al modulo."""
            history = x.get("chat_history") or []
            if SKIP_REWRITE_WHEN_NO_HISTORY and not history:
                q = x["question"].strip()
                logger.info(f"[PIPELINE] Initial query: {q}")
                logger.info("[REWRITE] Cronologia vuota: riscrittura saltata, uso la domanda originale.")
                logger.info("[TIMING] Query rewriting (saltata): 0.00s")
                return q
            return _timed("Query rewriting", rewrite_chain)(x)

        # La catena normale, cioe' tutto cio' che segue il percorso persona quando questo non
        # risponde da solo: riscrittura, router, recupero e generazione. Identica a prima.
        catena_normale = (
            RunnablePassthrough.assign(optimized_question=RunnableLambda(
                lambda x: x["optimized_question"] if x.get("optimized_question") is not None
                else rewrite_if_history(x)))
            | RunnablePassthrough.assign(routing_map=_timed("Router", router_chain))
            | RunnablePassthrough.assign(
                selected_collections=lambda x: select_collections(
                    x["routing_map"], list(self.retrievers.keys())
                )
            )
            | rag_chain
        )
        # A interruttore spento dopo il percorso persona viene la catena normale, esattamente come
        # prima: il ramo della struttura non viene costruito. Acceso, si cerca la struttura solo
        # se il percorso persona non ha riconosciuto nessuno (la persona ha la precedenza).
        if ENABLE_STRUCT_PATH:
            dopo_persona = (
                RunnablePassthrough.assign(
                    struttura=RunnableLambda(
                        lambda x: None if x.get("persona") else self._cerca_struttura(x["optimized_question"]))
                )
                | RunnableBranch(
                    (lambda x: bool(x.get("struttura")) and x["struttura"]["veloce"], struttura_chain),
                    catena_normale,
                )
            )
        else:
            dopo_persona = catena_normale

        # Main full chain
        main_chain = (
            RunnablePassthrough.assign(domain_status=RunnableLambda(domain_status_with_prefilter))
            | RunnableBranch(
                (lambda x: "FUORI_DOMINIO" in x["domain_status"].upper(), out_of_domain_chain),
                (lambda x: "CONVENEVOLI" in x["domain_status"].upper(), greeting_chain),
                (
                    RunnablePassthrough.assign(
                        optimized_question=RunnableLambda(rewrite_if_history)
                    ) | RunnablePassthrough.assign(
                        persona=RunnableLambda(lambda x: self._cerca_persona(x["optimized_question"]))
                    )
                    | RunnableBranch(
                        (lambda x: bool(x.get("persona")) and x["persona"]["veloce"], persona_chain),
                        dopo_persona,
                    )
                )
            )
        )
        
        # Abstraction for conversational session state persistence
        def get_session_history(session_id: str):
            # Ripristinato il 2026-08-28 alla versione della basecode originale: la riga che
            # azzerava la cronologia a ogni chiamata rendeva il bot privo di memoria conversazionale
            # (in contrasto con la documentazione) e svuotava di scopo la query rewriting.
            if session_id not in self.chat_histories:
                self.chat_histories[session_id] = InMemoryChatMessageHistory()
            return self.chat_histories[session_id]

        # ari_servers: the pieces of the chain used by ../rag.py, which runs the retrieval of main_chain step by
        # step and leaves the generation (answer_prompt | llm) and the history to the LLM server
        self._domain_status = domain_status_with_prefilter
        self._rewrite_if_history = rewrite_if_history
        self._router_chain = router_chain
        self._select_collections = select_collections
        self._answer_prompt = answer_prompt
        self._con_scheda = con_scheda
        self._con_struttura = con_struttura
        self.NO_CONTEXT_FALLBACK_ANSWER = NO_CONTEXT_FALLBACK_ANSWER
        self.OUT_OF_DOMAIN_ANSWER = out_of_domain_chain.invoke(None)["answer"]
        self.GREETING_ANSWER = greeting_chain.invoke(None)["answer"]

        return RunnableWithMessageHistory(
            main_chain,
            get_session_history,
            input_messages_key="question",
            output_messages_key="answer",
            history_messages_key="chat_history"
        )

    def retrieve_and_generate(self, user_input: str, session_id: str) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Public entry point of the architecture. Invokes the LCEL pipeline, 
        manages user tracking, and serializes document source metadata.
        
        Args:
            user_input (str): The raw string command or query submitted by the client.
            session_id (str): A unique string identifier binding the execution context to a user session.
            
        Returns:
            Tuple[str, List[Dict[str, Any]]]: 
                - The deterministic string response generated by the pipeline.
                - A serialized dictionary list of referenced chunks, mapping keys:
                  ['url', 'score', 'title', 'content'].
            
        Raises:
            ValueError: If the input tensor vector is None or empty.
        """
        risultato = self.retrieve_and_generate_con_contesto(user_input, session_id)
        return risultato[0], risultato[1]

    def retrieve_and_generate_con_contesto(self, user_input: str, session_id: str) -> Tuple[str, List[Dict[str, Any]], str, str]:
        """Come retrieve_and_generate, ma restituisce anche il CONTESTO esatto passato al modello
        per generare la risposta (stringa vuota se la domanda non e' arrivata alla generazione:
        saluti, fuori dominio, retrieval vuoto) e la RISPOSTA GREZZA, cioe' quella scritta dal
        modello prima della verifica dei nomi (uguale alla risposta finale se la verifica e' spenta
        o non ha sostituito nulla).

        Aggiunto il 05/10/2026 per stimare offline il potenziale del Prompt Lookup Decoding
        (stima_pld.py): il PLD copia sequenze di token dal prompt, quindi serve il testo che il
        modello ha visto davvero, e va confrontato con il testo che il modello ha generato davvero
        (la grezza: le sostituzioni della verifica dei nomi avvengono dopo la generazione). Le
        "fonti" non bastano: contengono il chunk figlio (page_content), mentre il prompt riceve il
        genitore (parent_content), le pagine espanse e l'eventuale scheda dell'anagrafe.
        retrieve_and_generate resta invariato per tutti i chiamanti esistenti (main.py, script di
        test). Restituisce i valori come variabili locali, quindi funziona anche con piu' utenti in
        parallelo (a differenza di self.ultimo_risultato, che viene sovrascritto dall'ultima
        richiesta finita).
        """
        if not user_input or not user_input.strip():
            raise ValueError("[ERROR] The input logical payload cannot be empty.")
            
        logger.info(f"[INFO] RAG pipeline invoked for Session -{session_id}-")
        
        # Execute the masked graph via LangChain abstraction
        result = self.conversational_pipeline.invoke(
            {"question": user_input},
            config={"configurable": {"session_id": session_id}}
        )
        self.ultimo_risultato = result   # per le misure: risposta grezza, sostituzioni, contesto
        contesto = result.get("context") or ""
        
        answer = result.get('answer', "Mi dispiace, non sono riuscito a generare una risposta. Riprova.")
        raw_sources = result.get('sources', [])
        
        # Structure and flatten document metadata matrices for frontend consumability
        formatted_sources = []
        for doc in raw_sources:
            source_data = {
                "content": doc.page_content,
                "url": doc.metadata.get('source_url', 'URL Non Disponibile'),
                "score": doc.metadata.get('relevance_score', 0.0),
                "title": doc.metadata.get('title', 'Titolo Non Specificato') 
            }
            formatted_sources.append(source_data)
            
        return answer, formatted_sources, contesto, result.get("answer_grezza", answer)


# Local test execution block
if __name__ == "__main__":
    rag_engine = DiemRAG()
    
    chat_id = "telegram_user_12345"
    domanda = "Quali sono i contatti del responsabile per l'internazionalizzazione al DIEM?"
    
    risposta, documenti = rag_engine.retrieve_and_generate(domanda, chat_id)
    
    logger.info(f"\n[BOT RESPONSE]\n{risposta}")
    logger.info(f"\n[BOT SOURCES]")
    for d in documenti:
        logger.info(f"- {d['url']} (Score: {d['score']:.3f})")