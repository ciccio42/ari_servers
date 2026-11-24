# Copied from DIEM_RAG/diem_RAG_routered.py of the RAG repository of the DIEM department (course Generative AI,
# local version) for ari_servers: no .env, the index, the model and the Ollama server from the environment set by
# ../rag.py (config/servers.yaml), the model on Ollama (ChatOllama) instead of transformers in this process (the
# same Qwen3-30B-A3B-Instruct-2507, fp16), and the pieces of the LCEL chain kept on the instance (_build_pipeline,
# end) so that ../rag.py runs the retrieval without the generation, which is done by the LLM server.
# Everything else is unchanged.
from datetime import datetime
import json
import os
from pathlib import Path
import logging
from typing import Tuple, List, Dict, Any

from langchain_chroma import Chroma
from sentence_transformers import CrossEncoder
from langchain_core.runnables import RunnableBranch, RunnableLambda, RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
import chromadb

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
PERSIST_DIRECTORY = Path(os.getenv("VECTORSTORE_DIR", str(BASE_DIR / "diem_vectorstore")))

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
IMPORTANTE: LE INFORMAZIONE RELATIVE A POSIZIONE E CAPIENZA DELLE AULE DIDATTICHE SONO PRESENTI NELLE COLLEZIONI DEI SINGOLI CORSI DI LAUREA.
IMPORTANTE 2: CONSIDERA I DOCUMENTI PIÙ RECENTI COME PIÙ RILEVANTI IN CASO DI INFORMAZIONI CONTRADDITORIE (es. REQUISITI DI IMMATRICOLAZIONE, VOTO DI LAUREA).
"""

# The model of the original: Qwen/Qwen3-30B-A3B-Instruct-2507 in transformers (bf16, greedy, 4096 new tokens). In
# ari_servers the same model on Ollama, fp16, temperature 0, with the same num_ctx for every request (another num_ctx
# makes Ollama load the model again)
OLLAMA_MODEL_ID = os.getenv("OLLAMA_MODEL_ID", "qwen3:30b-a3b-instruct-2507-fp16")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11500")
OLLAMA_NUM_CTX = int(os.getenv("OLLAMA_NUM_CTX", "16384"))


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
            
            # Binding to the persistent ChromaDB instance and dynamically loading all collections for retrieval
            self.client = chromadb.PersistentClient(path=str(PERSIST_DIRECTORY))
            collections = self.client.list_collections()
            self.retrievers = {}
            
            for c in collections:
                self.retrievers[c.name] = self._retrieve_vectorstore(
                    vs_dir=PERSIST_DIRECTORY, 
                    collection_name=c.name
                )

            logger.info("Reranker initialization...")
            
            # Cross-Encoder weights loading for semantic re-ranking
            self.reranker = CrossEncoder("BAAI/bge-reranker-v2-m3")

            logger.info("LLM initialization...")

            # LLM initialization: domain filter, rewriting and router (the answer is generated by the LLM server)
            self.llm = ChatOllama(
                model=OLLAMA_MODEL_ID, base_url=OLLAMA_BASE_URL, num_ctx=OLLAMA_NUM_CTX,
                temperature=0, num_predict=4096,
            )
            
            # In-memory dictionary for conversational state management
            self.chat_histories = {}
            
            self.conversational_pipeline = self._build_pipeline()
            logger.info("[SUCCESS] RAG ready to go...")
        except Exception as e:
            logger.error(f"Rag initialization failed: {str(e)}")
            raise

    def _retrieve_vectorstore(self, vs_dir: Path, collection_name: str):
        """
        Defines and configures a LangChain retriever instance for a specific ChromaDB collection.
        
        Args:
            vs_dir (Path): The absolute path to the persistent vectorstore directory.
            collection_name (str): The unique identifier of the target collection.
            
        Returns:
            VectorStoreRetriever: An instantiated LangChain retriever configured to extract the top K chunks.
            
        Raises:
            Exception: If an I/O error occurs or the specified collection cannot be loaded.
        """
        try:
            vectorstore = Chroma(
                collection_name=collection_name,
                embedding_function=self.embedding_model,
                persist_directory=str(vs_dir),
            )
            # Retrieve top-K candidates 
            return vectorstore.as_retriever(search_kwargs={"k": 20})
        except Exception as e:
            logger.error(f" Error loading vectorstore for collection '{collection_name}': {str(e)}")
            raise

    def _rerank_func(self, query: str, documents: list, top_n: int = 7) -> list:
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
            return out
        except Exception as e:
            logger.error(f"[ERROR] Error during reranking: {str(e)}")
            raise
        
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

            combined_docs = [doc for retriever in active_retrievers for doc in retriever.invoke(q)]
            logger.info(
                f"Retrieved (Pre-Reranking): {len(combined_docs)} documents (combined from {len(active_retrievers)} target retrievers)"
            )
            
            reranked = self._rerank_func(q, combined_docs)
            context = "\n\n".join(d.page_content for d in reranked)
            
            return {"context": context, "docs": reranked}
        except Exception as e:
            logger.error(f"Error during retrieval or formatting: {str(e)}")
            raise

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
        # 1. SEMANTIC GUARDRAIL (Zero-Shot Classification)
        # Bypasses expensive vector operations for out-of-domain or purely conversational intents.
        guardrail_prompt = ChatPromptTemplate.from_messages([
            ( "system",
            "Sei un filtro di classificazione semantica.\n"
            "Analizza la richiesta dell'utente e classificala in una delle seguenti tre categorie:\n\n"
            "- 'PERTINENTE': se la domanda riguarda il Dipartimento di Ingegneria (DIEM), l'Università di Salerno, corsi, docenti, laboratori o vita universitaria.\n"
            "- 'CONVENEVOLI': se l'utente sta semplicemente salutando, ringraziando o scambiando convenevoli (es. 'ciao', 'buongiorno', 'grazie mille', 'chi sei?').\n"
            "- 'FUORI_DOMINIO': se riguarda argomenti generali non accademici (es. ricette, politica, sport, programmazione non contestualizzata).\n\n"
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
             "Input: 'chi è il direttore del diem?'\n"
             "Output: Chi è il direttore del dipartimento di ingegneria dell'informazione ed elettrica e matematica applicata (DIEM)?\n\n"
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
            RunnablePassthrough.assign(today=lambda _: datetime.now().strftime("%d/%m/%Y"))
            | rewrite_prompt 
            | self.llm 
            | StrOutputParser() 
            | RunnableLambda(log_rewriter)
        )

        # SEMANTIC ROUTER
        # Maps the optimized query to the most relevant logical Chroma collections.
        router_prompt = ChatPromptTemplate.from_messages([
            ("system",
             "Sei un router per un sistema DIEM RAG.\n"
             "Decidi quali collezioni Chroma devono essere interrogate per la domanda data.\n"
             "Restituisci un oggetto JSON che mappa ogni nome di collezione a true o false.\n"
             "Restituisci solo JSON, nessun altro testo.\n\n"
             "Collezioni e descrizioni:\n"
             "{collection_descriptions}\n\n"
             "Nomi delle collezioni: {collection_names}\n"
             "Regole:\n"
             "- Usa solo i nomi di collezione forniti.\n"
             "- Includi sempre ogni nome di collezione nell'output JSON.\n"
             "- L'output deve essere un JSON valido con valori booleani."
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

        router_chain = (
            RunnablePassthrough.assign(
                collection_names=lambda _: ", ".join(self.retrievers.keys()),
                collection_list=lambda _: list(self.retrievers.keys()),
                collection_descriptions=lambda _: ROUTER_COLLECTION_DESCRIPTIONS,
            )
            | RunnablePassthrough.assign(
                routing_raw=(
                    router_prompt
                    | self.llm
                    | StrOutputParser()
                    | RunnableLambda(log_router)
                )
            )
            | RunnableLambda(lambda x: parse_router_output(x["routing_raw"], x["collection_list"]))
        )

        # ANSWER GENERATOR 
        # Synthesizes the final output while strongly conditioning the LLM to avoid hallucinations.
        answer_prompt = ChatPromptTemplate.from_messages([
            ( "system",
            "Sei l'Assistente Virtuale Ufficiale del dipartimento DIEM dell'Università di Salerno.\n"
            "Fornisci risposte chiare, professionali ed esaustive alle domande degli utenti.\n\n"
            "REGOLE DI COMPORTAMENTO RIGIDE:\n"
            "1. IMMEDESIMAZIONE TOTALE: Tratta le informazioni a tua disposizione come tua conoscenza diretta. È SEVERAMENTE VIETATO usare espressioni come 'in base ai documenti', 'secondo il testo', o 'i pdf dicono'. Rispondi in modo naturale e autorevole.\n"
            "2. FOCUS ASSOLUTO (ANTI-VERBOSITÀ): Rispondi ESCLUSIVAMENTE alla domanda posta nell'ultimo turno. È severamente vietato aggiungere paragrafi di approfondimento non richiesti, divagare o riprendere argomenti passati.\n"
            "3. UTILIZZO DELLA CRONOLOGIA: Leggi la cronologia della conversazione SOLO per comprendere il contesto dei pronomi. Non farti influenzare dagli argomenti passati per generare la risposta attuale.\n"
            "4. COMPLETEZZA: Fornisci risposte discorsive. Devi essere completo ma coinciso, riportando ove strettamente necessario elenchi, procedure, orari o scadenze utilizzando formattazioni chiare, ma sintetizzando il contenuto se troppo esteso, in maniera funzionale.\n"
            "5. ADERENZA AI FATTI: Non inventare o dedurre nulla.\n"
            "6. CONSAPEVOLEZZA TEMPORALE: Oggi è il {today}. Adeguati ai tempi verbali per eventi passati o futuri."
            ),
            ( "placeholder", "{chat_history}" ),
            ( "human",
            "INFORMAZIONI A TUA DISPOSIZIONE:\n---\n"
            "CONTESTO:\n{context}\n\n"
            "DOMANDA UTENTE:\n{question}\n\n"
            "RISPOSTA:"
            )
        ])

        rag_chain = (
            RunnablePassthrough.assign(
                retrieved_data=lambda x: self._retrieve_and_format(
                    x["optimized_question"],
                    selected_collections=x.get("selected_collections")
                )
            )
            | RunnableLambda(
                lambda x: {
                    "context": x["retrieved_data"]["context"],
                    "question": x["question"],
                    "sources": x["retrieved_data"]["docs"],
                    "chat_history": x["chat_history"],
                    "today": datetime.now().strftime("%d/%m/%Y")
                }
            )
            | RunnablePassthrough.assign(answer=(answer_prompt | self.llm | StrOutputParser()))
        )

        # Logical subchain to intercept non-document-related intents 
        out_of_domain_chain = RunnableLambda(
            lambda _: {
                "answer": "Mi dispiace, ma come assistente del DIEM posso rispondere solo a domande relative al dipartimento ed ai suoi corsi di studio.",
                "sources": [],
            }
        )

        greeting_chain = RunnableLambda(
            lambda _: {
                "answer": "Ciao! Sono l'Assistente Virtuale del Dipartimento DIEM dell'Università di Salerno. Posso aiutarti a trovare informazioni su corsi, docenti, laboratori o regolamenti didattici. Come posso esserti utile?",
                "sources": [],
            }
        )

        # Main full chain
        main_chain = (
            RunnablePassthrough.assign(domain_status=domain_guardrail_chain)
            | RunnableBranch(
                (lambda x: "FUORI_DOMINIO" in x["domain_status"].upper(), out_of_domain_chain),
                (lambda x: "CONVENEVOLI" in x["domain_status"].upper(), greeting_chain),
                (
                    RunnablePassthrough.assign(optimized_question=rewrite_chain)
                    | RunnablePassthrough.assign(routing_map=router_chain)
                    | RunnablePassthrough.assign(
                        selected_collections=lambda x: select_collections(
                            x["routing_map"], list(self.retrievers.keys())
                        )
                    )
                    | rag_chain
                )
            )
        )
        
        # Abstraction for conversational session state persistence
        def get_session_history(session_id: str):
            #HARDCODATO PER IL TEST, POI RIMETTI NORMALE
            # if session_id not in self.chat_histories:
            #     self.chat_histories[session_id] = InMemoryChatMessageHistory()
            # return self.chat_histories[session_id]
            return InMemoryChatMessageHistory()

        # ari_servers: the pieces of the chain used by ../rag.py, which runs the retrieval of main_chain step by
        # step and leaves the generation (answer_prompt | llm) and the history to the LLM server
        self._domain_guardrail_chain = domain_guardrail_chain
        self._rewrite_chain = rewrite_chain
        self._router_chain = router_chain
        self._select_collections = select_collections
        self._answer_prompt = answer_prompt
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
        if not user_input or not user_input.strip():
            raise ValueError("[ERROR] The input logical payload cannot be empty.")
            
        logger.info(f"[INFO] RAG pipeline invoked for Session -{session_id}-")
        
        # Execute the masked graph via LangChain abstraction
        result = self.conversational_pipeline.invoke(
            {"question": user_input},
            config={"configurable": {"session_id": session_id}}
        )
        
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
            
        return answer, formatted_sources


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