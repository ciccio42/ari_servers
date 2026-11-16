# Modelli della demo sulla DGX Spark: Whisper e LLM

Test del 2026-11-12, con le frasi del piano di test.

- **Whisper** (trascrizione dei 5 file audio di questa cartella): il modello in uso nella demo, **large-v3-turbo**, è
  il miglior compromesso tra velocità e accuratezza anche rispetto agli altri modelli Whisper. Small e i modelli più
  piccoli sono più veloci, ma sbagliano troppe parole in italiano; medium e large-v3 sono più lenti e non più precisi.
- **LLM** (risposte del bot sociale, [ultima parte](#llm-risposte-del-bot-sociale-chatgpt-con-e-senza-rag)): gpt-4o
  risponde in 1,3 s senza RAG e in 5,4 s con il RAG sui documenti medici; il RAG cambia le risposte alle domande
  mediche, che vengono dalle fonti caricate. Rasa aggiunge circa 40 ms, 10–25 ms per le risposte senza LLM.

## Risultati

| Modello | Parametri | Inferenza media | Varianza (s²) | Dev. std | p95 | RTF | WER | CER | Frasi esatte | Memoria GPU | Su disco (ONNX) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| tiny | 39 M | 0,053 s | 0,00027 | 0,017 s | 0,074 s | 0,007 | 50,0% | 20,2% | 0/5 | ~1,0 GB | 0,2 GB |
| base | 74 M | 0,074 s | 0,00043 | 0,021 s | 0,103 s | 0,010 | 34,1% | 12,1% | 1/5 | ~1,2 GB | 0,3 GB |
| small | 244 M | 0,129 s | 0,00121 | 0,035 s | 0,182 s | 0,017 | 15,9% | 5,2% | 2/5 | ~2,1 GB | 0,9 GB |
| medium | 769 M | 0,262 s | 0,00454 | 0,067 s | 0,356 s | 0,035 | **6,8%** | 3,6% | 3/5 | ~4,8 GB | 2,4 GB |
| **large-v3-turbo** | 809 M | **0,185 s** | 0,00065 | 0,025 s | 0,220 s | 0,025 | 9,1% | **2,8%** | 3/5 | ~3,6 GB | 2,1 GB |
| large-v3 | 1 550 M | 0,416 s | 0,01061 | 0,103 s | 0,561 s | 0,056 | 11,4% | 3,6% | 3/5 | ~8,1 GB | 4,7 GB |

- **Inferenza** (`asr_s`): estrazione delle feature, generazione e decodifica del testo, misurata dal server. Le
  statistiche sono sulle 100 chiamate misurate; la varianza dipende soprattutto dalla diversa lunghezza dei file.
- **RTF** (real time factor): tempo di inferenza diviso la durata dell'audio. 0,025 vuol dire che 1 s di parlato
  viene trascritto in 25 ms.
- **WER / CER**: errori (sostituzioni, cancellazioni, inserzioni) su parole e su caratteri, rispetto alle frasi di
  riferimento. Testo normalizzato: minuscolo, senza punteggiatura.
- **Frasi esatte**: trascrizioni uguali al riferimento dopo la normalizzazione.
- **Memoria GPU**: quella del processo del server, misurata da `nvidia-smi` dopo il caricamento.

Gli altri tempi sono quasi uguali per tutti i modelli: il VAD (Silero, su CPU) prende 0,017 s in media, il
trasferimento HTTP e il resto del server circa 0,001–0,002 s. Il tempo visto dal client è quindi l'inferenza più
circa 0,02 s (es. turbo: 0,203 s).

## Tempo di inferenza per file

Media sulle 20 ripetizioni; la deviazione standard sullo stesso file è al massimo 6,5 ms per tutti i modelli.

| File | Durata | tiny | base | small | medium | large-v3-turbo | large-v3 |
|---|---|---|---|---|---|---|---|
| speech_20261105_111304_743347.wav | 6,20 s | 0,033 s | 0,049 s | 0,090 s | 0,177 s | 0,151 s | 0,282 s |
| speech_20261105_111327_845648.wav | 6,78 s | 0,050 s | 0,070 s | 0,113 s | 0,243 s | 0,182 s | 0,402 s |
| speech_20261105_111426_146906.wav | 8,76 s | 0,071 s | 0,094 s | 0,180 s | 0,355 s | 0,218 s | 0,556 s |
| speech_20261105_111446_849471.wav | 6,14 s | 0,038 s | 0,056 s | 0,102 s | 0,211 s | 0,165 s | 0,335 s |
| speech_20261105_111545_246613.wav | 9,08 s | 0,073 s | 0,101 s | 0,158 s | 0,322 s | 0,206 s | 0,505 s |

Il tempo cresce con il numero di parole da generare più che con la durata dell'audio: il decoder genera una parola
alla volta. Per questo turbo, che ha 4 strati di decoder (medium 24, large-v3 32), è più veloce di medium pur
essendo più grande.

## Trascrizioni

Ogni modello ha dato la stessa trascrizione in tutte e 20 le ripetizioni di ogni file.

| Riferimento | tiny | base | small | medium | large-v3-turbo | large-v3 |
|---|---|---|---|---|---|---|
| qual è la ricetta della pizza? | Qual la ricetta della pizza? | Questa è la ricetta della pizza | Qua è la ricetta della pizza? | qual è la ricetta della pizza? | qual è la ricetta della pizza? | Qual è la ricetta della pizza? |
| cosa sai dirmi dell'università di salerno? | cosa salvi il midedolo università di salerno? | cosa sa di irmi dell'università di salerno? | cosa sai dirmi dell'università di Salerno? | Cosa sai dirmi dell'università di Salerno? | Cosa sai dirmi dell'Università di Salerno? | Cosa sai dirmi dell'Università di Salerno? |
| quali sono le soglie di emoglobina glicata che indicano un rischio di diabete? | Quali sono le soia di mogro binaglicata che indigano risco di diabeta? | quali sono le sue demoni? Bina cliccata che indica non rischio di diabetes | Quali sono le solle di monoglobina cliccata che indicano un rischio di diabete? | Quali sono le soglie di manucrobina glicata che indicano il rischio di diabete? | Quali sono le soglie di monoglobina glicata che indicano il rischio di diabete? | Quali sono le soglie di microcrobina glicata che indicano un rischio di diabete? |
| mi descrivi l'ambiente circostante? | mi descrivili a mente circostante | mi descrivi l'ambiente circostante | Mi descrivi l'ambiente circostante? | mi descrivi l'ambiente circostante? | Mi descrivi l'ambiente circostante? | Mi descrivi l'ambiente circostante? |
| chi può prescrivere gli agonisti del GLP-1 secondo la nota AIFA 100? | chi bo prescrivere gli agonisti del giù e le piu no secondo la nota ai facendo | Chi può prescrivere gli agonisti del GVL P1 secondo la nota Aifacendo? | Chi può prescrivere gli agonisti del GVLP1 secondo la nota I-100? | Chi può prescrivere gli agonisti del GLP-1 secondo la nota IFA 100? | Chi può prescrivere gli agonisti del GVLP1 secondo la nota AIFA 100? | Chi può prescrivere gli agonisti del GVLP1 secondo la nota AIFA100? |

## Conclusioni

1. **large-v3-turbo resta la scelta per la demo.** Ha il CER più basso, un WER vicino al migliore e, tra i modelli
   accurati, è il più veloce (0,185 s, RTF 0,025) e il più stabile (p95 0,220 s).
2. **medium** ha il WER più basso ed è l'unico che riconosce "GLP-1", ma costa il 40% di tempo in più di turbo e più
   memoria. La differenza di accuratezza è di una o due parole: non basta per cambiare modello.
3. **large-v3** non migliora su nessuna metrica e costa 2,3 volte il tempo di turbo e più del doppio della memoria.
4. **small** è un'alternativa solo se il tempo conta più delle parole: è 1,4 volte più veloce di turbo, ma sbaglia i
   termini medici ("solle di monoglobina cliccata", "nota I-100").
5. **tiny e base** non sono utilizzabili in italiano.
6. I termini medici sono il punto debole di tutti: "emoglobina" non la trascrive nessun modello. Se nella demo
   contano, conviene provare un prompt iniziale con il glossario, prima di passare a un modello più grande.

## Limiti del test

- **Campione piccolo.** 5 frasi, 66 parole: una parola sbagliata vale circa 1,5 punti di WER. Le 20 ripetizioni
  danno trascrizioni identiche, quindi per l'accuratezza contano come una sola. Tra medium, turbo e large-v3 le
  differenze non sono significative; per un confronto solido servono più frasi e più parlanti.
- **Frasi di riferimento.** Le prime quattro vengono dal piano di test; la quinta, che nel piano non c'è, è stata
  validata il 2026-11-12. Resta da controllare ascoltando l'audio "un rischio" o "il rischio" nella terza (medium e
  turbo scrivono "il"). Le frasi sono in [references.csv](references.csv): dopo una correzione basta rilanciare
  `wer.py`.
- **WER sensibile alla divisione delle parole.** "AIFA100" invece di "AIFA 100" conta 2 errori, perciò large-v3 ha
  WER 30,8% sull'ultimo file ma CER 4,5%. Per questo confronto il CER è più affidabile.
- **Solo la Spark.** Sulle Jetson (Orin 64 GB, Orin 32 GB di ARI) i tempi e la memoria disponibile cambiano.

## Configurazione

| | |
|---|---|
| Macchina | DGX Spark, GPU NVIDIA GB10, driver 580.159.03 (CUDA 13.0) |
| Server | `whisper_server` di ari_servers, immagine `ari_whisper_server:spark`, porta 8001 |
| Runtime | ONNX Runtime 1.23.2, `CUDAExecutionProvider`; torch 2.11.0, transformers 4.53.2, optimum 2.1.0 |
| Modelli | `openai/whisper-<modello>` di Hugging Face, esportati in ONNX con ottimizzazione O4 (fp16, solo GPU) |
| Lingua | italiano, task `transcribe`; VAD Silero con soglia 0,7 |
| Protocollo | i 5 WAV alternati a ogni giro, 3 giri di warmup e 20 misurati: 100 chiamate per modello. Client `bench_http.py` sulla stessa macchina, audio float32 a 16 kHz via HTTP |

| File | Durata |
|---|---|
| speech_20261105_111304_743347.wav | 6,200 s |
| speech_20261105_111327_845648.wav | 6,776 s |
| speech_20261105_111426_146906.wav | 8,760 s |
| speech_20261105_111446_849471.wav | 6,136 s |
| speech_20261105_111545_246613.wav | 9,080 s |
| **media** | **7,390 s** (varianza 2,025 s², dev. std 1,423 s) |

Al primo avvio ogni modello viene scaricato ed esportato in `models/whisper/`: il server è pronto dopo circa 40 s
(tiny, base), 70 s (small), 160 s (medium) e 240 s (large-v3). Turbo era già esportato e ha impiegato 131 s.

## Come ripetere il test

Da `ari_servers`, con il server avviato su un modello (es. `--model openai/whisper-small --port 8001`):
```bash
python3 whisper/tools/bench_http.py --url http://localhost:8001 --audio-dir audio_files --runs 20 --warmup 3 \
  --output latency_logs/runs_small.csv
python3 whisper/tools/summarize_whisper.py audio_files latency_logs/runs_small.csv     # tempi e trascrizioni
python3 whisper/tools/wer.py latency_logs/runs_small.csv --per-file                   # WER e CER
```
`bench_http.py` richiede numpy e soundfile (presenti nell'immagine). I risultati di questo test sono in
`latency_logs/models_2026-11-12/` (non versionata): `runs_<modello>.csv`, `summary_<modello>.txt`, `wer.csv`.

## LLM: risposte del bot sociale (ChatGPT, con e senza RAG)

Le domande del piano mandate al bot sociale di Rasa come le manda il robot: client → proxy Rasa (porta 5002) →
`action_social` → server LLM (porta 8010) → OpenAI. Un solo utente per giro, che pone le domande in sequenza: l'ultima
("e quali sono gli effetti collaterali?") verifica che il server tenga lo storico della conversazione.

### Tempi

| | RAG spento | RAG acceso |
|---|---|---|
| API di OpenAI | Chat Completions | Responses API con file_search sui documenti |
| Risposta vista dal client, media | **1,30 s** | **5,41 s** |
| Dev. std (varianza) | 0,36 s (0,126 s²) | 0,87 s (0,759 s²) |
| Minimo – massimo | 0,97 – 1,90 s | 4,35 – 6,33 s |

| Domanda | RAG spento | RAG acceso |
|---|---|---|
| qual è la ricetta della pizza? | 1,90 s | 5,73 s |
| cosa sai dirmi dell'università di salerno? | 0,97 s | 6,02 s |
| quali sono le soglie di emoglobina glicata che indicano un rischio di diabete? | 1,15 s | 4,35 s |
| chi può prescrivere gli agonisti del GLP-1 secondo la nota AIFA 100? | 1,27 s | 4,64 s |
| e quali sono gli effetti collaterali? | 1,19 s | 6,33 s |

Quasi tutto il tempo è la risposta di OpenAI (tabella seguente). Il lock dei server non ha mai fatto attendere
(`queue_s` 0), perché c'era un solo client.

### Tempi del server RASA

Dal latency log: il proxy registra il tempo di ogni risposta di Rasa (`rasa_s`, action compresa), il server LLM il
tempo della sua risposta (`total_s`); la differenza è il lavoro di Rasa (riconoscimento dell'intent, policy,
chiamata all'action server, client HTTP dell'action).

| Richieste | n | Risposta di Rasa (`rasa_s`) | di cui server LLM | **di cui Rasa** |
|---|---|---|---|---|
| bot sociale con il LLM, RAG spento | 5 | 1,295 s (dev. std 0,356) | 1,253 s | **42 ms** (dev. std 4; 36–46) |
| bot sociale con il LLM, RAG acceso | 5 | 5,414 s (dev. std 0,868) | 5,375 s | **38 ms** (dev. std 6; 29–44) |
| bot sociale senza LLM (età, emozione, avvio dei giochi...) | 13 | 25 ms (dev. std 8; 13–43) | – | 25 ms |
| gioco sasso carta forbice (porta 5003) | 7 | 10 ms (dev. std 1; 9–12) | – | 10 ms |
| gioco forza quattro (porta 5004) | 6 | 13 ms (dev. std 3; 10–16) | – | 13 ms |

Rasa gira su CPU e pesa al massimo 40 ms per risposta, il 3% della risposta del bot sociale senza RAG e meno
dell'1% con il RAG. Le richieste senza LLM sono quelle dei test del 2026-11-12 sullo stesso server; il proxy aggiunge
meno di 1 ms.

### Risposte

| Domanda | RAG spento | RAG acceso |
|---|---|---|
| qual è la ricetta della pizza? | Per fare la pizza, mescola 500 g di farina, 325 ml di acqua, 10 g di sale, 3 g di lievito e 1 cucchiaio di olio. Impasta, lascia lievitare per 2 ore, stendi, aggiungi pomodoro e mozzarella, e cuoci a 250°C per 10 minuti. | Per preparare la pizza, mescola farina, acqua, lievito e sale per ottenere l'impasto, lascialo lievitare, poi stendilo e aggiungi il condimento. Cuoci in forno molto caldo finché la pizza è dorata. |
| cosa sai dirmi dell'università di salerno? | L'Università di Salerno è una delle più antiche d'Europa, con origini che risalgono alla Scuola Medica Salernitana. Oggi offre una vasta gamma di corsi di laurea e ricerca in vari campi, ed è situata nel campus di Fisciano. | L'Università di Salerno è un'istituzione accademica italiana conosciuta per la qualità della sua ricerca e l'offerta formativa. È situata nella regione Campania e offre una vasta gamma di corsi di laurea e post-laurea in diverse discipline. |
| quali sono le soglie di emoglobina glicata che indicano un rischio di diabete? | Un valore di emoglobina glicata tra 5,7% e 6,4% indica prediabete, mentre un valore pari o superiore a 6,5% suggerisce diabete. | I valori di emoglobina glicata tra 6,0 e 6,49% indicano un rischio elevato di sviluppare il diabete. |
| chi può prescrivere gli agonisti del GLP-1 secondo la nota AIFA 100? | Secondo la nota AIFA 100, gli agonisti del GLP-1 possono essere prescritti da medici specialisti in endocrinologia, diabetologia e malattie del metabolismo. | Gli agonisti del GLP-1 possono essere prescritti dai medici di medicina generale o dagli specialisti autorizzati dalle Regioni, tramite il sistema Tessera Sanitaria. |
| e quali sono gli effetti collaterali? | Gli effetti collaterali degli agonisti del GLP-1 includono nausea, vomito, diarrea e perdita di appetito. Possono anche causare ipoglicemia, soprattutto se usati con altri farmaci ipoglicemizzanti. | Gli agonisti del GLP-1 possono causare nausea, vomito, e in alcuni casi pancreatite. Gli effetti sono generalmente lievi e transitori. |

### Conclusioni

1. **Il percorso funziona**: Rasa → server LLM → OpenAI risponde a tutte le domande, in italiano e entro la lunghezza
   chiesta dal prompt.
2. **Lo storico funziona**: la domanda di seguito ("e quali sono gli effetti collaterali?") è capita in entrambi i
   giri come riferita agli agonisti del GLP-1.
3. **Il RAG cambia le risposte mediche.** Con i documenti le soglie dell'emoglobina glicata (6,0–6,49%) e la nota
   AIFA 100 (anche i medici di medicina generale) vengono dalle fonti caricate; senza, gpt-4o risponde con la sua
   conoscenza generale (per le soglie quelle internazionali, 5,7–6,4%). Quale risposta sia corretta per la demo va
   valutato da chi conosce le fonti.
4. **Il RAG costa circa 4 s a risposta** e si applica a tutte le domande dell'action, anche a quelle non mediche
   (pizza, università), che diventano più lente e più generiche.
5. **Rasa non è un collo di bottiglia**: 10–25 ms per le risposte senza LLM, circa 40 ms in più quando chiama il
   server LLM. Il tempo del bot sociale è quello di OpenAI (1,3 s senza RAG, 5,4 s con il RAG), contro 0,2 s di
   Whisper.

### Limiti del test

- Una sola esecuzione per domanda: i tempi di OpenAI variano con il carico del servizio, e le risposte cambiano un
  po' da una chiamata all'altra (temperatura 0,3 senza RAG, default di OpenAI con il RAG).
- Non provate le domande sulla scena (`action_gpt_4o`): servono l'immagine della telecamera nel tracker di Rasa, che
  arriva dal robot.
- Accuratezza delle risposte mediche non valutata.

### Configurazione

| | |
|---|---|
| Server LLM | `llm_server` di ari_servers, immagine `ari_llm_server`, backend `openai` (`ChatGPTBackend`), porta 8010 |
| Modello | `gpt-4o` (servito da OpenAI come `gpt-4o-2024-08-06`), temperatura 0,3 senza RAG |
| RAG | `llm.rag: true` solo per il secondo giro (`LLM_RAG=true`): file_search sul vector store `vs_6adf80ee4cf481919b35fecdd7d1efd9` (diabetologia e cardiologia), 5 passaggi per ricerca, ricerca sempre (`tool_choice: required`) |
| Prompt | "Medical Prompt" di `rasa/bots/social/actions/phrases_rasa.yml`, risposte entro 250 caratteri |
| Storico | 4 messaggi per conversazione, la domanda compresa |
| Rasa | `rasa_server` (immagine `ari_rasa_server`), bot sociale in italiano, proxy senza lock (`rasa.exclusive: false`) |

### Come ripetere il test

Da `ari_servers`, con `OPENAI_API_KEY` in `config/secrets.env`:
```bash
bash llm/docker/run.bash start                    # RAG come da config/servers.yaml (llm.rag)
bash rasa/docker/run.bash start
bash rasa/docker/run.bash test "quali sono le soglie di emoglobina glicata che indicano un rischio di diabete?"
bash llm/docker/run.bash stop && LLM_RAG=true bash llm/docker/run.bash start   # un giro con il RAG
```
I tempi di ogni risposta sono anche nel latency log (`latency_logs/latency_<data>_<host>.csv`, righe
`rasa_proxy.done` per Rasa, `llm_server.done` e `openai.reply` per il LLM).
