# Prompt Engineering Fondamentale
### Corso RAG per applicazioni ERP e CRM
#### Incontro 2 - [20/05/2025]

---

## Agenda dell'incontro
- Benvenuto e riepilogo incontro precedente
- Principi del Prompt Engineering
- Componenti di un prompt efficace
- Tecniche di prompting: zero-shot, one-shot, few-shot
- Strategie avanzate di prompting
- **Pausa (15 min)**
- Problemi comuni e soluzioni
- Implementazione di prompt con contesto
- Esercitazioni pratiche
- Caso d'uso: prompt per ERP/CRM
- Conclusione e compiti

---

## Obiettivi dell'incontro
- Comprendere i principi fondamentali del prompt engineering
- Identificare e applicare i componenti di un prompt efficace
- Differenziare tra approcci zero-shot, one-shot e few-shot
- Riconoscere e risolvere problemi comuni nei prompt
- Implementare prompt strutturati con contesto
- Sviluppare prompt specifici per ambiti ERP/CRM

---

## Riepilogo Incontro Precedente

### Concetti chiave affrontati
- Fondamenti dell'AI Generativa
- Evoluzione dei modelli linguistici
- Architettura Transformer
- Principali LLM sul mercato
- Setup dell'ambiente di sviluppo
- Prime API calls a OpenAI e Claude

---

## Principi del Prompt Engineering

### Cos'è il Prompt Engineering?
- **Definizione**: arte e scienza di comunicare efficacemente con i LLM
- **Obiettivo**: ottenere output precisi, pertinenti e utili
- **Importanza**: determina fino al 90% della qualità dell'output
- **Evoluzione**: da semplici query a istruzioni strutturate

---

## Principi del Prompt Engineering

### Principi fondamentali
- **Chiarezza**: istruzioni precise e non ambigue
- **Specificità**: dettagli sufficienti sul risultato desiderato
- **Contesto**: informazioni rilevanti per inquadrare il task
- **Struttura**: organizzazione logica delle richieste
- **Iterazione**: miglioramento progressivo dei prompt

---

## Principi del Prompt Engineering

### Impatto del prompt engineering
- Stessa domanda, prompt diversi → risultati radicalmente diversi
- Riduzione di allucinazioni e risposte fuori tema
- Ottimizzazione di tempo e costi (meno token necessari)
- Personalizzazione dell'output per casi d'uso specifici

---

## Componenti di un Prompt Efficace

### Elementi strutturali
- **Istruzioni**: cosa deve fare il modello
- **Contesto**: informazioni di background
- **Esempi**: dimostrazioni del risultato atteso
- **Input specifico**: dati su cui lavorare
- **Output format**: formato desiderato della risposta

---

## Componenti di un Prompt Efficace

### Ruoli espliciti (sistema)
```
Sei un consulente ERP esperto specializzato in SAP.
Il tuo compito è analizzare problemi di implementazione
e suggerire soluzioni pratiche.
```

### Query (utente)
```
Stiamo riscontrando errori di sincronizzazione tra il
modulo HR e il modulo Finanziario. Come possiamo
diagnosticare e risolvere questo problema?
```

---

## Componenti di un Prompt Efficace

### Specificazione del formato
```
Fornisci la risposta nel seguente formato:
1. Diagnosi del problema (max 100 parole)
2. Possibili cause (elenco puntato)
3. Procedura di risoluzione passo-passo
4. Prevenzione futura (2-3 raccomandazioni)
```

### Vincoli e limitazioni
```
Limita la risposta a consigli implementabili 
senza personalizzazioni del codice. Non includere
soluzioni che richiedono interventi del supporto SAP.
```

---

## Zero-shot, One-shot e Few-shot Learning

### Zero-shot learning
- **Definizione**: il modello esegue un task senza esempi
- **Quando usarlo**: per compiti semplici o standard
- **Vantaggi**: risparmio token, prompt più snelli
- **Svantaggi**: risultati meno prevedibili, più variabili

---

## Zero-shot, One-shot e Few-shot Learning

### Esempio Zero-shot
```
Classifica il seguente feedback di un cliente in 
positivo, neutro o negativo:

"Il sistema CRM ha migliorato la nostra gestione 
dei lead, ma l'interfaccia è ancora complessa 
e alcuni report non funzionano correttamente."
```

---

## Zero-shot, One-shot e Few-shot Learning

### One-shot learning
- **Definizione**: fornisce un singolo esempio del task
- **Quando usarlo**: per chiarire il formato o livello di dettaglio
- **Vantaggi**: chiarisce aspettative minimizzando token
- **Svantaggi**: limita la comprensione di pattern complessi

---

## Zero-shot, One-shot e Few-shot Learning

### Esempio One-shot
```
Classifica il seguente feedback di un cliente in positivo, 
neutro o negativo.

Esempio:
Feedback: "Il vostro ERP ha risolto molti problemi, ma la 
reportistica è ancora lenta."
Classificazione: Neutro (aspetti positivi e negativi bilanciati)

Ora classifica:
"Il sistema CRM ha migliorato la nostra gestione dei lead, 
ma l'interfaccia è ancora complessa e alcuni report non 
funzionano correttamente."
```

---

## Zero-shot, One-shot e Few-shot Learning

### Few-shot learning
- **Definizione**: fornisce multipli esempi del task
- **Quando usarlo**: per task complessi o con pattern specifici
- **Vantaggi**: calibra meglio il modello, risultati più prevedibili
- **Svantaggi**: consuma più token, prompt più lunghi

---

## Zero-shot, One-shot e Few-shot Learning

### Esempio Few-shot
```
Classifica il feedback clienti come:
P (positivo), N (neutro), NEG (negativo)

Esempi:
"Implementazione perfetta, tempi rispettati" → P
"Sistema funzionale ma setup laborioso" → N
"Troppe funzioni non operative, supporto lento" → NEG
"Buona integrazione ma prezzo elevato" → N

Ora classifica:
"Il sistema CRM ha migliorato la nostra gestione dei lead, 
ma l'interfaccia è ancora complessa e alcuni report non 
funzionano correttamente."
```

---

## Strategie Avanzate di Prompting

### Chain-of-Thought (CoT)
- Incoraggia il modello a mostrare il ragionamento
- "Pensa passo-passo" migliora accuratezza su problemi complessi
- Utile per calcoli, logica, troubleshooting

```
Analizza il seguente problema di performance ERP. 
Ragiona passo-passo per identificare la causa principale 
e suggerire soluzioni.
```

---

## Strategie Avanzate di Prompting

### Esempi Chain-of-Thought
```
Problema: Il report mensile di vendite impiega 30 minuti per essere generato, 
mentre prima richiedeva solo 5 minuti.

Passo 1: Consideriamo quando è iniziato il problema...
Passo 2: Valutiamo cosa è cambiato nel sistema...
Passo 3: Analizziamo il volume di dati coinvolti...
...
```

---

## Strategie Avanzate di Prompting

### Prompt Decomposition
- Scomporre problemi complessi in sotto-problemi
- Affrontare sequenzialmente parti della soluzione
- Costruire prompt che si sviluppano in più passaggi

```
Prima analizzeremo i requisiti del cliente.
Poi identificheremo i moduli CRM necessari.
Infine, prepareremo un piano di implementazione.
```

---

## Strategie Avanzate di Prompting

### ReAct (Reasoning + Acting)
- Combinazione di ragionamento e azioni concrete
- Il modello alterna riflessione e decisioni operative
- Formato: Thought → Action → Observation → Thought

```
Thought: Devo analizzare questo problema di integrazione CRM-ERP.
Action: Identificare i punti di connessione tra i sistemi.
Observation: I dati cliente non si sincronizzano correttamente.
Thought: Potrebbe essere un problema di mapping dei campi...
```

---

## PAUSA (15 minuti)

---

## Problemi Comuni e Soluzioni

### Allucinazioni
- **Problema**: il modello inventa informazioni non verificabili
- **Soluzioni**:
  - Richiedere fonti e riferimenti
  - Limitare risposta a fatti certi
  - Istruire il modello a dichiarare incertezze

```
Basati solo sui fatti forniti. Se non hai informazioni 
sufficienti, indica chiaramente cosa non puoi determinare.
```

---

## Problemi Comuni e Soluzioni

### Risposte vaghe o generiche
- **Problema**: output troppo generico, poco utile
- **Soluzioni**:
  - Richiedere dettagli specifici
  - Specificare livello di profondità
  - Fornire metriche o KPI

```
Fornisci almeno 3 strategie concrete, con esempi 
specifici di implementazione in un contesto ERP SAP.
```

---

## Problemi Comuni e Soluzioni

### Bias e preferenze
- **Problema**: risposte influenzate da bias impliciti
- **Soluzioni**:
  - Richiedere prospettive bilanciate
  - Specificare criteri oggettivi
  - Esplicitare punti di vista da considerare

```
Presenta vantaggi e svantaggi di ciascuna soluzione CRM,
considerando aziende di diverse dimensioni e settori.
```

---

## Problemi Comuni e Soluzioni

### Distrazioni dal prompt originale
- **Problema**: il modello ignora parti dell'istruzione
- **Soluzioni**:
  - Strutturare visivamente il prompt
  - Numerare richieste e parti
  - Enfatizzare punti critici

```
[IMPORTANTE] Assicurati di completare TUTTE le 4 sezioni
richieste, prestando particolare attenzione al punto 3.
```

---

## Implementazione di Prompt con Contesto

### Aggiunta di contesto esplicito
- Scenario aziendale
- Dati rilevanti
- Vincoli e limitazioni
- Obiettivi specifici

```
Contesto: Azienda manifatturiera con 500 dipendenti,
5 stabilimenti in Europa, attualmente utilizza un ERP
legacy sviluppato internamente, con dati distribuiti
in silos. Budget per nuovo sistema: 1M€.
```

---

## Implementazione di Prompt con Contesto

### Prompt template per analisi ERP
```
RUOLO: Consulente implementazione ERP
CONTESTO: {descrizione_azienda}
SITUAZIONE ATTUALE: {sistemi_esistenti}
OBIETTIVI: {scopi_implementazione}
VINCOLI: {limitazioni_tecniche_budget}
RICHIESTA: Fornisci una roadmap di implementazione
con le seguenti sezioni:
1. Analisi situazione attuale
2. Selezione moduli prioritari
3. Timeline implementazione
4. Gestione rischi
5. KPI per misurare successo
```

---

## Esercitazioni Pratiche

### Esercizio 1: Miglioramento iterativo
- Partire da un prompt base
- Identificare punti deboli nella risposta
- Migliorare il prompt progressivamente
- Analizzare differenze nei risultati

### Esercizio 2: Confronto approcci
- Stesso problema con zero-shot vs few-shot
- Confrontare qualità, precisione e utilità
- Documentare pro e contro di ciascun approccio

---

## Caso d'uso: Prompt per ERP/CRM

### Estrazione di insight da dati cliente
```
Analizza i seguenti dati di interazione cliente estratti
dal CRM e identifica:
1. Pattern comportamentali ricorrenti
2. Opportunità di upselling/cross-selling
3. Rischio di churn (abbandono)
4. Suggerimenti per migliorare engagement

Dati cliente: {dati_json}
```

---

## Caso d'uso: Prompt per ERP/CRM

### Troubleshooting problemi di implementazione
```
Sei un esperto di implementazione [SISTEMA ERP/CRM].
Un cliente riscontra il seguente errore durante 
la configurazione del modulo [NOME MODULO]:

"{messaggio_errore}"

Fornisci:
1. Diagnosi probabile
2. Procedura step-by-step per verificare la causa
3. Soluzioni per i 3 scenari più probabili
4. Prevenzione futura
```

---

## Caso d'uso: Prompt per ERP/CRM

### Generazione di documentazione utente
```
Crea una guida utente per la funzionalità [NOME FUNZIONE]
del nostro sistema [ERP/CRM]. La documentazione sarà 
utilizzata da utenti con competenze [LIVELLO] e dovrebbe:

1. Spiegare lo scopo della funzionalità
2. Illustrare casi d'uso comuni
3. Fornire istruzioni passo-passo con screenshot
4. Includere FAQ e troubleshooting base
5. Essere scritta in linguaggio [formale/informale]
```

---

## Conclusione

### Riepilogo concetti chiave
- Il prompt è l'interfaccia umano-LLM
- Struttura, chiarezza e specificità sono fondamentali
- Differenti approcci per differenti scenari
- L'iterazione è la chiave del successo
- Contesto e dettagli migliorano la precisione

---

## Conclusione

### Compiti per il prossimo incontro
- Sviluppare 3 prompt per casi d'uso ERP/CRM specifici
- Testare approcci zero-shot vs few-shot sui propri esempi
- Documentare le iterazioni di miglioramento dei prompt
- Leggere materiale su chain-of-thought prompting

---

## Risorse aggiuntive
- [OpenAI Prompt Engineering Guide](https://platform.openai.com/docs/guides/prompt-engineering)
- [Anthropic Claude Prompt Engineering](https://docs.anthropic.com/claude/docs/introduction-to-prompting)
- "Prompt Engineering for Developers" (corso online)
- [LangChain Documentation - Prompting](https://docs.langchain.com/docs/components/prompts)
- [PromptingGuide.ai](https://www.promptingguide.ai/)

---

## Domande?

---

## Grazie dell'attenzione!
### Prossimo incontro: Retrieval Augmented Generation (RAG) - Fondamenti
