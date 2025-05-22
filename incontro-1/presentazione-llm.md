# Introduzione ai Large Language Models (LLM)
### Corso RAG per applicazioni ERP e CRM
#### Incontro 1

---

## Agenda dell'incontro
- Benvenuto e introduzione
- Fondamenti dell'AI Generativa
- Storia ed evoluzione dei modelli linguistici
- Architettura Transformer
- Modelli principali sul mercato
- **Pausa (15 min)**
- Setup dell'ambiente di sviluppo
- Primi passi con le API
- Approfondimento funzionalità
- Esercitazione guidata
- Conclusione e compiti

---

## Obiettivi dell'incontro
- Comprendere i concetti fondamentali dell'AI generativa
- Conoscere l'evoluzione dei modelli linguistici
- Capire i principi dell'architettura Transformer
- Identificare i principali LLM disponibili
- Configurare un ambiente di sviluppo funzionante
- Realizzare le prime chiamate API a OpenAI e Claude
- Sviluppare una semplice applicazione di generazione testo

---

## Introduzione e presentazione

### Chi sono
- Salvatore Frasconà
- Esperienze con AI e LLM
- Progetti rilevanti

### Il corso completo
- 8 incontri bisettimanali di 4 ore
- Focus su applicazioni in ambito ERP e CRM
- Equilibrio 50-50 tra teoria e pratica

---

## Fondamenti dell'AI Generativa

### Che cos'è l'AI Generativa?
- Sistemi che **creano** contenuti originali
- Apprendono pattern da grandi dataset
- Generano output simili ma non identici
- Differenze con AI discriminativa (classificazione)

---

## Fondamenti dell'AI Generativa

### Tipologie principali
- **Testo**: LLM come GPT, Claude
- **Immagini**: DALL-E, Midjourney, Stable Diffusion
- **Audio**: sistemi TTS, musica
- **Video**: generazione da prompt o immagini

---

## Fondamenti dell'AI Generativa

### Comprensione vs Generazione
- Modelli discriminativi → comprensione/classificazione
- Modelli generativi → creazione di nuovi contenuti
- LLM moderni → entrambe le capacità
- Base probabilistica della generazione

---

## Storia ed Evoluzione dei Modelli Linguistici

### Primi approcci (1950-2000)
- Sistemi rule-based (ELIZA)
- Modelli statistici n-gram
- Limitazioni: nessuna comprensione semantica

---

## Storia ed Evoluzione dei Modelli Linguistici

### Era pre-Transformer (2000-2017)
- Word2Vec (2013): rappresentazioni vettoriali
- Reti neurali ricorrenti (RNN)
- Long Short-Term Memory (LSTM)
- Problemi: memoria limitata, difficoltà con sequenze lunghe

---

## Storia ed Evoluzione dei Modelli Linguistici

### Rivoluzione Transformer (2017-oggi)
- "Attention is All You Need" (Vaswani et al., 2017)
- BERT (2018): comprensione bidirezionale
- GPT (2018): modello generativo unidirezionale
- Scaling: da milioni a trilioni di parametri

---

## Storia ed Evoluzione dei Modelli Linguistici

### Progressione dei modelli
- GPT-1 (2018): 117M parametri
- GPT-2 (2019): 1.5B parametri
- GPT-3 (2020): 175B parametri
- GPT-4 (2023): >1T parametri (stimati)
- Claude, LLaMA, Gemini: competizione e innovazione

---

## Architettura Transformer

### Principio fondamentale: Attention
- Focalizzazione su parti rilevanti della sequenza
- Cattura relazioni indipendentemente dalla distanza
- Permette parallelizzazione (vs RNN sequenziali)

---

## Architettura Transformer

### Componenti chiave
- **Self-attention**: relazioni all'interno di una sequenza
- **Multi-head attention**: attenzione su diversi aspetti
- **Feed-forward networks**: elaborazione non-lineare
- **Layer normalization**: stabilizzazione dell'addestramento
- **Positional encoding**: informazioni sulla posizione

---

## Architettura Transformer

### Encoder vs Decoder
- **Encoder**: processa l'input bidirezionalmente (BERT)
- **Decoder**: genera output autoregressivamente (GPT)
- **Encoder-Decoder**: traduzione, riassunti (T5, BART)

Approfondimento: https://medium.com/@reyhaneh.esmailbeigi/bert-gpt-and-bart-a-short-comparison-5d6a57175fca

![Architettura Transformer](https://miro.medium.com/max/1400/1*BHzGVskWGS_3jEcYYi6miQ.png)

---

## Modelli Principali sul Mercato

### OpenAI
- **GPT-3.5**: ottimo rapporto prestazioni/costo
- **GPT-4(o)**: multimodale, ragionamento avanzato
- **Instruct/Chat**: varianti ottimizzate per dialogo

---

## Modelli Principali sul Mercato

### Altri modelli proprietari
- **Claude** (Anthropic): focus su sicurezza e allineamento
- **Gemini** (Google): competitore di GPT-4, capacità multimodali
- **Copilot** (Microsoft): basato su GPT, specializzato per coding

---

## Modelli Principali sul Mercato

### Modelli open source
- **LLaMA** (Meta): base per molti modelli derivati
- **Mistral**: eccellenti performance per dimensioni ridotte
- **Falcon**: addestrato su dati multilinguistici
- **Phi-3** (Microsoft): piccole dimensioni, grandi capacità

---

## Modelli Principali sul Mercato

### Confronto prestazioni/dimensioni
| Modello | Parametri | Contesto | Punti di forza |
|---------|-----------|----------|----------------|
| GPT-3.5 | ~175B | 16K | Versatilità, costo |
| GPT-4 | >1T | 128K | Ragionamento, istruzioni |
| Claude 3 | ~>175B | 200K | Contesto lungo, sicurezza |
| Mistral | 7B | 32K | Efficienza, open source |

---

## PAUSA (15 minuti)

---

## Setup dell'Ambiente di Sviluppo

### Prerequisiti
- Python 3.8+ installato
- Pip (gestore pacchetti) funzionante
- Editor di codice (VS Code consigliato)
- Account e API key OpenAI/Anthropic

---

## Setup dell'Ambiente di Sviluppo

### Creazione ambiente virtuale
```bash
# Windows
python -m venv llm_venv
llm_venv\Scripts\activate

# macOS/Linux
python3 -m venv llm_venv
source llm_venv/bin/activate
```

---

## Setup dell'Ambiente di Sviluppo

### Installazione librerie
```bash
pip install openai anthropic python-dotenv requests pandas jupyter
```

### Configurazione API keys
```python
# .env file (non committare su git!)
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
```

---

## Primi Passi con le API

### Struttura base OpenAI
```python
import openai
import os
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

response = openai.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "Sei un assistente CRM esperto."},
        {"role": "user", "content": "Cos'è un lead scoring?"}
    ]
)

print(response.choices[0].message.content)
```

---

## Primi Passi con le API

### Struttura base Anthropic (Claude)
```python
from anthropic import Anthropic
import os
from dotenv import load_dotenv

load_dotenv()
client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

message = client.messages.create(
    model="claude-3-sonnet-20240229",
    max_tokens=300,
    messages=[
        {"role": "user", "content": "Quali sono le best practice per un CRM?"}
    ]
)

print(message.content[0].text)
```

---

## Approfondimento Funzionalità API

### Parametri di generazione
- **temperature**: creatività (0.0-2.0)
- **top_p**: diversità del vocabolario (0.0-1.0)
- **max_tokens**: lunghezza massima output
- **stop**: sequenze per terminare generazione
- **n**: numero di completamenti

---

## Approfondimento Funzionalità API

### Ruoli nei messaggi (OpenAI)
- **system**: istruzioni e contesto generale
- **user**: input dell'utente
- **assistant**: risposte precedenti del modello
- **function**: output di chiamate a funzioni

---

## Approfondimento Funzionalità API

### Gestione token e costi
- Token: unità base di elaborazione (~4 caratteri)
- Input e output vengono conteggiati
- Modelli più potenti = costi maggiori
- Ottimizzazione: usare modelli adeguati al task

---

## Esercitazione Guidata

### Applicazione base: Consulente ERP/CRM
- Interfaccia a riga di comando
- Input utente → LLM → Risposta
- Personalizzazione con diverse istruzioni
- Salvataggio conversazione

---

## Esercitazione Guidata

### Esercizio: modificare parametri
- Provare diverse temperature
- Confrontare modelli diversi
- Modificare le istruzioni di sistema
- Osservare e discutere i risultati

---

## Conclusione

### Riepilogo concetti chiave
- LLM = modelli statistici su larga scala
- Transformer = architettura fondamentale
- Attenzione = meccanismo chiave
- API = interfaccia standardizzata

---

## Conclusione

### Compiti per il prossimo incontro
- Sperimentare con diverse istruzioni di sistema
- Creare un semplice script interattivo
- Leggere documentazione su prompt engineering
- Portare esempi di query ERP/CRM

---

## Risorse aggiuntive
- [Documentazione OpenAI](https://platform.openai.com/docs)
- [Documentazione Anthropic](https://docs.anthropic.com)
- "Attention is All You Need" (paper originale)
- [Hugging Face - corso NLP](https://huggingface.co/learn/nlp-course)
- [LangChain documentazione](https://docs.langchain.com)

---

## Domande?

---

## Grazie dell'attenzione!
### Prossimo incontro: Prompt Engineering Fondamentale
