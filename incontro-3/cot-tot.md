# Prompt Engineering Avanzato
### Corso RAG per applicazioni ERP e CRM
#### Incontro 3 - [13/05/2025]

---

## Agenda dell'incontro
- Benvenuto e riepilogo incontri precedenti (15 min)
- Tecniche avanzate di Prompt Engineering (30 min)
- Chain-of-Thought e Tree-of-Thought (30 min)
- Prompt templating e parametrizzazione (30 min)
- **Pausa (15 min)**
- LangChain: introduzione e funzionalità per prompting (30 min)
- Laboratorio pratico: implementation di CoT (30 min)
- Laboratorio pratico: LangChain PromptTemplates (45 min)
- Esercitazione: assistente virtuale per query CRM (45 min)
- Conclusione e compiti (15 min)

---

## Obiettivi dell'incontro
- Padroneggiare tecniche avanzate di prompt engineering
- Implementare Chain-of-Thought nei prompt
- Sviluppare prompt template riutilizzabili
- Utilizzare LangChain per gestire prompting complessi
- Creare un assistente CRM con prompting avanzato
- Applicare tutte le tecniche in contesti ERP/CRM reali

---

## Riepilogo Incontri Precedenti

### Incontro 1: Fondamenti LLM
- Architettura Transformer
- Principali modelli sul mercato
- Setup ambiente di sviluppo
- Prime chiamate API

### Incontro 2: Prompt Engineering Base
- Componenti di un prompt efficace
- Zero-shot, one-shot, few-shot
- Problemi comuni e soluzioni
- Contesto nei prompt

---

## Tecniche Avanzate di Prompt Engineering

### Oltre i prompt di base
- Da prompt semplici a sistemi di prompting
- L'importanza del ragionamento strutturato
- Prompt che evolvono e si adattano
- Metacognizione nei modelli linguistici

---

## Tecniche Avanzate di Prompt Engineering

### Principali approcci avanzati
- **Chain-of-Thought**: ragionamento esplicito step-by-step
- **Tree-of-Thought**: esplorazione di percorsi di ragionamento alternativi
- **ReAct**: alternanza di ragionamento e azioni
- **Self-reflection**: valutazione critica dell'output
- **Prompt ensembling**: combinare prompt diversi

---

## Chain-of-Thought Prompting

### Cos'è la Chain-of-Thought?
- Tecnica che stimola il modello a ragionare passo-passo
- "Pensa step by step" come trigger
- Migliora significativamente accuratezza su problemi complessi
- Rende esplicito il processo di ragionamento interno

![Chain-of-Thought Diagram](https://miro.medium.com/max/700/1*8q_6-VMTCjVHN9xk-qA8Ug.png)

---

## Chain-of-Thought Prompting

### Quando utilizzarla
- Problemi matematici e calcoli
- Analisi logica e deduttiva
- Troubleshooting tecnico
- Valutazione di casi aziendali complessi
- Qualsiasi problema che beneficia di decomposizione

---

## Chain-of-Thought Prompting

### Implementazione base
```
Analizza il seguente problema ERP: 
"Un'azienda con 3 filiali utilizza moduli finanziari separati 
che devono essere consolidati. La filiale A riporta ricavi 
di €240.000 con margine 30%, la filiale B riporta ricavi di 
€180.000 con margine 25%, la filiale C riporta ricavi di 
€320.000 con margine 35%. Calcola il ricavo totale, il 
profitto totale e il margine percentuale complessivo."

Ragiona passo-passo per risolvere il problema.
```

---

## Chain-of-Thought Prompting

### Few-Shot CoT
```
Ecco come risolvo problemi complessi di ERP:

Problema 1: "Un CRM ha 150 lead nel pipeline con conversione 
media del 20%. Se ogni cliente genera €2.000, quale revenue potenziale?"
Ragionamento: Devo calcolare quanti lead diventeranno clienti. 
150 lead × 20% conversione = 30 clienti. Poi calcolo il ricavo:
30 clienti × €2.000 = €60.000 di ricavo potenziale.

Il tuo problema: [PROBLEMA]
```

---

## Tree-of-Thought Prompting

### Cos'è il Tree-of-Thought?
- Estensione del Chain-of-Thought
- Esplora multiple linee di ragionamento parallele
- Valuta diverse alternative e seleziona la migliore
- Permette backtracking quando un approccio fallisce

---

## Tree-of-Thought Prompting

### Implementazione base
```
Problema: "Quale strategia di migrazione ERP sarebbe migliore per 
un'azienda manifatturiera di medie dimensioni: big bang, phased 
roll-out, o parallel adoption?"

Considera tre diversi approcci alla soluzione:
Approccio 1: Valutazione basata sui rischi
[Ragiona passo-passo]

Approccio 2: Valutazione basata sui costi
[Ragiona passo-passo]

Approccio 3: Valutazione basata sull'impatto operativo
[Ragiona passo-passo]

Conclusione: Confronta i tre approcci e determina la soluzione migliore.
```

---

## ReAct Prompting

### Reasoning + Acting
- Combinazione di ragionamento e azioni
- Il modello alterna:
  - **Thought**: ragionamento sul problema
  - **Action**: azione da intraprendere
  - **Observation**: risultato dell'azione
  - **New Thought**: rivalutazione

---

## ReAct Prompting

### Esempio applicato a CRM
```
Sei un consulente CRM che deve diagnosticare problemi
di implementazione. Segui un approccio ReAct:

Problema: "Gli agenti lamentano che il nuovo CRM è più lento 
del precedente nel caricare le schede cliente."

Thought: Devo identificare potenziali cause di lentezza.
Action: Verificare se il problema riguarda tutte le schede cliente o solo alcune.
Observation: Solo le schede di clienti con storico ordini lungo sono lente.
Thought: Potrebbe essere un problema di query al database...
```

---

## Self-Reflection Prompting

### Valutazione critica dell'output
- Il modello genera una risposta e poi la valuta
- Identifica errori, omissioni, bias
- Propone miglioramenti
- Iterazione fino a soddisfare criteri di qualità

---

## Self-Reflection Prompting

### Esempio in contesto ERP
```
Problema: "Suggerisci i KPI più importanti per monitorare 
l'efficacia di un nuovo sistema ERP in ambito manifatturiero."

Prima risposta: [RISPOSTA INIZIALE]

Ora valuta la risposta precedente considerando:
1. Sono inclusi KPI da tutte le aree rilevanti?
2. I KPI sono misurabili e specifici?
3. Mancano KPI essenziali per il settore manifatturiero?
4. La risposta è applicabile alla realtà aziendale?

Sulla base di questa valutazione, fornisci una risposta migliorata.
```

---

## Prompt Templating e Parametrizzazione

### Cos'è il prompt templating?
- Creazione di strutture riutilizzabili per i prompt
- Separazione di struttura e contenuto variabile
- Personalizzazione dinamica in base al contesto
- Standardizzazione per consistenza nei risultati

---

## Prompt Templating e Parametrizzazione

### Struttura di un template
```python
template = """
Sei un esperto di {sistema} con esperienza in {industria}.
Devi {azione} per il cliente che sta affrontando il seguente problema:
"{problema}"

Fornisci una risposta che includa:
{sezioni}

Il tuo output deve essere dettagliato ma conciso, adatto a un {audience} 
con livello di competenza {livello}.
"""
```

---

## Prompt Templating e Parametrizzazione

### Esempio di implementazione Python
```python
def create_erp_consultant_prompt(system, industry, problem):
    template = """
    Sei un consulente specializzato in {system} con focus su {industry}.
    Analizza il seguente problema: "{problem}"
    
    Fornisci:
    1. Diagnosi della causa principale
    2. Soluzione step-by-step
    3. Preventivo di tempi e costi
    4. Suggerimenti per prevenzione futura
    """
    
    return template.format(
        system=system,
        industry=industry,
        problem=problem
    )

# Utilizzo
prompt = create_erp_consultant_prompt(
    "SAP S/4HANA", 
    "manufacturing", 
    "Errori di sincronizzazione tra produzione e inventario"
)
```

---

## LangChain: Introduzione e Funzionalità

### Cos'è LangChain?
- Framework per applicazioni LLM
- Facilita creazione di sistemi complessi
- Componenti modulari e riutilizzabili
- Focus sulle catene di elaborazione

---

## LangChain: Introduzione e Funzionalità

### Componenti principali per prompting
- **PromptTemplates**: gestione template di prompt
- **LLMChain**: collegamento tra prompt e modelli
- **SequentialChain**: concatenazione di passaggi
- **Memory**: gestione del contesto conversazionale
- **Tools/Agents**: azioni basate sul ragionamento

---

## LangChain: PromptTemplates

### Creazione e gestione di template
```python
from langchain.prompts import PromptTemplate

template = """
Sei un consulente {system} con {years} anni di esperienza.
Il cliente chiede: "{question}"
Fornisci una risposta professionale.
"""

prompt_template = PromptTemplate(
    input_variables=["system", "years", "question"],
    template=template
)

# Utilizzo
formatted_prompt = prompt_template.format(
    system="SAP", 
    years="10",
    question="Come posso integrare il modulo HR con Finance?"
)
```

---

## LangChain: Few-Shot Templates

### Template con esempi
```python
from langchain.prompts import FewShotPromptTemplate, PromptTemplate

examples = [
    {"query": "Errore nel workflow di approvazione", 
     "response": "Verifica i livelli di autorizzazione..."}, 
    {"query": "Report finanziario non aggiornato", 
     "response": "Controlla lo scheduling dei job batch..."}
]

example_formatter = PromptTemplate(
    input_variables=["query", "response"],
    template="Query: {query}\nRisposta: {response}"
)

few_shot_prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_formatter,
    prefix="Sei un esperto tecnico SAP. Ecco alcuni esempi:",
    suffix="Query: {input_query}\nRisposta:",
    input_variables=["input_query"]
)
```

---

## LangChain: LLMChain

### Collegamento di prompt e modelli
```python
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain

# Inizializzazione del modello
llm = ChatOpenAI(model_name="gpt-3.5-turbo")

# Creazione della catena
chain = LLMChain(
    llm=llm,
    prompt=prompt_template,
    verbose=True
)

# Esecuzione
response = chain.run(
    system="Oracle ERP", 
    years="8", 
    question="Come automatizzare la riconciliazione?"
)
```

---

## LangChain: Sequential Chains

### Concatenazione di passaggi
```python
from langchain.chains import SimpleSequentialChain

# Chain per analisi del problema
analysis_chain = LLMChain(
    llm=llm,
    prompt=analysis_prompt_template,
    output_key="analysis"
)

# Chain per generazione soluzione
solution_chain = LLMChain(
    llm=llm,
    prompt=solution_prompt_template,
    output_key="solution"
)

# Concatenazione
sequential_chain = SimpleSequentialChain(
    chains=[analysis_chain, solution_chain],
    verbose=True
)
```

---

## PAUSA (15 minuti)

---

## Laboratorio pratico: Implementazione di CoT

Implementiamo insieme Chain-of-Thought per:
1. Analisi KPI per CRM
2. Troubleshooting ERP
3. Valutazione ROI implementazione

Codice e struttura nei file di esercitazione

---

## Laboratorio pratico: LangChain PromptTemplates

Creiamo insieme:
1. Template per assistente ERP con esempi few-shot
2. Catene sequenziali per analisi/soluzione problemi
3. Template con ragionamento CoT incorporato

Esercizi guidati nei file di laboratorio

---

## Esercitazione: Assistente virtuale per query CRM

Realizziamo un assistente completo che:
1. Analizza query clienti su sistema CRM
2. Utilizza CoT per troubleshooting
3. Genera risposte dettagliate con template
4. Gestisce diversi scenari d'uso (vendite, supporto, analisi)

Progetto completo nei file di esercitazione

---

## Conclusione e compiti

### Riepilogo concetti chiave
- Chain-of-Thought migliora ragionamento complesso
- Template standardizzano e migliorano qualità
- LangChain semplifica gestione prompt avanzati
- Approcci dinamici superano limiti dei prompt statici

---

## Conclusione e compiti

### Compiti per il prossimo incontro
- Estendere assistente virtuale con memoria contestuale
- Implementare un esempio di Tree-of-Thought per problema complesso
- Creare template personalizzati per vostra area di interesse
- Leggere documentazione su database vettoriali (prossimo argomento)

---

## Risorse aggiuntive
- [LangChain Documentation](https://python.langchain.com/docs/get_started)
- [Chain-of-Thought Prompting - Paper originale](https://arxiv.org/abs/2201.11903)
- [Tree-of-Thought Prompting - Paper](https://arxiv.org/abs/2305.10601)
- [Anthropic Claude Cookbook - Advanced Prompting](https://docs.anthropic.com/claude/docs/advanced-prompting)
- [ReAct Prompting Paper](https://arxiv.org/abs/2210.03629)

---

## Domande?

---

## Grazie dell'attenzione!
### Prossimo incontro: RAG
