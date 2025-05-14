from langchain.chat_models import ChatAnthropic, ChatOpenAI
from langchain.prompts import PromptTemplate, ChatPromptTemplate, HumanMessagePromptTemplate
from langchain.chains import LLMChain, SequentialChain
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
import os
from dotenv import load_dotenv
import json

load_dotenv()

"""
STRUCTURED OUTPUT & SEQUENTIAL CHAINS EXAMPLE
Esempio avanzato di LangChain che combina:
1. Output strutturato con parser
2. Catene sequenziali per analisi multi-step
3. Prompt specializzati per contesti ERP/CRM
"""

# Inizializzazione modello
model = ChatAnthropic(
    model="claude-3-sonnet-20240229",
    temperature=0.2
)

# Schemi per output strutturato
response_schemas = [
    ResponseSchema(name="problem_category", 
                   description="La categoria del problema (Technical, Process, Data, Integration, Security)"),
    ResponseSchema(name="root_causes", 
                   description="Lista delle possibili cause principali del problema"),
    ResponseSchema(name="severity", 
                   description="Livello di severità del problema: Low, Medium, High, Critical"),
    ResponseSchema(name="business_impact", 
                   description="Descrizione dell'impatto sul business"),
    ResponseSchema(name="immediate_actions", 
                   description="Lista di azioni immediate da intraprendere"),
    ResponseSchema(name="long_term_solutions", 
                   description="Lista di soluzioni a lungo termine")
]

# Parser per output strutturato
output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
format_instructions = output_parser.get_format_instructions()

# Chain 1: Analisi del problema
analysis_template = """
Sei un consulente esperto in sistemi ERP/CRM con 15+ anni di esperienza in troubleshooting.

PROBLEMA: {problem}
CONTESTO: {context}

Analizza questo problema e fornisci una valutazione dettagliata che includa:
1. La categoria del problema
2. Le possibili cause principali
3. Livello di severità del problema
4. Impatto sul business

{format_instructions}
"""

analysis_prompt = PromptTemplate(
    template=analysis_template,
    input_variables=["problem", "context"],
    partial_variables={"format_instructions": format_instructions}
)

analysis_chain = LLMChain(
    llm=model,
    prompt=analysis_prompt,
    output_key="analysis"
)

# Chain 2: Generazione soluzione basata sull'analisi
solution_template = """
Sei un consulente senior in sistemi ERP/CRM specializzato in soluzioni pratiche e implementabili.

PROBLEMA ORIGINALE: {problem}
CONTESTO: {context}

ANALISI TECNICA: {analysis}

In base all'analisi fornita, sviluppa un piano d'azione completo che includa:
1. Azioni immediate da intraprendere nelle prossime 24-48 ore
2. Soluzioni a lungo termine per prevenire il ripetersi del problema
3. Raccomandazioni per migliorare i processi correlati

Il piano deve essere concreto, dettagliato e adatto al contesto dell'azienda.

{format_instructions}
"""

solution_prompt = PromptTemplate(
    template=solution_template,
    input_variables=["problem", "context", "analysis"],
    partial_variables={"format_instructions": format_instructions}
)

solution_chain = LLMChain(
    llm=model,
    prompt=solution_prompt,
    output_key="solution"
)

# Concatenazione delle catene
erp_troubleshooting_chain = SequentialChain(
    chains=[analysis_chain, solution_chain],
    input_variables=["problem", "context"],
    output_variables=["analysis", "solution"],
    verbose=True
)

# Funzione per elaborare e mostrare le risposte in modo ordinato
def process_response(response_text, title):
    print(f"\n=== {title} ===\n")
    try:
        # Cerchiamo di estrarre la parte JSON dalla risposta
        json_str = response_text.split("```json")[1].split("```")[0].strip()
        data = json.loads(json_str)
        for key, value in data.items():
            print(f"\n{key.upper().replace('_', ' ')}:")
            if isinstance(value, list):
                for i, item in enumerate(value):
                    print(f"  {i+1}. {item}")
            else:
                print(f"  {value}")
    except:
        # Se fallisce l'estrazione JSON, mostriamo il testo completo
        print(response_text)

# Funzione demo per eseguire la catena
def run_erp_troubleshoot_demo():
    # Problema di esempio ERP
    problem = """
    Dopo l'ultimo aggiornamento del sistema ERP (versione 4.5.2), diverse transazioni finanziarie 
    falliscono durante la fase di posting con errore 'Invalid cost center assignment'. Questo 
    problema sta ritardando la chiusura mensile.
    """
    
    # Contesto aziendale
    context = """
    Azienda: Multinazionale manifatturiera con 2000 dipendenti
    Sistema ERP: SAP S/4HANA 2021
    Moduli implementati: FI, CO, MM, PP, SD, HR
    Personalizzazioni: Moderate (20% custom code)
    Recenti modifiche: Aggiornamento versione eseguito 5 giorni fa, migrazione da S/4HANA 2020
    Urgenza: Alta (chiusura mensile prevista entro 3 giorni)
    """
    
    # Esecuzione della catena
    result = erp_troubleshooting_chain({
        "problem": problem,
        "context": context
    })
    
    # Elaborazione e visualizzazione delle risposte
    process_response(result["analysis"], "ANALISI DEL PROBLEMA")
    process_response(result["solution"], "PIANO D'AZIONE")

if __name__ == "__main__":
    run_erp_troubleshoot_demo()
