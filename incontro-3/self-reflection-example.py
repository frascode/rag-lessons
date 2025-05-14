from langchain_anthropic import ChatAnthropic
from langchain.prompts import PromptTemplate
from langchain.output_parsers import ResponseSchema, StructuredOutputParser

import json


"""
SELF-REFLECTION PROMPTING EXAMPLE
Implementazione che dimostra la tecnica di prompt con auto-valutazione,
dove il modello prima genera una risposta e poi la valuta criticamente per migliorarla.
"""

# Inizializzazione modello
model = ChatAnthropic(
    model="claude-3-7-sonnet-20250219",
)

# Prompt per la risposta iniziale
initial_prompt_template = """
Sei un consulente senior specializzato in sistemi ERP per il settore manifatturiero.

RICHIESTA DEL CLIENTE:
{query}

Fornisci una risposta professionale ed esaustiva a questa richiesta, basandoti 
sulla tua esperienza con sistemi ERP in ambito manifatturiero.
"""

initial_prompt = PromptTemplate(
    input_variables=["query"],
    template=initial_prompt_template
)

# Prompt per l'auto-valutazione
reflection_prompt_template = """
Sei un revisore esperto con competenze in sistemi ERP e business consulting.

RICHIESTA ORIGINALE DEL CLIENTE:
{query}

RISPOSTA FORNITA:
{initial_response}

Ora, valuta criticamente questa risposta considerando i seguenti criteri:
1. Completezza: La risposta copre tutti gli aspetti della domanda?
2. Precisione tecnica: Le informazioni tecniche sono accurate e aggiornate?
3. Applicabilità pratica: I consigli sono implementabili nel contesto specifico?
4. Chiarezza: La risposta è ben strutturata e facilmente comprensibile?
5. Personalizzazione: La risposta tiene conto delle specificità del settore manifatturiero?

Per ciascun criterio, assegna un punteggio da 1 a 5 e fornisci spiegazioni dettagliate.
Inoltre, identifica eventuali omissioni o imprecisioni nella risposta.

{format_instructions}
"""

# Schema per l'output strutturato della valutazione
reflection_schemas = [
    ResponseSchema(name="completeness", 
                  description="Valutazione della completezza (1-5) con spiegazione"),
    ResponseSchema(name="technical_accuracy", 
                  description="Valutazione della precisione tecnica (1-5) con spiegazione"),
    ResponseSchema(name="practical_applicability", 
                  description="Valutazione dell'applicabilità pratica (1-5) con spiegazione"),
    ResponseSchema(name="clarity", 
                  description="Valutazione della chiarezza e struttura (1-5) con spiegazione"),
    ResponseSchema(name="customization", 
                  description="Valutazione della personalizzazione per il settore manifatturiero (1-5) con spiegazione"),
    ResponseSchema(name="omissions", 
                  description="Lista delle omissioni o aspetti non considerati nella risposta"),
    ResponseSchema(name="inaccuracies", 
                  description="Lista delle imprecisioni o errori tecnici nella risposta"),
    ResponseSchema(name="overall_score", 
                  description="Punteggio complessivo (1-5) considerando tutti i criteri")
]

# Parser per output strutturato della valutazione
reflection_parser = StructuredOutputParser.from_response_schemas(reflection_schemas)
reflection_format_instructions = reflection_parser.get_format_instructions()

reflection_prompt = PromptTemplate(
    input_variables=["query", "initial_response"],
    template=reflection_prompt_template,
    partial_variables={"format_instructions": reflection_format_instructions}
)

# Prompt per la risposta migliorata
improvement_prompt_template = """
Sei un consulente senior specializzato in sistemi ERP per il settore manifatturiero.

RICHIESTA ORIGINALE DEL CLIENTE:
{query}

RISPOSTA INIZIALE:
{initial_response}

VALUTAZIONE CRITICA:
{reflection}

Basandoti sulla valutazione critica sopra indicata, crea una versione migliorata 
della risposta originale che risolva le omissioni, corregga le imprecisioni e 
migliori gli aspetti carenti identificati nella valutazione.

La risposta migliorata deve essere completa, tecnicamente accurata, praticamente 
applicabile, chiara e ben strutturata, e personalizzata per il settore manifatturiero.
"""

improvement_prompt = PromptTemplate(
    input_variables=["query", "initial_response", "reflection"],
    template=improvement_prompt_template
)


initial_chain = initial_prompt | model

reflection_chain = reflection_prompt | model

improvement_chain = improvement_prompt | model

# Query di esempio
example_queries = [
    """
    Stiamo valutando l'implementazione di un nuovo sistema ERP per la nostra azienda 
    manifatturiera (produzione di componenti automobilistici, 250 dipendenti, 
    fatturato €30M). Quali moduli dovrebbero avere la priorità nella prima fase 
    di implementazione e come dovremmo strutturare il processo di migrazione per 
    minimizzare l'impatto sulle operazioni quotidiane?
    """,
    
    """
    Il nostro sistema ERP (SAP S/4HANA) sta generando report di produzione con tempi 
    ciclo anomali in alcuni reparti. I dati non corrispondono a quanto osservato sul 
    campo dagli operatori. Come possiamo identificare l'origine di queste discrepanze 
    e correggerle per avere dati affidabili per la pianificazione della produzione?
    """
]

def execute_self_reflection_chain(query):
    """Esegue la catena completa di risposta iniziale, auto-valutazione e miglioramento"""
    print(f"\n\n=== QUERY: ===\n{query}")
    print("\n" + "="*80 + "\n")
    
    # Passo 1: Generare la risposta iniziale
    print("GENERAZIONE RISPOSTA INIZIALE...")
    initial_result = initial_chain.invoke({'query': query})
    print("\nRISPOSTA INIZIALE:")
    print(initial_result.content)
    print("\n" + "="*80 + "\n")
    
    # Passo 2: Valutare criticamente la risposta
    print("GENERAZIONE AUTO-VALUTAZIONE...")
    reflection_result = initial_chain.invoke({'query': query, 'initial_response':initial_result})
    print("\nAUTO-VALUTAZIONE:")
    print(reflection_result.content)
    print("\n" + "="*80 + "\n")
    
    # Passo 3: Generare la risposta migliorata
    print("GENERAZIONE RISPOSTA MIGLIORATA...")
    print(f"CRITICA FATTA DAL MODELLO: {reflection_result.content}")
    improved_result = improvement_chain.invoke({'query': query, 'initial_response': initial_result, 'reflection': reflection_result})

    print("\nRISPOSTA MIGLIORATA:")
    print(improved_result.content)
    print("\n" + "="*80 + "\n")
    
    # Riepilogo miglioramenti
    # print("CONFRONTO:")
    # try:
    #     reflection_json = extract_json_from_markdown(reflection_result)
    #     reflection_data = json.loads(reflection_json)
    #     
    #     print(f"Punteggio complessivo: {reflection_data.get('overall_score', 'N/A')}/5")
    #     print("\nPrincipalI omissioni nella risposta iniziale:")
    #     for omission in reflection_data.get('omissions', []):
    #         print(f"- {omission}")
    #     
    #     print("\nPrincipalI imprecisioni nella risposta iniziale:")
    #     for inaccuracy in reflection_data.get('inaccuracies', []):
    #         print(f"- {inaccuracy}")
    # except:
    #     print("Non è stato possibile estrarre i dati strutturati dalla valutazione.")
    # 
    # return {
    #     "initial_response": initial_result,
    #     "reflection": reflection_result,
    #     "improved_response": improved_result
    # }

def extract_json_from_markdown(text):
    """Estrae JSON da una risposta in formato markdown"""
    if "```json" in text:
        return text.split("```json")[1].split("```")[0].strip()
    elif "```" in text:
        return text.split("```")[1].split("```")[0].strip()
    else:
        # Tentativo di trovare JSON nell'output
        import re
        match = re.search(r'({.*})', text, re.DOTALL)
        if match:
            return match.group(1)
        return text

def run_self_reflection_demo():
    """Esegue la demo di self-reflection prompting"""
    print("=== DEMO SELF-REFLECTION PROMPTING ===\n")
    print("Questa demo mostra come un modello LLM può generare una risposta, auto-valutarla e migliorarla.")
    
    # Eseguiamo la catena su una delle query di esempio
    execute_self_reflection_chain(example_queries[0])

if __name__ == "__main__":
    run_self_reflection_demo()
