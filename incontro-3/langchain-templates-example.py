from langchain_anthropic import ChatAnthropic
from langchain.prompts import PromptTemplate, FewShotPromptTemplate
"""
LANGCHAIN PROMPTING EXAMPLE
Dimostrazione di utilizzo di LangChain per creare e gestire template di prompting
avanzati in contesti ERP/CRM.
"""

anthropic_llm = ChatAnthropic(
    model="claude-3-7-sonnet-20250219",
    max_tokens_to_sample = 10000
)

# Creazione di un prompt template semplice
simple_template = PromptTemplate(
    input_variables=["system", "problem", "company_type"],
    template="""
    Sei un consulente esperto del sistema {system}.
    
    PROBLEMA:
    {problem}
    
    CONTESTO:
    L'azienda è di tipo {company_type}.
    
    Fornisci una soluzione dettagliata e step-by-step per risolvere questo problema.
    """
)

# Creazione di un prompt template con Chain-of-Thought
cot_template = PromptTemplate(
    input_variables=["system", "problem", "company_type"],
    template="""
    Sei un consulente senior del sistema {system} con 10+ anni di esperienza.
    
    PROBLEMA:
    {problem}
    
    CONTESTO:
    L'azienda è di tipo {company_type}.
    
    Affronta il problema ragionando passo per passo:
    1. Analizza le possibili cause del problema
    2. Considera le implicazioni tecniche e di business
    3. Valuta le possibili soluzioni alternative
    4. Seleziona la soluzione ottimale
    5. Fornisci un piano di implementazione dettagliato
    
    Mostra esplicitamente ogni passaggio del tuo ragionamento.
    """
)

# Creazione di un FewShotPromptTemplate per casi d'uso di troubleshooting ERP
examples = [
    {
        "problem": "Il modulo Finance non si sincronizza con HR",
        "analysis": """
        Cause possibili:
        1. Mapping errato dei campi tra i moduli
        2. Problemi nei job scheduler di sincronizzazione
        3. Errori nelle configurazioni di integrazione
        4. Permessi insufficienti per gli account di servizio
        
        La causa più probabile è un errore di configurazione nei job scheduler
        perché questo tipo di problema tipicamente si manifesta dopo aggiornamenti
        o modifiche di configurazione."""
    },
    {
        "problem": "Gli utenti non riescono ad accedere al CRM da mobile",
        "analysis": """
        Cause possibili:
        1. Problemi di autenticazione specifica per mobile
        2. Vincoli di sicurezza della rete aziendale
        3. Incompatibilità versione app mobile
        4. Configurazione errata del gateway mobile
        
        La causa più probabile è legata alla configurazione del gateway mobile
        perché questi problemi solitamente si verificano quando ci sono restrizioni
        di rete o certificati di sicurezza non aggiornati."""
    }
]

example_formatter = PromptTemplate(
    input_variables=["problem", "analysis"],
    template="PROBLEMA: {problem}\nANALISI: {analysis}"
)

few_shot_prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_formatter,
    prefix="""Sei un esperto di troubleshooting per sistemi ERP e CRM.
    Ecco alcuni esempi di come analizzo problemi simili:""",
    suffix="""
    PROBLEMA: {problem}
    
    Fornisci un'analisi dettagliata seguendo l'approccio mostrato negli esempi:""",
    input_variables=["problem"]
)

# Creazione delle catene LLM
simple_chain = simple_template | anthropic_llm
cot_chain = cot_template | anthropic_llm
few_shot_chain = few_shot_prompt | anthropic_llm


# Primo passo
categorization = PromptTemplate(
    input_variables=["query"],
    template="""
        Sei un classificatore e devi etichettare la query secondo queste possibilità:
            - feedback.
            - assistenza.
            - altro.
    """
)

chabot_nts = categorization | anthropic_llm 

# Una catena che faccia la seguente cosa:
# 1. Valuta il tipo di richiesta dell'utente tra feedback, assistenza o altro.
#   1.1 Se la richiesta è un feedback chiama un action che lancia una post verso un server per "salvare" il feedback.
#   1.2 Se la richiesta è etichettata come assistenza chiama un action che apre una issue su Jira con le specifiche del problema.
#   1.3 Se la richiesta è etichettata come "altro" allora rispondi che non puoi fornire soluzioni per quella richiesta
# 2. Valuta la risposta precedente del modello e assicurati che i feedback non siano stati fraintesi con l'assistenza.
#   2.1 Se la richiesta è stata fraintesa chiama un action che annulla le modifiche delle actions in 1.x
#   2.2 Se la richiesta è coerente con le azioni fatte non effettuare nessun'altra azione.


# Funzione demo per eseguire le catene
def run_chains_demo():
    # Input del problema
    problem = "Gli utenti segnalano che la creazione di nuovi ordini è molto lenta e a volte va in timeout"
    system = "SAP S/4HANA"
    company_type = "manufacturing con 500 dipendenti"
    
    # print("\n=== PROMPT TEMPLATE SEMPLICE ===")
    # simple_result = simple_chain.invoke({'system': system, 'problem': problem, 'company_type': company_type})
    # print(simple_result.content)

    
    print("\n=== PROMPT TEMPLATE CON CHAIN-OF-THOUGHT SAMPLE 1 ===")
    cot_result = cot_chain.invoke({'system': system, 'problem': problem, 'company_type': company_type})
    print(cot_result.content)

    
    print("\n=== PROMPT TEMPLATE CON CHAIN-OF-THOUGHT SAMPLE 2 ===")
    cot_result = cot_chain.invoke({'system': system, 'problem': problem, 'company_type': company_type})
    print(cot_result.content)

    # print("\n=== PROMPT TEMPLATE FEW-SHOT ===")
    # few_shot_result = few_shot_chain.invoke({'system': system, 'problem': problem, 'company_type': company_type})
    # print(few_shot_result.content)

if __name__ == "__main__":
    run_chains_demo()
