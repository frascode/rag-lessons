from langchain_anthropic import ChatAnthropic
from langchain.prompts import PromptTemplate

# Una catena che faccia la seguente cosa:
# 1. Valuta il tipo di richiesta dell'utente tra feedback, assistenza o altro.
#   1.1 Se la richiesta è un feedback chiama un action che lancia una post verso un server per "salvare" il feedback.
#   1.2 Se la richiesta è etichettata come assistenza chiama un action che apre una issue su Jira con le specifiche del problema.
#   1.3 Se la richiesta è etichettata come "altro" allora rispondi che non puoi fornire soluzioni per quella richiesta
# 2. Valuta la risposta precedente del modello e assicurati che i feedback non siano stati fraintesi con l'assistenza.
#   2.1 Se la richiesta è stata fraintesa chiama un action che annulla le modifiche delle actions in 1.x
#   2.2 Se la richiesta è coerente con le azioni fatte non effettuare nessun'altra azione.

# Creazione di un prompt template semplice
prompt_template = PromptTemplate(
    input_variables=["query"],
    template="""
        Sei un classificatore di richieste. Il tuo compito è analizzare la seguente richiesta:
            {query}
        
        e fornire il nome della funzione adatta basandoti su questa lista:
            - save_feedback(feedback)
            - open_issue(issue)
        
        oltre al nome della funzione devi ritornare anche i dati utili per passare gli input alle funzioni.
    """
)

def save_feedback(feedback):
    # Qui si deve inserire la logica che salva il feedback sulla propria base dati.
    # Noi per mock ritorneremo True se il Feedback non è nullo.
    print(f"Save feedback {feedback}")
    return feedback is not None

def open_issue(issue):
    # Qui si deve inserire la logica che apre la issue sul proprio sistema di ticketing.
    # Noi per mock ritorneremo True se la Issue non è nulla.
    print(f"Open issue {issue}")
    return issue is not None       


anthropic_llm = ChatAnthropic(
    model="claude-3-7-sonnet-20250219"
)

query_classifier = prompt_template | anthropic_llm

query_classifier.invoke({'query': "Il vostro software è ottimo!"})

# Aggiungere tutti gli elementi necessari per irrobustire la soluzione.
# ...
