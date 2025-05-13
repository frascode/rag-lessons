import openai
import os
from dotenv import load_dotenv

load_dotenv()
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Esempio di Zero-shot prompting per analisi sentiment feedback cliente
def analyze_feedback_zero_shot(feedback):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "Sei un analista CRM esperto."},
            {"role": "user", "content": f"Classifica pensando nel seguente modo: - Tono del feedback; - Lunghezza del feedback. Ll seguente feedback cliente fornendo un punteggio da 1 a 10.\n\n'{feedback}'"}
        ]
    )
    return response.choices[0].message.content

# Test con alcuni feedback
feedback_samples = [ 
    "Il vostro sistema CRM ha migliorato significativamente la nostra gestione dei clienti. L'interfaccia è intuitiva e il supporto è eccellente.",
    "L'implementazione è andata bene ma abbiamo riscontrato alcuni problemi con l'integrazione dei dati legacy. Il supporto tecnico ha risposto con tempi accettabili.",
    "Dopo tre mesi dall'implementazione, continuiamo a riscontrare errori frequenti. La formazione fornita è stata insufficiente e il sistema è troppo complesso per i nostri operatori."
]

print("=== ANALISI FEEDBACK CON ZERO-SHOT LEARNING ===\n")
for i, feedback in enumerate(feedback_samples):
    print(f"Feedback #{i+1}:")
    print(f"\"{feedback}\"\n")
    result = analyze_feedback_zero_shot(feedback)
    print(f"Analisi:\n{result}\n")
    print("-" * 50)
