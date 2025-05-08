import openai
import os
from dotenv import load_dotenv

load_dotenv()
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Esempio di Few-shot prompting per analisi feedback cliente
def analyze_feedback_few_shot(feedback):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "Sei un analista CRM esperto."},
            {"role": "user", "content": f"""Classifica il seguente feedback cliente come POSITIVO, NEUTRO o NEGATIVO e fornisci una breve spiegazione:

Esempi:
Feedback: "Il vostro ERP è potente ma l'onboarding è stato più lungo del previsto. Ora che siamo operativi siamo soddisfatti."
Classificazione: NEUTRO
Spiegazione: Il feedback contiene elementi positivi (sistema potente, soddisfazione attuale) e negativi (onboarding lungo), risultando complessivamente bilanciato.

Feedback: "L'aggiornamento all'ultima versione del CRM ha risolto tutti i problemi precedenti e ha aggiunto funzionalità che ci hanno permesso di aumentare le conversioni del 15%."
Classificazione: POSITIVO
Spiegazione: Il feedback menziona esclusivamente miglioramenti e risultati positivi quantificabili.

Feedback: "Dopo sei mesi di utilizzo, il rapporto costo-beneficio è deludente. Le funzionalità promesse in fase di vendita non sono all'altezza e i bug frequenti rallentano il nostro lavoro quotidiano."
Classificazione: NEGATIVO
Spiegazione: Il feedback esprime delusione, funzionalità inadeguate e problemi tecnici senza menzionare aspetti positivi.

Ora classifica questo:
'{feedback}'"""}
        ]
    )
    return response.choices[0].message.content

# Test con gli stessi feedback degli esempi precedenti
feedback_samples = [
    "Il vostro sistema CRM ha migliorato significativamente la nostra gestione dei clienti. L'interfaccia è intuitiva e il supporto è eccellente.",
    "L'implementazione è andata bene ma abbiamo riscontrato alcuni problemi con l'integrazione dei dati legacy. Il supporto tecnico ha risposto con tempi accettabili.",
    "Dopo tre mesi dall'implementazione, continuiamo a riscontrare errori frequenti. La formazione fornita è stata insufficiente e il sistema è troppo complesso per i nostri operatori."
]

print("=== ANALISI FEEDBACK CON FEW-SHOT LEARNING ===\n")
for i, feedback in enumerate(feedback_samples):
    print(f"Feedback #{i+1}:")
    print(f"\"{feedback}\"\n")
    result = analyze_feedback_few_shot(feedback)
    print(f"Analisi:\n{result}\n")
    print("-" * 50)
