from anthropic import Anthropic
import os
from dotenv import load_dotenv

load_dotenv()
client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

"""
CHAIN-OF-THOUGHT PROMPTING EXAMPLE
Implementazione che dimostra come migliorare il ragionamento complesso
in contesti ERP/CRM tramite l'utilizzo di Chain-of-Thought.

Questo script confronta approcci standard e CoT sulla stessa query.
"""

# Query problema complesso relativo a ERP
problem_query = """
La nostra azienda (settore manifatturiero con 250 dipendenti) sta valutando l'implementazione 
di un nuovo ERP. Abbiamo ricevuto preventivi da tre fornitori:

Fornitore A: Costo licenze €120.000, costo implementazione €200.000, costo manutenzione annua €30.000. 
Tempo stimato implementazione: 8 mesi. Moduli: HR, Finance, Produzione, Magazzino, CRM base.

Fornitore B: Costo licenze €90.000, costo implementazione €250.000, costo manutenzione annua €25.000. 
Tempo stimato implementazione: 10 mesi. Moduli: HR, Finance, Produzione, Magazzino, CRM avanzato, BI.

Fornitore C: Soluzione cloud, costo annuo €80.000 tutto incluso. 
Tempo stimato implementazione: 6 mesi. Moduli: HR, Finance, Produzione, Magazzino, CRM base, supporto mobile.

Considerando un orizzonte temporale di 5 anni, quale fornitore offre il miglior rapporto qualità/prezzo 
e perché? Considera anche fattori qualitativi oltre al TCO.
"""

# Prompt standard senza CoT
standard_prompt = f"""
Sei un consulente esperto in sistemi ERP e CRM.

{problem_query}
"""

# Prompt con Chain-of-Thought
cot_prompt = f"""
Sei un consulente esperto in sistemi ERP e CRM.

{problem_query}

Ragiona passo per passo per analizzare approfonditamente la situazione e determinare la migliore soluzione.
Considera i seguenti aspetti in sequenza:
1. Calcola il TCO (Total Cost of Ownership) per ciascun fornitore su 5 anni
2. Valuta il rapporto tempo/funzionalità per ciascuna opzione
3. Confronta i vantaggi qualitativi delle diverse soluzioni
4. Analizza rischi e opportunità di ciascun approccio
5. Fornisci una raccomandazione finale basata sull'analisi complessiva
"""

# Funzione per ottenere risposta da Claude
def get_claude_response(prompt):
    message = client.messages.create(
        model="claude-3-7-sonnet-20250219",
        max_tokens=1500,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    return message.content[0].text

# Esecuzione con entrambi gli approcci
print("=== RISPOSTA CON PROMPT STANDARD ===")
standard_response = get_claude_response(standard_prompt)
print(standard_response)
print("\n" + "="*80 + "\n")

print("=== RISPOSTA CON CHAIN-OF-THOUGHT ===")
cot_response = get_claude_response(cot_prompt)
print(cot_response)
