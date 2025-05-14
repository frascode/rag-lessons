from anthropic import Anthropic
import os
from dotenv import load_dotenv

load_dotenv()
client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

"""
TREE-OF-THOUGHT PROMPTING EXAMPLE
Implementazione che dimostra il ragionamento a più percorsi per l'analisi
di un problema complesso relativo a CRM, esplorando diverse alternative.
"""

problem_query = """
La nostra azienda B2B (software per industria farmaceutica, 50 dipendenti, 200 clienti) 
sta riscontrando un aumento del churn rate dal 5% al 15% nell'ultimo anno. 
Dobbiamo decidere come affrontare la situazione considerando che:

- I feedback di uscita indicano insoddisfazione sul supporto tecnico e nuove funzionalità
- Il team di supporto è composto da 5 persone e già sovraccarico
- I competitor hanno recentemente rilasciato nuove funzionalità
- Il budget per miglioramenti è limitato a €100.000 per i prossimi 6 mesi
- Il ciclo di vendita è lungo (6-9 mesi) e acquisire nuovi clienti costa 5 volte più che mantenere quelli esistenti

Quale sarebbe la migliore strategia per ridurre il churn rate e migliorare la retention?
"""

# Prompt con Tree-of-Thought
tot_prompt = f"""
Sei un esperto consulente di strategia per aziende SaaS B2B con focus su CRM e customer retention.

{problem_query}

Analizza il problema esplorando tre diverse direzioni strategiche. Per ciascuna, sviluppa un ragionamento approfondito e valuta pro/contro:

PERCORSO STRATEGICO 1: Miglioramento del supporto tecnico
- Analisi dettagliata: [Ragiona passo-passo su come potenziare il supporto con risorse limitate]
- Valutazione costi/benefici: [Analizza l'impatto economico di questa strategia]
- Timeline di implementazione: [Stima i tempi necessari]
- Pro e contro di questo approccio: [Elenca vantaggi e svantaggi]

PERCORSO STRATEGICO 2: Sviluppo nuove funzionalità prioritarie
- Analisi dettagliata: [Ragiona passo-passo su come identificare e sviluppare le funzionalità più impattanti]
- Valutazione costi/benefici: [Analizza l'impatto economico di questa strategia]
- Timeline di implementazione: [Stima i tempi necessari]
- Pro e contro di questo approccio: [Elenca vantaggi e svantaggi]

PERCORSO STRATEGICO 3: Strategia di engagement e customer success
- Analisi dettagliata: [Ragiona passo-passo su come migliorare il coinvolgimento dei clienti]
- Valutazione costi/benefici: [Analizza l'impatto economico di questa strategia]
- Timeline di implementazione: [Stima i tempi necessari]
- Pro e contro di questo approccio: [Elenca vantaggi e svantaggi]

CONCLUSIONE:
Confronta i tre percorsi strategici e identifica l'approccio ottimale o una combinazione di elementi dalle diverse strategie. Fornisci un piano d'azione concreto e motivato.
"""

# Otteniamo la risposta da Claude
response = client.messages.create(
    model="claude-3-7-sonnet-20250219",
    max_tokens=2500,
    messages=[
        {"role": "user", "content": tot_prompt}
    ]
)

print("=== RISPOSTA CON TREE-OF-THOUGHT ===")
print(response.content[0].text)
