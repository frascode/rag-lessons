import json
import os
from dotenv import load_dotenv
from anthropic import Anthropic

load_dotenv()
client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

"""
REACT PROMPTING EXAMPLE (Reasoning + Acting)
Implementazione che dimostra l'approccio ReAct per analisi complessa di dati CRM,
alternando ragionamento (Thought) e azioni (Action) con osservazioni (Observation).
"""

# Simulazione di dati CRM per l'esercizio


def get_mock_crm_data():
    """Simula un'API CRM reale restituendo dati di esempio"""
    mock_data = {
        "sales_data": {
            "q1_2025": {"revenue": 1250000, "deals_closed": 48, "avg_deal_size": 26042},
            "q2_2025": {"revenue": 1420000, "deals_closed": 52, "avg_deal_size": 27308},
            "q3_2025": {"revenue": 980000, "deals_closed": 35, "avg_deal_size": 28000},
            "q4_2025": {"revenue": 1680000, "deals_closed": 58, "avg_deal_size": 28966}
        },
        "customer_segments": {
            "enterprise": {"count": 18, "churn_rate": 0.05, "avg_revenue": 85000},
            "mid_market": {"count": 45, "churn_rate": 0.12, "avg_revenue": 42000},
            "small_business": {"count": 130, "churn_rate": 0.18, "avg_revenue": 15000}
        },
        "sales_funnel": {
            "leads": 850,
            "qualified_leads": 320,
            "opportunities": 180,
            "proposals": 95,
            "closed_won": 58,
            "closed_lost": 37
        },
        "sales_team": {
            "alice": {"deals": 22, "revenue": 620000, "avg_sales_cycle": 85},
            "bob": {"deals": 18, "revenue": 480000, "avg_sales_cycle": 92},
            "charlie": {"deals": 25, "revenue": 530000, "avg_sales_cycle": 78},
            "diana": {"deals": 15, "revenue": 850000, "avg_sales_cycle": 120}
        },
        "product_categories": {
            "software_licenses": {"revenue": 2800000, "margin": 0.75},
            "professional_services": {"revenue": 850000, "margin": 0.45},
            "support_maintenance": {"revenue": 1680000, "margin": 0.82}
        }
    }
    return mock_data

# Funzioni che simulano "azioni" che l'assistente può eseguire


def get_sales_data(period=None):
    """Ottiene dati di vendita per un periodo specifico o tutti"""
    data = get_mock_crm_data()["sales_data"]
    if period and period in data:
        return {period: data[period]}
    return data


def get_customer_segment_data(segment=None):
    """Ottiene dati per un segmento cliente specifico o tutti"""
    data = get_mock_crm_data()["customer_segments"]
    if segment and segment in data:
        return {segment: data[segment]}
    return data


def get_sales_funnel_data():
    """Ottiene dati del funnel di vendita"""
    return get_mock_crm_data()["sales_funnel"]


def get_sales_team_performance(member=None):
    """Ottiene performance del team vendite, generale o per specifico membro"""
    data = get_mock_crm_data()["sales_team"]
    if member and member in data:
        return {member: data[member]}
    return data


def get_product_category_data(category=None):
    """Ottiene dati per una categoria prodotto specifica o tutte"""
    data = get_mock_crm_data()["product_categories"]
    if category and category in data:
        return {category: data[category]}
    return data


def calculate_kpi(metric_name):
    """Calcola KPI specifici basati sui dati disponibili"""
    data = get_mock_crm_data()

    if metric_name == "conversion_rate":
        # Lead to deal conversion rate
        return data["sales_funnel"]["closed_won"] / data["sales_funnel"]["leads"] * 100

    elif metric_name == "avg_deal_value":
        # Average deal value across all quarters
        total_revenue = sum(q["revenue"] for q in data["sales_data"].values())
        total_deals = sum(q["deals_closed"]
                          for q in data["sales_data"].values())
        return total_revenue / total_deals

    elif metric_name == "revenue_per_employee":
        # Revenue per employee
        total_revenue = sum(q["revenue"] for q in data["sales_data"].values())
        total_employees = len(data["sales_team"])
        return total_revenue / total_employees

    elif metric_name == "overall_churn_rate":
        # Weighted churn rate across segments
        segments = data["customer_segments"]
        total_customers = sum(s["count"] for s in segments.values())
        weighted_churn = sum(s["count"] * s["churn_rate"]
                             for s in segments.values())
        return weighted_churn / total_customers * 100

    else:
        return f"Unknown KPI: {metric_name}"


# Prompt ReAct per analisi CRM
react_prompt = """
Sei un analista CRM esperto che utilizza l'approccio Reasoning + Acting (ReAct) per affrontare problemi complessi.

RICHIESTA:
Interpreta la richiesta utente è

Per rispondere, utilizza l'approccio ReAct:
1. Thought: rifletti sul problema e su quali dati sono necessari
2. Action: indica quale azione specifica intendi compiere tra:
   - get_sales_data(period=optional)
   - get_customer_segment_data(segment=optional)
   - get_sales_funnel_data()
   - get_sales_team_performance(member=optional)
   - get_product_category_data(category=optional)
   - calculate_kpi(metric_name)
   - generate_chart(chart_type, data_source)

Alla fine, fornisci l'azione da intraprendere. Specificando solo il nome dell'azione.
"""

# Esecuzione del ciclo ReAct


def execute_react_cycle(action_name, **params):
    """Esegue l'azione specificata con i parametri forniti"""
    if action_name == "get_sales_data":
        return get_sales_data(**params)
   # elif action_name == "get_customer_segment_data":
   #     return get_customer_segment_data(**params)
   # elif action_name == "get_sales_funnel_data":
   #     return get_sales_funnel_data()
   # elif action_name == "get_sales_team_performance":
   #     return get_sales_team_performance(**params)
   # elif action_name == "get_product_category_data":
   #     return get_product_category_data(**params)
   # elif action_name == "calculate_kpi":
   #     metric_name = params.get("metric_name")
   #     return calculate_kpi(metric_name)
   # elif action_name == "generate_chart":
   #     return generate_chart(params.get("chart_type"), params.get("data_source"))
   # else:
   #     return f"Error: Unknown action '{action_name}'"

# Simulazione di un'interazione ReAct con Claude


def run_react_analysis():
    print("=== ANALISI CRM CON APPROCCIO REACT ===\n")

    # Iniziamo con il prompt iniziale
    print("PROMPT INIZIALE:")
    print(react_prompt)
    print("\n" + "="*80 + "\n")

    action = client.messages.create(
        model="claude-3-7-sonnet-20250219",
        max_tokens=300,
    
        messages=[
            {"role": "user", "content": react_prompt},
            {"role": "user", "content": "Per rispondere alla richiesta della direzione commerciale, devo prima ottenere una panoramica completa dei dati di vendita per capire il trend generale e il contesto. Quale azione devo utilizzare? Rispondi solo col nome della funziona senza ulteriori dati"}
        ]
    )

    # Simuliamo alcune iterazioni (in un'app reale, questo sarebbe un ciclo guidato dalle risposte del modello)
    reactions = [
        # {
        #     "thought": "Per rispondere alla richiesta della direzione commerciale, devo prima ottenere una panoramica completa dei dati di vendita per capire il trend generale e il contesto.",
        #     "action": "get_sales_data",
        #     "params": {}
        # },
        # {
        #     "thought": "Ora che ho i dati di vendita per quarti, vedo che il Q4 ha avuto la performance migliore. Devo esaminare i segmenti di clienti per identificare quelli più profittevoli e a rischio.",
        #     "action": "get_customer_segment_data",
        #     "params": {}
        # },
        # {
        #     "thought": "I segmenti Enterprise hanno il churn rate più basso e il valore più alto. Vediamo ora come si comporta il funnel di vendita per capire dove possiamo migliorare la conversione.",
        #     "action": "get_sales_funnel_data",
        #     "params": {}
        # },
        # {
        #     "thought": "Noto un calo significativo tra leads qualificati e opportunità. Vediamo anche le performance del team vendite per capire come allocare al meglio le risorse.",
        #     "action": "get_sales_team_performance",
        #     "params": {}
        # },
        # {
        #     "thought": "Diana ha un valore medio delle vendite molto più alto ma meno deal totali, mentre Charlie chiude più deal ma con valore inferiore. Vediamo anche quali categorie di prodotti generano più margine.",
        #     "action": "get_product_category_data",
        #     "params": {}
        # },
        # {
        #     "thought": "Support & Maintenance ha il margine più alto, seguito da Software Licenses. Calcolo alcuni KPI aggiuntivi per completare l'analisi.",
        #     "action": "calculate_kpi",
        #     "params": {"metric_name": "conversion_rate"}
        # },
        # {
        #     "thought": "Il tasso di conversione totale è solo del 6.8%. Vediamo anche il valore medio dei deal.",
        #     "action": "calculate_kpi",
        #     "params": {"metric_name": "avg_deal_value"}
        # },
        # {
        #     "thought": "Per visualizzare meglio il funnel di vendita e identificare i colli di bottiglia, genero un grafico del funnel.",
        #     "action": "generate_chart",
        #     "params": {"chart_type": "funnel", "data_source": "sales_funnel"}
        # }
    ]

    # Eseguiamo le azioni e mostriamo i risultati
    # for i, reaction in enumerate(reactions):
    #     print(f"ITERAZIONE {i+1}:")
    #     print(f"Thought: {reaction['thought']}")
    #     print(
    #         f"Action: {reaction['action']}({', '.join([f'{k}={v}' for k, v in reaction['params'].items()])})")
# 
    #     # Esegui l'azione e mostra il risultato
    #     result = execute_react_cycle(reaction['action'], **reaction['params'])
    #     print(f"Observation: {json.dumps(result, indent=2)}")
    #     print("\n" + "-"*80 + "\n")
    
    print(action.content[0].text)
    result = execute_react_cycle(action)
    print(result)

    # Conclusione con Claude
#    conclusion_prompt = """
#    Basandoti sul processo ReAct e sulle osservazioni raccolte, sintetizza le tue scoperte e fornisci raccomandazioni
#    concrete per la direzione commerciale, rispondendo alle 4 domande iniziali:
#    1. Quali segmenti di clienti sono più profittevoli e quali a più alto rischio
#    2. Come ottimizzare il funnel di vendita per aumentare il tasso di conversione
#    3. Come riallocare le risorse del team vendite per massimizzare la revenue
#    4. Quale mix di prodotti/servizi promuovere per massimizzare i margini
#    #"""

#    print("RICHIESTA DI CONCLUSIONE:")
#    print(conclusion_prompt)
#    print("\n" + "="*80 + "\n")

    # In un'app reale, qui invieremmo la richiesta a Claude
    # Simuliamo una risposta di conclusione per scopi dimostrativi
    #print("CONCLUSIONE DELL'ANALISI REACT:")
    #conclusion = """
    ## Analisi delle Performance di Vendita e Raccomandazioni
    #
    #Basandomi sull'analisi completa dei dati CRM, ho identificato diverse opportunità di miglioramento:
    #
    ### 1. Segmenti di clienti: profittabilità e rischio
    #
    #- **Segmento Enterprise**: Il più profittevole (€85K di revenue media) con il minor rischio (churn rate 5%)
    #- **Segmento Mid-Market**: Buona revenue (€42K) ma churn preoccupante (12%)
    #- **Small Business**: Alto churn (18%) e valore più basso (€15K)
    #
    #**Raccomandazione**: Investire nell'acquisizione di clienti Enterprise e in strategie di retention per Mid-Market.
    #
    ### 2. Ottimizzazione del funnel di vendita
    #
    #Il tasso di conversione complessivo è solo del 6.8%, con due colli di bottiglia principali:
    #- Da leads qualificati (320) a opportunità (180): -44%
    #- Da proposte (95) a closed won (58): -39%
    #
    #**Raccomandazione**: Migliorare il processo di qualificazione per identificare meglio le opportunità valide e rivedere l'approccio alle proposte per aumentare il tasso di chiusura.
    #
    ### 3. Riallocazione risorse team vendite
    #
    #- **Diana**: Chiude meno deal (15) ma con valore molto alto (€850K totale, media €56.7K)
    #- **Charlie**: Più deal (25) ma valore inferiore (€530K totale, media €21.2K)
    #
    #**Raccomandazione**: Specializzare Diana sul segmento Enterprise e assegnare a Charlie i clienti Small Business. Fornire coaching a Bob che ha i cicli di vendita più lunghi.
    #
    ### 4. Mix prodotti/servizi ottimale
    #
    #- **Support & Maintenance**: Margine eccellente (82%)
    #- **Software Licenses**: Buon margine (75%) e volume alto (€2.8M)
    #- **Professional Services**: Margine inferiore (45%)
    #
    #**Raccomandazione**: Incentivare il cross-selling di Support & Maintenance su tutte le vendite di Software. Limitare le personalizzazioni nei Professional Services per migliorare i margini.
    #
    ### Piano d'azione prioritario
    #
    #1. Riorganizzare il team vendite in base alla specializzazione per segmento
    #2. Implementare un nuovo processo di qualificazione leads più rigoroso
    #3. Lanciare pacchetti Support & Maintenance premium per clienti Enterprise
    #4. Rivedere il processo di proposta commerciale con template ottimizzati
    #"""

    #print(conclusion)


if __name__ == "__main__":
    run_react_analysis()
