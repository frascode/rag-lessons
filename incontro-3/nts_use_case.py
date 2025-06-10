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



"""
Entities:
  Partenze:
    attrs: [IDPartenza(int,PK), DataPartenza(date), IDCliente(int,FK), IDDestinazione(int,FK), IDVettore(int,FK), Quantità_Prenotata(float), Avv(bit), TBC(bit), OK(bit), Euro(bit), StatoPartenza(int), Stampato(bit), Quantità_Consegnata(float), Annotazioni_Partenze(varchar)]
    rels: 
      - belongs_to: CLIENTI_ANAGRAFICA (IDCliente->IDCliente_Anagrafica)
      - belongs_to: Clienti_Destinazioni (IDDestinazione->IDDestinazione)
      - belongs_to: Vettori (IDVettore->IDVettore)

  CLIENTI_ANAGRAFICA:
    attrs: [IDCliente_Anagrafica(int,PK), IDCategoria(int,FK), CodiceArca(varchar), Intestazione(varchar), PrefissoPartitaIVA(varchar), PartitaIva(varchar), CodiceFiscale(varchar), CodiceDestinatario(varchar), Pec(varchar), NazioneSedeLegale(varchar), IndirizzoSedeLegale(varchar), ProvinciaSedeLegale(varchar), ComuneSedeLegale(varchar), CAPSedeLegale(varchar), Tel1(varchar), Cel1(varchar), Fax1(varchar), Skype1(varchar), Email(varchar), SitoWeb(varchar), NoteFatture(varchar), NoteDDT(varchar), Annotazioni(varchar), Predefinito(bit), LocalitàComuneEsteroSedeLegale(varchar), ZipCodeSedeLegale(varchar), Valuta(int), Fido(float), IDCodiceIva(int,FK), IDContoCorrente(int,FK), IDAgente1(int,FK), IDAgente2(int,FK), IDModalitàDiPagamento(int,FK), IDDilazioneDiPagamento(int,FK), ScontoCliente(float), GiorniDiViaggio(int), AltriDatiGestionali(bit), ScontoExtraContabile(float), PercentualeAssicurazione(float), NoteSconto(varchar), TipoTrasporto_Porto(int), UtilizzaNotificaSpedizione(bit), ConStampaArrivi(bit), UtilizzaCodiceEdiPerUnitàDiMisura(bit), TipoTrasporto_TrasportoACura(varchar), IDRiferimentoAziendale(int,FK), AbbreviazioneC(varchar), CartellaAssociata(varchar), EsenzioneIva(bit), PasswordWeb(varchar), ResoPedane(bit), PalletPerFicheDaRendere(int), TipoTrasporto_Causale(varchar)]
    rels: 
      - has_many: Partenze (IDCliente_Anagrafica<-IDCliente)
      - has_many: Clienti_Destinazioni (IDCliente_Anagrafica<-IDCliente_Anagrafica)

  Clienti_Destinazioni:
    attrs: [IDDestinazione(int,PK), IDCliente_Anagrafica(int,FK), DescrizioneSedeLogistica(varchar), NazioneSedeLogistica(varchar), LocalitàComuneEsteroSedeLogistica(varchar), IndirizzoSedeLogistica(varchar), ProvinciaSedeLogistica(varchar), ComuneSedeLogistica(varchar), CAPSedeLogistica(varchar), ZipCodeSedeLogistica(varchar), Tel(varchar), Cel(varchar), Fax(varchar), Skype(varchar), Email(varchar), CodiceUnivoco(varchar), Predefinita(bit), CodiceUnivocoDestinazione(varchar), AbbreviazioneDestinazione(varchar)]
    rels: 
      - belongs_to: CLIENTI_ANAGRAFICA (IDCliente_Anagrafica->IDCliente_Anagrafica)
      - has_many: Partenze (IDDestinazione<-IDDestinazione)
    
  Vettori:
    attrs: [IDVettore(int,PK), Tipo(int), Intestazione(varchar), PrefissoPartitaIVA(varchar), PartitaIva(varchar), CodiceFiscale(varchar), IscrizioneAlbo(varchar), NazioneSedeLegale(varchar), LocalitàComuneEsteroSedeLegale(varchar), IndirizzoSedeLegale(varchar), ProvinciaSedeLegale(varchar), ComuneSedeLegale(varchar), CAPSedeLegale(varchar), ZipCodeSedeLegale(varchar), Tel1(varchar), Cel1(varchar), Fax1(varchar), Skype1(varchar), Email(varchar), SitoWeb(varchar), Annotazioni(varchar), AbbreviazioneV(varchar)]
    rels: 
      - has_many: Partenze (IDVettore<-IDVettore)






"""