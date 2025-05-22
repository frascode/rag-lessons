import os
from dotenv import load_dotenv
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import create_retrieval_chain
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain import hub

# Caricamento variabili d'ambiente
load_dotenv()

# Configurazione API keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Funzione per creare un retriever RAG base


def create_basic_rag(docs_dir, persist_dir="./chroma_db"):
    """
    Crea un retriever RAG base utilizzando documenti da una directory.
    Effettua il chunking, embedding e memorizzazione in un DB vettoriale.

    Args:
        docs_dir: Directory contenente i documenti
        persist_dir: Directory dove persistere il DB vettoriale

    Returns:
        retriever: Componente di retrieval configurato
    """
    print("Caricamento documenti...")
    # Carica documenti dalla directory (supporta .txt, .pdf, .docx, etc.)
    loader = DirectoryLoader(docs_dir)
    documents = loader.load()
    print(f"Caricati {len(documents)} documenti")

    print("Suddivisione in chunks...")
    # Suddivide i documenti in chunks più piccoli
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # Dimensione di ogni chunk
        chunk_overlap=200,  # Sovrapposizione per mantenere contesto
        separators=["\n\n", "\n", " ", ""],  # Priorità di separazione
        length_function=len  # Funzione per calcolare lunghezza
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Creati {len(chunks)} chunks")

    print("Creazione embeddings e memorizzazione nel DB vettoriale...")
    # Crea embeddings e memorizza in Chroma DB
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_dir
    )

    # Crea e configura il retriever
    retriever = vectorstore.as_retriever(
        search_type="similarity",  # Strategia: similarity, mmr
        search_kwargs={"k": 1}     # Numero di documenti da recuperare
    )
    print("Retriever RAG configurato con successo!")

    return retriever, chunks

# Funzione per creare un retriever ibrido


def create_hybrid_retriever(chunks, vectorstore, weights=[0.5, 0.5]):
    """
    Crea un retriever ibrido che combina ricerca vettoriale e BM25.

    Args:
        chunks: I documenti suddivisi in chunks
        vectorstore: Il database vettoriale
        weights: Pesi per ciascun retriever [bm25_weight, vector_weight]

    Returns:
        hybrid_retriever: Retriever ibrido configurato
    """
    print("Configurazione retriever ibrido...")

    # Creazione retriever vettoriale
    vector_retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}
    )

    # Creazione retriever BM25 (keyword-based)
    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = 4

    # Creazione retriever ibrido (ensemble)
    hybrid_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=weights
    )

    print("Retriever ibrido configurato con successo!")
    return hybrid_retriever

# Funzione per creare una chain RAG completa


def create_rag_chain(retriever, system_prompt=None):
    """
    Crea una catena RAG completa che combina retriever e LLM.

    Args:
        retriever: Componente di retrieval già configurato
        system_prompt: Prompt di sistema personalizzato

    Returns:
        qa_chain: Chain RAG configurata
    """
    # Prompt template predefinito se non fornito
    if system_prompt is None:
        system_prompt = """
        Sei un assistente esperto in sistemi ERP e CRM. 
        Utilizza SOLO le informazioni fornite nel CONTESTO per rispondere alla domanda dell'utente.
        Se l'informazione non è presente nel contesto, ammetti di non sapere invece di inventare.
        
        
        Il tuo contesto sarà formato da due file:
            - file1.txt: Rappresenta la pagina wikipedia dei CRM.
            - file2.txt: Rappresenta il modello per l'entità cliente nel nostro ecosistema.
            
        
        Quando ti vengono chieste delle informazioni inerenti a "com'è il cliente" devi fare un focus su file2.txt
        
        CONTESTO:
        {context}
        
        DOMANDA: {question}
        
        RISPOSTA:
        """

    # Configurazione del prompt
    prompt = ChatPromptTemplate.from_template(system_prompt)

    # Configurazione del modello LLM
    llm = ChatOpenAI(
        model="gpt-3.5-turbo",  # Modello da utilizzare
        temperature=0         # Bassa temperatura per risposte più deterministiche
    )

    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")

    combine_docs_chain = create_stuff_documents_chain(
        llm, retrieval_qa_chat_prompt
    )

    retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)

    return retrieval_chain

# Funzione per eseguire query sulla chain RAG


def query_rag(qa_chain, question):
    """
    Esegue una query sulla chain RAG.

    Args:
        qa_chain: Chain RAG configurata
        question: Domanda da porre

    Returns:
        response: Risposta generata
    """
    print(f"\nDomanda: {question}")
    print("Elaborazione in corso...")

    response = qa_chain.invoke({'input': question})

    print("\nContesto usato per generare la riposta: ")
    
    for doc in response['context']:
        print(f"Content -> {doc.page_content}")

    print("\nRisposta:")
    print(response['answer'])

    return response

# Esempio di utilizzo


def main():
    # Directory contenente i documenti
    DOCS_DIR = "./docs/crm"

    # Crea retriever RAG base
    basic_retriever, chunks = create_basic_rag(DOCS_DIR)

    # # Crea retriever ibrido
    # vectorstore = Chroma(
    #     persist_directory="./chroma_db",
    #     embedding_function=OpenAIEmbeddings()
    # )
    # hybrid_retriever = create_hybrid_retriever(chunks, vectorstore, weights=[0.3, 0.7])

    # Crea chain RAG per ERP/CRM
    # system_prompt = """
    # Sei un consulente esperto in sistemi ERP e CRM con esperienza in SAP, Oracle e Microsoft Dynamics.
    # Rispondi alle domande dell'utente in modo preciso e professionale, basandoti ESCLUSIVAMENTE
    # sulle informazioni contenute nel contesto fornito.
    #
    # Se le informazioni nel contesto non sono sufficienti, indica chiaramente ciò che non puoi determinare,
    # ma proponi comunque approcci ragionevoli basati sulle migliori pratiche del settore.
    #
    # CONTESTO:
    # {context}
    #
    # DOMANDA: {question}
    #
    # RISPOSTA:
    # """

    qa_chain = create_rag_chain(basic_retriever)

    # Esempi di domande ERP/CRM
    questions = [
        "Quali sono le migliori pratiche per l'implementazione di un CRM in un'azienda di medie dimensioni?",
        # "Come posso migliorare il processo di lead scoring nel nostro sistema CRM?",
        # "Quali sono le differenze principali tra SAP e Microsoft Dynamics per la gestione inventario?",
        # "Come dovrei strutturare un piano di migrazione da un ERP legacy a un sistema cloud?",
        "Quanti tipi di 'customer' posso avere?",
        "Come si implementa Redis?",
    ]

    # Test con alcune domande
    for question in questions:
        query_rag(qa_chain, question)
        print("\n" + "-"*80 + "\n")
        
    print(chunks)


# Esecuzione
if __name__ == "__main__":
    main()
