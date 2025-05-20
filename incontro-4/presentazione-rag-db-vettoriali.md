# DB Vettoriali e RAG Avanzato
### Corso RAG per applicazioni ERP e CRM
#### Incontro 4 - [20/05/2025]

---

## Agenda dell'incontro
- Benvenuto e riepilogo incontri precedenti
- Fondamenti dei Database Vettoriali
- Retrieval Augmented Generation (RAG)
- RAG Avanzato
- **Pausa (15 min)**
- Graph RAG
- Demo integrazione RAG con sistemi ERP/CRM
- Esercitazioni pratiche
- Casi d'uso in contesti aziendali
- Conclusione e compiti

---

## Obiettivi dell'incontro
- Comprendere i principi fondamentali dei DB vettoriali
- Padroneggiare l'architettura RAG di base
- Implementare tecniche RAG avanzate
- Esplorare le potenzialità di Graph RAG
- Applicare RAG a dati ERP/CRM reali
- Sviluppare una pipeline RAG completa

---

## Riepilogo Incontri Precedenti

### Concetti chiave affrontati
- Fondamenti LLM e architettura Transformer
- Principi di Prompt Engineering
- Tecniche di prompting: zero-shot, one-shot, few-shot
- Chain-of-Thought e strategie avanzate
- Implementazione di prompt con contesto

---

## Database Vettoriali

### Che cosa sono i DB vettoriali?
- **Definizione**: database ottimizzati per memorizzare embedding (rappresentazioni vettoriali)
- **Scopo**: ricerca semantica e di similarità
- **Differenza dai DB tradizionali**: focus su "significato" vs exact match
- **Applicazioni**: ricerca documentale, raccomandazioni, clustering

---

## Database Vettoriali

### Concetti fondamentali
- **Embedding**: rappresentazione numerica di contenuti in spazio vettoriale
- **Dimensionalità**: tipicamente 768-1536 dimensioni per testi
- **Distanza vettoriale**: misura di similarità (coseno, euclidea, ecc.)
- **Indici vettoriali**: strutture per ottimizzare ricerca (HNSW, IVF, ecc.)
- **ANN (Approximate Nearest Neighbor)**: compromesso velocità/precisione

---

## Database Vettoriali

### Principali soluzioni sul mercato
- **Pinecone**: completamente gestito, facile integrazione
- **Weaviate**: graph-based, ibrido vettoriale-semantico
- **Milvus**: open-source, alta scalabilità
- **Chroma**: leggero, dedicato a RAG
- **Qdrant**: flessibile, ricco di funzionalità
- **PostgreSQL + pgvector**: estensione SQL tradizionale

---

## Database Vettoriali

### Confronto prestazioni/caratteristiche
| Database | Tipo | Scalabilità | Funzionalità | Complessità |
|---------|------|------------|--------------|------------|
| Pinecone | Cloud | Alta | Media | Bassa |
| Weaviate | Cloud/Self-hosted | Alta | Alta | Media |
| Chroma | Self-hosted | Media | Media | Bassa |
| PostgreSQL+pgvector | Self-hosted | Media | Alta | Media |
| Qdrant | Cloud/Self-hosted | Alta | Alta | Media |

---

## Database Vettoriali

### Esempio di query vettoriale (Python)
```python
# Creazione embedding dalla query
query = "Come gestire lead inattivi nel CRM"
query_embedding = embedding_model.encode(query)

# Ricerca nei documenti più simili
results = vector_db.search(
    vector=query_embedding,
    namespace="crm_documentation",
    top_k=5
)

# Utilizzo dei risultati
for result in results:
    print(f"Documento: {result.id}")
    print(f"Score similarità: {result.score}")
    print(f"Contenuto: {result.metadata['text'][:100]}...")
```

---

## Retrieval Augmented Generation (RAG)

### Cos'è RAG?
- **Definizione**: tecnica che combina recupero di informazioni e generazione LLM
- **Paper originale**: "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks" (2020)
- **Vantaggio principale**: supera le limitazioni di conoscenza degli LLM
- **Applicazioni**: Q&A, assistenti specializzati, knowledge management

---

## Retrieval Augmented Generation (RAG)

### Architettura RAG base
1. **Indicizzazione**:
   - Raccolta documenti
   - Chunking (suddivisione)
   - Creazione embedding
   - Memorizzazione in DB vettoriale

2. **Query**:
   - Embedding della query
   - Retrieval documenti rilevanti
   - Incorporazione in prompt (context)
   - Generazione risposta con LLM

---

## Retrieval Augmented Generation (RAG)

### Pipeline RAG: Indicizzazione
```python
# 1. Caricamento documenti
documents = load_documents("./crm_docs/")

# 2. Chunking (suddivisione in parti gestibili)
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
chunks = text_splitter.split_documents(documents)

# 3. Creazione embedding
embeddings = OpenAIEmbeddings()

# 4. Memorizzazione in DB vettoriale
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db"
)
```

---

## Retrieval Augmented Generation (RAG)

### Pipeline RAG: Query
```python
# 1. Configurazione retriever
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 4}
)

# 2. Creazione prompt template
template = """
Sei un assistente CRM esperto. Usa il seguente contesto 
per rispondere alla domanda dell'utente.

Contesto:
{context}

Domanda: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

# 3. Setup del modello LLM
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

# 4. Creazione pipeline RAG
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
)

# 5. Esecuzione query
response = rag_chain.invoke("Come posso migliorare il lead scoring?")
```

---

## RAG Avanzato

### Limiti del RAG base
- Retrieval impreciso
- Context window limitata 
- Allucinazioni residue
- Mancanza di reasoning
- Risposta non strutturata

---

## RAG Avanzato

### Tecniche di chunking avanzate
- **Semantic chunking**: divisione basata su significato
- **Overlap ottimizzato**: mantenimento contesto tra chunk
- **Chunking gerarchico**: documenti → sezioni → paragrafi
- **Metadati arricchiti**: aggiunta informazioni strutturali

```python
hierarchical_splitter = HierarchicalTextSplitter(
    chunk_sizes=[2000, 1000, 500],
    separators=["\n## ", "\n### ", "\n"],
    keep_separator=True
)
```

---

## RAG Avanzato

### Hybrid Search
- Combinazione di multiple strategie di ricerca:
  - **Keyword search**: parole chiave (BM25, SPLADE)
  - **Dense retrieval**: embedding (cosine similarity)
  - **Reranking**: riordinamento risultati per rilevanza

```python
# Hybrid retriever
hybrid_retriever = MultiSearchRetriever(
    retrievers=[
        bm25_retriever,
        vector_retriever,
        neural_retriever
    ],
    weights=[0.3, 0.4, 0.3]
)
```

---

## RAG Avanzato

### Multi-query RAG
- Generazione automatica di query multiple per una domanda
- Diversificazione intent e prospettive
- Aggregazione risultati di retrieval
- Deduplicazione contenuti recuperati

```python
# Query expansion con LLM
def generate_queries(question, n=3):
    system = "Genera {n} diverse riformulazioni della domanda per migliorare il retrieval."
    response = llm.invoke([
        SystemMessage(content=system),
        HumanMessage(content=question)
    ])
    return parse_queries(response.content)
```

---

## RAG Avanzato

### Retrieval-centric evaluation
- Metriche di qualità del retrieval:
  - **Relevance**: pertinenza dei documenti
  - **Precision@K**: % documenti rilevanti tra i primi K
  - **Recall**: % documenti rilevanti recuperati
  - **Faithfulness**: aderenza della risposta ai documenti

```python
# Esempio framework evaluation
def evaluate_retrieval(retriever, dataset, metrics=["relevance", "precision"]):
    results = []
    for query in dataset:
        retrieved_docs = retriever.get_relevant_documents(query.question)
        metrics_result = calculate_metrics(
            retrieved_docs, 
            query.ground_truth,
            metrics
        )
        results.append(metrics_result)
    return aggregate_results(results)
```

---

## PAUSA (15 minuti)

---

## Graph RAG

### Cosa sono i Knowledge Graph?
- **Definizione**: rappresentazione della conoscenza come grafo
- **Componenti**: nodi (entità), archi (relazioni)
- **Vantaggi**: rappresentazione strutturata, contestualizzazione, connessioni implicite
- **Applicazioni in ERP/CRM**: modellazione clienti, prodotti, processi, eventi

---

## Graph RAG

### Da RAG tradizionale a Graph RAG
- **RAG tradizionale**: documenti come unità atomiche
- **Graph RAG**: entità e relazioni come unità di conoscenza
- **Differenza fondamentale**: knowledge retrieval vs document retrieval
- **Vantaggi**: risposta più precisa, navigazione rapida tra concetti collegati

---

## Graph RAG

### Architettura Graph RAG
1. **Knowledge Graph Construction**:
   - Estrazione entità e relazioni dai documenti
   - Normalizzazione e linking
   - Strutturazione grafo

2. **Vector-enhanced Graph**:
   - Embedding di nodi e relazioni
   - Indici vettoriali per similarity search
   
3. **Query processor**:
   - Entity linking sulla query
   - Path finding e subgraph retrieval
   - Incorporazione contesto strutturato

---

## Graph RAG

### Implementazione con Weaviate o Neo4j
```python
# Creazione grafo della conoscenza
graph = KnowledgeGraph()

# Estrazione entità e relazioni 
for document in documents:
    entities = extract_entities(document.content)
    relations = extract_relations(document.content)
    
    for entity in entities:
        graph.add_node(entity.id, entity.attributes)
    
    for relation in relations:
        graph.add_edge(
            relation.source, 
            relation.target,
            relation.type
        )

# Query sul grafo
results = graph.query("""
    MATCH (c:Customer)-[r:PURCHASED]->(p:Product)
    WHERE p.category = 'CRM'
    RETURN c, p, r
    LIMIT 5
""")
```

---

## Graph RAG

### Subgraph retrieval
```python
def retrieve_subgraph(query, kg, depth=2):
    # 1. Identificare entità nella query
    entities = entity_extractor.extract(query)
    
    # 2. Mappare entità ai nodi del grafo
    seed_nodes = kg.find_nodes(entities)
    
    # 3. Espandere grafo da nodi iniziali
    subgraph = kg.expand_from_nodes(
        seed_nodes,
        max_depth=depth,
        max_nodes=50
    )
    
    # 4. Serializzare sottografo per LLM
    return subgraph.to_text_representation()
```

---

## Demo integrazione RAG con sistemi ERP/CRM

### Caso d'uso: Assistente commerciale intelligente
- **Setup**:
  - Dataset: documentazione prodotti, storico clienti, knowledge base
  - Embedding: modello personalizzato settoriale
  - LLM: Claude 3.5 Sonnet

- **Features**:
  - Q&A sui prodotti specifici
  - Suggerimento cross-selling basato su storico
  - Troubleshooting guidato
  - Generazione proposte personalizzate

---

## Demo integrazione RAG con sistemi ERP/CRM

### Architettura della soluzione
```
┌───────────┐    ┌──────────────┐    ┌─────────────┐
│  ERP/CRM  │───>│  Data Loader │───>│ Processor   │
│  System   │    └──────────────┘    │ Embedder    │
└───────────┘                        └─────────────┘
                                           │
┌───────────┐    ┌──────────────┐    ┌─────▼─────┐
│ Web/Chat  │<───│  RAG Chain   │<───│ Vector DB │
│ Interface │    │  LLM         │    │           │
└───────────┘    └──────────────┘    └───────────┘
```

---

## Demo integrazione RAG con sistemi ERP/CRM

### Integrazione con SAP
```python
# Connessione a SAP
sap_client = SapClient(
    ashost='sap.example.com',
    sysnr='00',
    client='100',
    user=SAP_USER,
    passwd=SAP_PASSWORD
)

# Estrazione dati cliente
def extract_customer_data(customer_id):
    customer_basic = sap_client.call(
        'BAPI_CUSTOMER_GETDETAIL',
        CustomerID=customer_id
    )
    
    sales_history = sap_client.call(
        'BAPI_SALESORDER_GETLIST',
        CustomerID=customer_id
    )
    
    return {
        "basic": customer_basic,
        "sales": sales_history
    }

# Incorporazione dati nel RAG
def enrich_prompt_with_sap_data(question, customer_id):
    customer_data = extract_customer_data(customer_id)
    return f"""
    Contesto cliente SAP:
    {format_customer_data(customer_data)}
    
    Domanda: {question}
    """
```

---

## Esercitazioni pratiche

### Esercizio 1: Pipeline RAG base
- Creare embedding da documentazione ERP/CRM
- Implementare retriever con Chroma DB
- Sviluppare prompt template contestualizzato
- Testare con query ambigue

### Esercizio 2: Ottimizzazione retrieval
- Confrontare chunking strategies
- Implementare hybrid search
- Misurare precision e recall
- Iterare per migliorare risultati

---

## Esercitazioni pratiche

### Esercizio 3: Graph RAG con dati CRM
- Estrarre struttura entità-relazioni da CRM:
  - Clienti → Acquisti → Prodotti
  - Utenti → Interazioni → Ticket
- Costruire knowledge graph minimale
- Implementare query su relazioni
- Confrontare risultati con RAG tradizionale

---

## Casi d'uso in contesti aziendali

### CRM: Assistente commerciale intelligente
- **Use case**: supporto in tempo reale durante chiamate clienti
- **Dati utilizzati**: storico interazioni, prodotti, prezzi, offerte
- **Valore aggiunto**: up-selling mirato, risposta rapida, personalizzazione
- **Misure successo**: tasso conversione, tempi risposta, NPS

---

## Casi d'uso in contesti aziendali

### ERP: Troubleshooting automatizzato
- **Use case**: diagnosi e risoluzione problemi implementazione
- **Dati utilizzati**: log errori, documentazione tecnica, knowledge base
- **Valore aggiunto**: risoluzione più rapida, minor carico supporto
- **Misure successo**: MTTR (tempo medio risoluzione), % autocorrezione

---

## Casi d'uso in contesti aziendali

### ERP/CRM: Analisi documentale intelligente
- **Use case**: estrazione informazioni da documenti non strutturati
- **Dati utilizzati**: contratti, email, note, documenti tecnici
- **Valore aggiunto**: automazione input dati, riduzione errori manuali
- **Misure successo**: accuratezza estrazione, tempo risparmiato

---

## Conclusione

### Riepilogo concetti chiave
- I DB vettoriali abilitano ricerca semantica
- RAG supera limitazioni di conoscenza degli LLM
- Tecniche avanzate migliorano precisione e contesto
- Graph RAG sfrutta relazioni strutturate
- Integrazione con ERP/CRM apre scenari ad alto valore

---

## Conclusione

### Compiti per il prossimo incontro
- Implementare pipeline RAG completa su dataset proprio
- Sperimentare tecniche di chunking avanzate
- Documentare metriche di performance e iterazioni
- Preparare domande su problemi specifici incontrati

---

## Risorse aggiuntive
- [LangChain RAG Documentation](https://python.langchain.com/docs/use_cases/question_answering/)
- [LlamaIndex Docs - Advanced RAG](https://docs.llamaindex.ai/)
- [Weaviate - Knowledge Graph & Vector DB](https://weaviate.io/developers/weaviate)
- "Building RAG Applications with LLMs" (paper)
- [Neo4j Vector Search Documentation](https://neo4j.com/docs/vector-search/current/)
- [Pinecone Learning Center - RAG](https://www.pinecone.io/learn/retrieval-augmented-generation/)

---

## Domande?

---

## Grazie dell'attenzione!
### Prossimo incontro: Personalizzazione ed Evaluation RAG
