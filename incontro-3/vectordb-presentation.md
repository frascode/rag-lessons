# VectorDB
### Corso RAG per applicazioni ERP e CRM

---

## Agenda dell'incontro
- Benvenuto e riepilogo incontro precedente
- Introduzione ai VectorDB: cosa sono e perché sono fondamentali
- Rappresentazione vettoriale: embeddings e spazi vettoriali
- Principali VectorDB sul mercato
- Architettura e funzionamento interno
- Metriche di similarità e algoritmi di ricerca
- Implementazione pratica con Pinecone e ChromaDB
- Integrazione con LLM per sistemi RAG
- Ottimizzazione e best practices
- Caso d'uso: VectorDB per knowledge base ERP/CRM
- Esercitazioni pratiche
- Conclusione e compiti

---

## Obiettivi dell'incontro
- Comprendere il ruolo dei VectorDB nei sistemi RAG
- Padroneggiare i concetti di embedding e similarità vettoriale
- Implementare e configurare un VectorDB
- Integrare VectorDB con LLM per retrieval efficace
- Ottimizzare performance e costi
- Applicare VectorDB a scenari ERP/CRM reali

---

## Riepilogo Incontro Precedente

### Concetti chiave affrontati
- Principi del Prompt Engineering
- Componenti di un prompt efficace
- Tecniche zero-shot, one-shot, few-shot
- Chain-of-Thought e ReAct prompting
- Strategie avanzate di prompting
- Prompt specifici per ERP/CRM

---

## Introduzione ai VectorDB

### Il problema della ricerca tradizionale
- **Ricerca keyword-based**: limitata a match esatti
- **Mancanza di comprensione semantica**: "automobile" ≠ "macchina"
- **Scalabilità**: milioni di documenti, performance degradanti
- **Contesto perso**: difficoltà nel catturare relazioni complesse

---

## Introduzione ai VectorDB

### Cos'è un VectorDB?
- **Database specializzato** per memorizzare e cercare vettori ad alta dimensionalità
- **Ricerca per similarità** invece che per match esatto
- **Ottimizzato** per operazioni vettoriali su larga scala
- **Fondamentale** per sistemi RAG e AI generativa

---

## Introduzione ai VectorDB

### Perché sono cruciali per RAG?
```
Documento → Embedding → Vector → Storage in VectorDB
Query → Embedding → Vector → Similarity Search → Documenti rilevanti
```

- Permettono retrieval semantico efficiente
- Scalano a milioni/miliardi di documenti
- Mantengono contesto e relazioni semantiche
- Abilitano ricerca real-time

---

## Rappresentazione Vettoriale

### Cosa sono gli embeddings?
- **Definizione**: rappresentazione numerica densa di testo/dati
- **Dimensionalità**: tipicamente 384-1536 dimensioni
- **Proprietà**: testi simili → vettori vicini nello spazio
- **Modelli**: OpenAI, Sentence Transformers, Cohere, etc.

---

## Rappresentazione Vettoriale

### Esempio pratico di embedding
```python
# Testo originale
text1 = "Il modulo CRM gestisce i contatti clienti"
text2 = "Sistema per amministrare relazioni con i customer"
text3 = "Report finanziario trimestrale"

# Dopo embedding (semplificato)
vector1 = [0.2, -0.5, 0.8, 0.1, ...]  # 1536 dimensioni
vector2 = [0.3, -0.4, 0.7, 0.2, ...]  # Simile a vector1
vector3 = [-0.6, 0.2, -0.3, 0.9, ...] # Molto diverso
```

---

## Rappresentazione Vettoriale

### Modelli di embedding popolari
- **OpenAI text-embedding-ada-002**: 1536 dimensioni, generale
- **text-embedding-3-small/large**: nuovi modelli OpenAI
- **Sentence-BERT**: open source, multilingue
- **Cohere embed-v3**: ottimizzato per search
- **Custom embeddings**: fine-tuned su dominio specifico

---

## Principali VectorDB sul Mercato

### Pinecone
- **Cloud-native**: fully managed, serverless
- **Pro**: semplicità, scalabilità automatica
- **Contro**: costo, vendor lock-in
- **Use case**: produzione, alta disponibilità

```python
import pinecone
pinecone.init(api_key="xxx")
index = pinecone.Index("erp-knowledge")
```

---

## Principali VectorDB sul Mercato

### ChromaDB
- **Open source**: self-hosted o cloud
- **Pro**: gratuito, flessibile, developer-friendly
- **Contro**: gestione infrastruttura
- **Use case**: prototipazione, sviluppo

```python
import chromadb
client = chromadb.Client()
collection = client.create_collection("crm-docs")
```

---

## Principali VectorDB sul Mercato

### Altri player importanti
- **Weaviate**: GraphQL API, moduli AI integrati
- **Qdrant**: Rust-based, alta performance
- **Milvus**: scalabilità enterprise, GPU support
- **FAISS**: libreria Facebook, non è un DB completo
- **pgvector**: estensione PostgreSQL

---

## Architettura e Funzionamento

### Componenti principali
```
┌─────────────────┐     ┌──────────────┐
│   Embedding     │────▶│    Index     │
│     Model       │     │  Structure   │
└─────────────────┘     └──────────────┘
         │                      │
         ▼                      ▼
┌─────────────────┐     ┌──────────────┐
│    Vectors      │────▶│   Storage    │
│  (dimensioni)   │     │   Engine     │
└─────────────────┘     └──────────────┘
```

---

## Architettura e Funzionamento

### Processo di indicizzazione
1. **Ingestion**: documenti/testi in input
2. **Chunking**: divisione in segmenti gestibili
3. **Embedding**: conversione in vettori
4. **Indexing**: organizzazione per ricerca efficiente
5. **Storage**: persistenza con metadati

---

## Architettura e Funzionamento

### Strutture dati per indicizzazione
- **Flat Index**: ricerca esaustiva, precisa ma lenta
- **IVF (Inverted File)**: clustering per velocizzare
- **HNSW**: grafi gerarchici, bilanciamento speed/accuracy
- **LSH**: hashing per similarità approssimata
- **Product Quantization**: compressione vettori

---

## Metriche di Similarità

### Distanza Euclidea
- Distanza geometrica nello spazio
- Intuitiva ma sensibile alla scala
- Formula: √Σ(ai - bi)²

```python
def euclidean_distance(vec1, vec2):
    return np.sqrt(np.sum((vec1 - vec2) ** 2))
```

---

## Metriche di Similarità

### Cosine Similarity
- Misura l'angolo tra vettori
- Indipendente dalla magnitudine
- Ideale per testo/embeddings
- Range: [-1, 1] dove 1 = identici

```python
def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm_product = np.linalg.norm(vec1) * np.linalg.norm(vec2)
    return dot_product / norm_product
```

---

## Metriche di Similarità

### Dot Product
- Prodotto scalare tra vettori
- Veloce da calcolare
- Assume vettori normalizzati
- Utilizzato da molti modelli

### Manhattan Distance
- Somma differenze assolute
- "Distanza del taxi"
- Robusta agli outlier

---

## Implementazione con Pinecone

### Setup iniziale
```python
import pinecone
from openai import OpenAI

# Inizializzazione
pinecone.init(
    api_key="YOUR_PINECONE_API_KEY",
    environment="us-west1-gcp"
)

# Creazione index
pinecone.create_index(
    name="erp-knowledge-base",
    dimension=1536,  # OpenAI embeddings
    metric="cosine",
    pods=1,
    pod_type="p1.x1"
)
```

---

## Implementazione con Pinecone

### Inserimento documenti
```python
import openai

def create_embeddings(texts):
    client = OpenAI()
    response = client.embeddings.create(
        model="text-embedding-ada-002",
        input=texts
    )
    return [item.embedding for item in response.data]

# Preparazione dati
documents = [
    {"id": "doc1", "text": "Configurazione modulo HR in SAP", 
     "metadata": {"module": "HR", "type": "config"}},
    {"id": "doc2", "text": "Integrazione CRM con sistema email",
     "metadata": {"module": "CRM", "type": "integration"}}
]

# Embedding e upsert
embeddings = create_embeddings([doc["text"] for doc in documents])
index = pinecone.Index("erp-knowledge-base")

for doc, emb in zip(documents, embeddings):
    index.upsert([(doc["id"], emb, doc["metadata"])])
```

---

## Implementazione con Pinecone

### Query e retrieval
```python
def semantic_search(query, top_k=5, filter=None):
    # Embedding della query
    query_embedding = create_embeddings([query])[0]
    
    # Ricerca
    results = index.query(
        vector=query_embedding,
        top_k=top_k,
        filter=filter,  # es: {"module": "CRM"}
        include_metadata=True
    )
    
    return results.matches

# Esempio di utilizzo
results = semantic_search(
    "Come configurare permessi utente?",
    filter={"module": "HR"}
)

for match in results:
    print(f"Score: {match.score:.3f}")
    print(f"Document: {match.metadata}")
```

---

## Implementazione con ChromaDB

### Setup e configurazione
```python
import chromadb
from chromadb.config import Settings

# Client locale
client = chromadb.Client()

# Client persistente
client = chromadb.PersistentClient(path="./chroma_db")

# Creazione collection
collection = client.create_collection(
    name="crm_documents",
    metadata={"hnsw:space": "cosine"}
)
```

---

## Implementazione con ChromaDB

### Gestione documenti
```python
# Aggiunta documenti con embedding automatico
collection.add(
    documents=[
        "Procedura backup database CRM",
        "Configurazione API REST per integrazioni",
        "Gestione ruoli e permessi utente"
    ],
    metadatas=[
        {"type": "procedure", "module": "CRM"},
        {"type": "technical", "module": "API"},
        {"type": "security", "module": "AUTH"}
    ],
    ids=["doc1", "doc2", "doc3"]
)

# Query con filtri
results = collection.query(
    query_texts=["come fare backup dei dati?"],
    n_results=3,
    where={"module": "CRM"}
)
```

---

## Integrazione con LLM

### Pattern RAG completo
```python
def rag_pipeline(user_query, context_docs=5):
    # 1. Retrieval
    relevant_docs = semantic_search(user_query, top_k=context_docs)
    
    # 2. Context building
    context = "\n\n".join([
        f"[{doc.metadata['type']}] {doc.metadata['text']}" 
        for doc in relevant_docs
    ])
    
    # 3. Augmented generation
    prompt = f"""
    Contesto dai documenti aziendali:
    {context}
    
    Domanda utente: {user_query}
    
    Rispondi basandoti esclusivamente sul contesto fornito.
    """
    
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )
    
    return response.choices[0].message.content
```

---

## Ottimizzazione e Best Practices

### Chunking strategies
```python
def smart_chunking(text, chunk_size=500, overlap=50):
    """Chunking con overlap per mantenere contesto"""
    chunks = []
    sentences = text.split('. ')
    
    current_chunk = []
    current_size = 0
    
    for sentence in sentences:
        sentence_size = len(sentence)
        
        if current_size + sentence_size > chunk_size:
            chunks.append('. '.join(current_chunk))
            # Overlap: mantieni ultime frasi
            current_chunk = current_chunk[-2:]
            current_size = sum(len(s) for s in current_chunk)
        
        current_chunk.append(sentence)
        current_size += sentence_size
    
    return chunks
```

---

## Ottimizzazione e Best Practices

### Metadata design
```python
# Metadata ricchi per filtering efficace
metadata = {
    "doc_id": "ERP-2024-001",
    "module": "finance",
    "sub_module": "accounts_payable",
    "doc_type": "procedure",
    "language": "it",
    "version": "2.1",
    "last_updated": "2024-01-15",
    "access_level": "internal",
    "tags": ["invoice", "payment", "workflow"],
    "relevance_score": 0.95
}

# Query con filtri multipli
results = index.query(
    vector=query_embedding,
    filter={
        "$and": [
            {"module": {"$eq": "finance"}},
            {"access_level": {"$in": ["public", "internal"]}},
            {"version": {"$gte": "2.0"}}
        ]
    }
)
```

---

## Ottimizzazione e Best Practices

### Performance tuning
1. **Batch operations**: inserimenti/query in gruppo
2. **Dimensionalità**: bilanciare precisione vs performance
3. **Sharding**: distribuzione dati per scalabilità
4. **Caching**: memorizzare query frequenti
5. **Hybrid search**: combinare vector + keyword search

```python
# Batch upsert esempio
def batch_upsert(documents, batch_size=100):
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i+batch_size]
        embeddings = create_embeddings([doc["text"] for doc in batch])
        
        vectors = [
            (doc["id"], emb, doc["metadata"]) 
            for doc, emb in zip(batch, embeddings)
        ]
        
        index.upsert(vectors=vectors)
```

---

## Caso d'uso: Knowledge Base ERP/CRM

### Architettura sistema RAG per ERP
```
┌─────────────────┐     ┌─────────────────┐
│  Documentazione │     │   Ticket/Case   │
│   Tecnica ERP   │     │   Support CRM   │
└────────┬────────┘     └────────┬────────┘
         │                       │
         ▼                       ▼
┌─────────────────────────────────────────┐
│          Preprocessing Pipeline          │
│  (Chunking, Cleaning, Enrichment)       │
└────────────────┬─────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────┐
│            Vector Database              │
│  (Embeddings + Metadata strutturati)    │
└────────────────┬─────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────┐
│         RAG Application Layer           │
│    (Query → Retrieve → Generate)        │
└─────────────────────────────────────────┘
```

---

## Caso d'uso: Knowledge Base ERP/CRM

### Implementazione pratica
```python
class ERPKnowledgeBase:
    def __init__(self, vector_db_name="erp-kb"):
        self.index = pinecone.Index(vector_db_name)
        self.openai = OpenAI()
        
    def index_document(self, doc_path, doc_type, module):
        """Indicizza documento ERP con metadata ricchi"""
        # Leggi e processa documento
        with open(doc_path, 'r') as f:
            content = f.read()
        
        # Chunking intelligente
        chunks = self.smart_chunk(content)
        
        # Crea embeddings e metadata
        for i, chunk in enumerate(chunks):
            chunk_id = f"{doc_path}#chunk{i}"
            embedding = self.create_embedding(chunk)
            
            metadata = {
                "source": doc_path,
                "chunk_index": i,
                "doc_type": doc_type,
                "module": module,
                "content": chunk[:200],  # preview
                "timestamp": datetime.now().isoformat()
            }
            
            self.index.upsert([(chunk_id, embedding, metadata)])
```

---

## Caso d'uso: Knowledge Base ERP/CRM

### Query intelligenti con context
```python
def answer_erp_question(self, question, user_role="user"):
    """Risponde a domande ERP con contesto appropriato"""
    
    # Determina moduli rilevanti dalla domanda
    relevant_modules = self.detect_modules(question)
    
    # Costruisci filtro basato su ruolo utente
    filter_query = {
        "$and": [
            {"module": {"$in": relevant_modules}},
            {"access_level": {"$in": self.get_user_permissions(user_role)}}
        ]
    }
    
    # Retrieval con re-ranking
    initial_results = self.index.query(
        vector=self.create_embedding(question),
        top_k=20,
        filter=filter_query
    )
    
    # Re-rank basato su relevance + recency
    ranked_docs = self.rerank_results(initial_results, question)
    
    # Genera risposta
    context = self.build_context(ranked_docs[:5])
    response = self.generate_answer(question, context)
    
    return response, ranked_docs[:5]  # risposta + fonti
```

---

## Esercitazioni Pratiche

### Esercizio 1: Setup VectorDB locale
1. Installare ChromaDB
2. Creare collection per documenti CRM
3. Implementare funzioni CRUD base
4. Testare ricerca semantica

### Esercizio 2: Pipeline di indicizzazione
1. Caricare documentazione ERP (PDF/TXT)
2. Implementare chunking strategy
3. Generare embeddings
4. Strutturare metadata efficaci

---

## Esercitazioni Pratiche

### Esercizio 3: RAG system completo
```python
# Template per esercizio
class SimpleRAG:
    def __init__(self):
        # TODO: Inizializza ChromaDB
        # TODO: Inizializza OpenAI client
        pass
    
    def add_knowledge(self, documents):
        # TODO: Processa e indicizza documenti
        pass
    
    def query(self, question):
        # TODO: Implementa retrieval
        # TODO: Genera risposta con context
        pass

# Test con casi reali ERP/CRM
rag = SimpleRAG()
rag.add_knowledge([
    "Il modulo HR gestisce anagrafiche dipendenti...",
    "Per configurare il workflow fatturazione..."
])

response = rag.query("Come aggiungo un nuovo dipendente?")
```

---

## Esercitazioni Pratiche

### Esercizio 4: Ottimizzazione e monitoring
1. Implementare metriche di quality
2. A/B testing diverse strategie
3. Monitoring performance
4. Analisi costi

```python
def evaluate_retrieval_quality(queries_test_set):
    """Valuta qualità del retrieval"""
    metrics = {
        "precision_at_k": [],
        "recall": [],
        "avg_similarity": []
    }
    
    for query, expected_docs in queries_test_set:
        results = vector_search(query)
        # Calcola metriche...
    
    return metrics
```

---

## Best Practices per Produzione

### Gestione del ciclo di vita
- **Versioning**: tracciare versioni embeddings/modelli
- **Backup**: strategie di disaster recovery
- **Monitoring**: metriche performance e qualità
- **Updates**: aggiornamento incrementale documenti

### Security considerations
- **Access control**: chi può query/modificare
- **Data privacy**: encryption at rest/in transit
- **Audit logging**: tracciabilità operazioni
- **Compliance**: GDPR, data residency

---

## Troubleshooting Comune

### Problemi frequenti e soluzioni

1. **"Risultati non pertinenti"**
   - Verifica qualità embeddings
   - Ottimizza chunking strategy
   - Enrichment metadata

2. **"Performance degradate"**
   - Index optimization
   - Batch operations
   - Caching layer

3. **"Costi elevati"**
   - Dimensionalità embeddings
   - Filtering pre-search
   - Hybrid approach

---

## Integrazione Avanzata

### Multi-modal search
```python
# Ricerca che combina testo + metadata strutturati
def hybrid_search(text_query, filters, weights={"text": 0.7, "meta": 0.3}):
    # Vector search
    text_results = vector_search(text_query)
    
    # Metadata filtering
    meta_results = metadata_search(filters)
    
    # Combine and rerank
    combined = merge_results(text_results, meta_results, weights)
    return combined
```

### Cross-lingual search
- Embeddings multilingue
- Query expansion
- Translation layer

---

## Conclusione

### Riepilogo concetti chiave
- VectorDB abilitano ricerca semantica scalabile
- Embeddings catturano significato, non solo keywords
- Scelta del DB dipende da use case specifico
- Metadata e chunking strategy sono critici
- Integrazione con LLM crea sistemi RAG potenti

---

## Conclusione

### Compiti per il prossimo incontro
1. Implementare un VectorDB per knowledge base aziendale
2. Sperimentare con diverse strategie di chunking
3. Confrontare performance Pinecone vs ChromaDB
4. Creare pipeline RAG per un caso d'uso specifico
5. Documentare metriche e lesson learned

---

## Risorse Aggiuntive

### Documentazione ufficiale
- [Pinecone Docs](https://docs.pinecone.io/)
- [ChromaDB Docs](https://docs.trychroma.com/)
- [Weaviate Docs](https://weaviate.io/developers/weaviate)
- [LangChain VectorStores](https://python.langchain.com/docs/modules/data_connection/vectorstores/)

### Papers e approfondimenti
- "Efficient and robust approximate nearest neighbor search"
- "Dense Passage Retrieval for Open-Domain QA"
- "REALM: Retrieval-Augmented Language Model Pre-Training"

### Tools e utilities
- [Embedding Projector](https://projector.tensorflow.org/)
- [Vector Database Benchmark](https://github.com/erikbern/ann-benchmarks)

---

## Domande?

### Contatti per supporto
- Email corso: corso-rag@example.com
- Slack channel: #vectordb-help
- Office hours: Giovedì 15:00-17:00

---

## Grazie dell'attenzione!
### Prossimo incontro: Chiamate a funzione e Agenti AI