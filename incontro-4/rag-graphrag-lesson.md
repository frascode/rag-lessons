# RAG e GraphRAG
### Corso RAG per applicazioni ERP e CRM - Lezione Finale

---

## Agenda dell'incontro
- Benvenuto e riepilogo corso
- Deep dive su architetture RAG
- Componenti e pipeline RAG avanzate
- Strategie di chunking e retrieval
- Introduzione a GraphRAG
- NanoGraphRAG: implementazione pratica
- Hybrid RAG: combinare approcci
- Valutazione e metriche RAG
- Deployment e produzione
- Case study completo ERP/CRM
- Laboratorio pratico intensivo

---

## Obiettivi della lezione
- Padroneggiare architetture RAG complete
- Implementare pipeline RAG production-ready
- Comprendere e applicare GraphRAG
- Ottimizzare performance e costi
- Deployare sistemi RAG scalabili
- Costruire un RAG completo per ERP/CRM

---

## Riepilogo del Corso

### Il percorso completato
1. **Lezione 1**: Fondamenti LLM e modelli
2. **Lezione 2**: Prompt Engineering avanzato
3. **Lezione 3**: VectorDB e ricerca semantica
4. **Lezione 4**: RAG e GraphRAG (oggi)

### Competenze acquisite
- Comprensione profonda degli LLM
- Tecniche di prompting efficaci
- Gestione database vettoriali
- **Oggi**: orchestrazione completa RAG

---

## Deep Dive su RAG

### Cos'è realmente RAG?
**R**etrieval **A**ugmented **G**eneration

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│   Query     │────▶│   Retrieval  │────▶│  Augmented  │
│   Input     │     │   System     │     │ Generation  │
└─────────────┘     └──────────────┘     └─────────────┘
                            │
                            ▼
                    ┌──────────────┐
                    │  Knowledge   │
                    │     Base     │
                    └──────────────┘
```

- **Supera** i limiti di conoscenza degli LLM
- **Riduce** allucinazioni
- **Permette** aggiornamenti real-time
- **Abilita** personalizzazione dominio-specifica

---

## Architettura RAG Moderna

### Componenti essenziali
```python
class ModernRAG:
    def __init__(self):
        self.document_processor = DocumentProcessor()
        self.embedder = EmbeddingModel()
        self.vector_store = VectorDatabase()
        self.retriever = HybridRetriever()
        self.reranker = CrossEncoderReranker()
        self.generator = LLMGenerator()
        self.evaluator = RAGEvaluator()
```

1. **Ingestion Pipeline**: processamento documenti
2. **Embedding Layer**: conversione semantica
3. **Retrieval System**: ricerca multi-modale
4. **Augmentation Engine**: contesto optimization
5. **Generation Module**: risposta finale
6. **Feedback Loop**: continuous improvement

---

## Pipeline RAG Avanzata

### 1. Document Processing
```python
class AdvancedDocumentProcessor:
    def __init__(self):
        self.parsers = {
            'pdf': PDFParser(),
            'docx': DocxParser(),
            'html': HTMLParser(),
            'csv': CSVParser(),
            'json': JSONParser()
        }
        self.chunker = SmartChunker()
        self.enricher = MetadataEnricher()
    
    def process_document(self, doc_path, doc_type='auto'):
        # Auto-detect tipo documento
        if doc_type == 'auto':
            doc_type = self.detect_type(doc_path)
        
        # Parse documento
        raw_content = self.parsers[doc_type].parse(doc_path)
        
        # Pulizia e normalizzazione
        clean_content = self.clean_text(raw_content)
        
        # Chunking intelligente
        chunks = self.chunker.chunk(
            clean_content,
            method='semantic',  # semantic, sliding_window, recursive
            chunk_size=512,
            overlap=50
        )
        
        # Arricchimento metadata
        enriched_chunks = self.enricher.enrich(chunks, doc_path)
        
        return enriched_chunks
```

---

## Pipeline RAG Avanzata

### 2. Smart Chunking Strategies
```python
class SmartChunker:
    def semantic_chunk(self, text, max_size=512):
        """Chunking basato su boundaries semantici"""
        sentences = self.sentence_splitter.split(text)
        chunks = []
        current_chunk = []
        current_size = 0
        
        for sentence in sentences:
            # Calcola similarità semantica con chunk corrente
            if current_chunk:
                similarity = self.calculate_coherence(current_chunk, sentence)
                
                # Se bassa coerenza o size limit, nuovo chunk
                if similarity < 0.7 or current_size + len(sentence) > max_size:
                    chunks.append({
                        'text': ' '.join(current_chunk),
                        'coherence_score': self.chunk_coherence(current_chunk)
                    })
                    current_chunk = [sentence]
                    current_size = len(sentence)
                    continue
            
            current_chunk.append(sentence)
            current_size += len(sentence)
        
        return chunks
    
    def hierarchical_chunk(self, document):
        """Chunking multi-livello per documenti strutturati"""
        return {
            'document': self.create_summary(document),
            'sections': [self.chunk_section(s) for s in document.sections],
            'paragraphs': [self.chunk_paragraph(p) for p in document.paragraphs],
            'sentences': [self.chunk_sentence(s) for s in document.sentences]
        }
```

---

## Retrieval Strategies

### Multi-Stage Retrieval
```python
class MultiStageRetriever:
    def __init__(self, vector_store, bm25_index):
        self.vector_store = vector_store
        self.bm25 = bm25_index
        self.cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        
    def retrieve(self, query, top_k=10):
        # Stage 1: Broad retrieval (Vector + BM25)
        vector_results = self.vector_store.search(query, top_k=50)
        bm25_results = self.bm25.search(query, top_k=50)
        
        # Stage 2: Fusion dei risultati
        fused_results = self.reciprocal_rank_fusion(
            vector_results, 
            bm25_results,
            k=60  # RRF parameter
        )
        
        # Stage 3: Re-ranking con Cross-Encoder
        reranked = self.cross_encoder.rerank(
            query=query,
            documents=fused_results[:30],
            top_k=top_k
        )
        
        # Stage 4: Diversity sampling
        diverse_results = self.mmr_sampling(reranked, lambda_param=0.7)
        
        return diverse_results
```

---

## Retrieval Strategies

### Query Expansion e Reformulation
```python
class QueryOptimizer:
    def __init__(self, llm_client):
        self.llm = llm_client
        
    def expand_query(self, original_query):
        """Espande query con sinonimi e termini correlati"""
        prompt = f"""
        Query originale: {original_query}
        
        Genera:
        1. 3 riformulazioni alternative
        2. Termini chiave correlati
        3. Domande specifiche implicite
        
        Output in JSON.
        """
        
        expansion = self.llm.generate(prompt)
        return json.loads(expansion)
    
    def hypothetical_document_embedding(self, query):
        """HyDE: genera documento ipotetico per miglior retrieval"""
        prompt = f"""
        Domanda: {query}
        
        Scrivi un paragrafo di risposta ideale che conterrebbe
        le informazioni necessarie per rispondere a questa domanda.
        Focus su fatti e dettagli specifici.
        """
        
        hypothetical_doc = self.llm.generate(prompt)
        return self.embedder.embed(hypothetical_doc)
```

---

## Augmentation Strategies

### Context Window Optimization
```python
class ContextOptimizer:
    def __init__(self, max_context_length=4000):
        self.max_length = max_context_length
        self.relevance_scorer = RelevanceScorer()
        
    def optimize_context(self, retrieved_docs, query):
        """Ottimizza contesto per massima rilevanza"""
        
        # Score documenti per rilevanza
        scored_docs = [
            {
                'doc': doc,
                'relevance': self.relevance_scorer.score(query, doc),
                'length': len(doc['text']),
                'position_weight': 1.0 / (idx + 1)  # decay posizionale
            }
            for idx, doc in enumerate(retrieved_docs)
        ]
        
        # Knapsack optimization per massimizzare valore
        selected_docs = self.knapsack_selection(
            scored_docs, 
            self.max_length
        )
        
        # Riordina per coerenza narrativa
        ordered_docs = self.narrative_ordering(selected_docs)
        
        # Comprimi se necessario
        if self.total_length(ordered_docs) > self.max_length:
            ordered_docs = self.compress_documents(ordered_docs)
            
        return ordered_docs
```

---

## Generation Strategies

### Advanced Prompting per RAG
```python
class RAGPromptBuilder:
    def build_prompt(self, query, context, metadata=None):
        """Costruisce prompt ottimizzato per RAG"""
        
        # Sistema di citation
        cited_context = self.add_citations(context)
        
        prompt = f"""<context>
{cited_context}
</context>

<instructions>
Sei un assistente esperto per sistemi ERP/CRM. 
Rispondi alla domanda basandoti ESCLUSIVAMENTE sul contesto fornito.
Se l'informazione non è presente nel contesto, dichiaralo esplicitamente.
Cita le fonti usando [1], [2], etc.
</instructions>

<query>{query}</query>

<metadata>
Modulo: {metadata.get('module', 'N/A')}
Tipo richiesta: {metadata.get('request_type', 'general')}
Livello utente: {metadata.get('user_level', 'standard')}
</metadata>

Risposta:"""
        
        return prompt
    
    def add_citations(self, documents):
        """Aggiunge riferimenti per citation"""
        cited = []
        for idx, doc in enumerate(documents, 1):
            source = doc.get('metadata', {}).get('source', 'Unknown')
            cited.append(f"[{idx}] {doc['text']} (Fonte: {source})")
        return "\n\n".join(cited)
```

---

## Introduzione a GraphRAG

### Limiti del RAG tradizionale
- **Relazioni perse**: documenti trattati come isolati
- **Contesto limitato**: difficoltà con query multi-hop
- **No reasoning**: manca capacità di inferenza
- **Scalabilità**: retrieval inefficiente su larga scala

### GraphRAG: la soluzione
```
Documents → Knowledge Graph → Graph Traversal → Enhanced Context → Generation
```

- **Cattura relazioni** esplicite tra entità
- **Abilita reasoning** multi-hop
- **Migliora comprensione** del dominio
- **Riduce ridondanza** nelle informazioni

---

## GraphRAG Architecture

### Componenti chiave
```python
class GraphRAG:
    def __init__(self):
        self.graph_builder = KnowledgeGraphBuilder()
        self.entity_extractor = EntityExtractor()
        self.relation_extractor = RelationExtractor()
        self.graph_store = Neo4jStore()  # o NetworkX per prototipazione
        self.graph_retriever = GraphRetriever()
        self.path_ranker = PathRanker()
```

### Pipeline GraphRAG
1. **Entity Extraction**: identifica entità nei documenti
2. **Relation Extraction**: estrae relazioni tra entità
3. **Graph Construction**: costruisce knowledge graph
4. **Graph Indexing**: indicizza per retrieval efficiente
5. **Query Processing**: trasforma query in graph traversal
6. **Subgraph Retrieval**: estrae sottografi rilevanti
7. **Context Building**: costruisce contesto da paths

---

## NanoGraphRAG Implementation

### Setup iniziale
```python
# Installazione
# pip install nano-graphrag networkx pyvis

from nano_graphrag import GraphRAG, QueryParam
from nano_graphrag.base import BaseKVStorage
import networkx as nx

class ERPGraphRAG:
    def __init__(self, working_dir="./erp_graphrag"):
        # Configurazione base
        self.graph_rag = GraphRAG(
            working_dir=working_dir,
            enable_llm_cache=True,
            entity_extraction_config={
                "model": "gpt-4",
                "prompt_template": self.get_entity_prompt()
            },
            relation_extraction_config={
                "model": "gpt-4", 
                "prompt_template": self.get_relation_prompt()
            }
        )
        
    def get_entity_prompt(self):
        return """Extract business entities from this ERP/CRM document.
        Focus on: modules, processes, users, roles, data objects.
        Format: {"entities": [{"name": "...", "type": "...", "description": "..."}]}
        """
        
    def get_relation_prompt(self):
        return """Extract relationships between entities.
        Focus on: uses, manages, contains, depends_on, triggers.
        Format: {"relations": [{"source": "...", "target": "...", "type": "...", "properties": {}}]}
        """
```

---

## NanoGraphRAG Implementation

### Costruzione del Knowledge Graph
```python
class ERPKnowledgeGraphBuilder:
    def __init__(self, graph_rag):
        self.graph_rag = graph_rag
        self.graph = nx.DiGraph()
        
    async def build_from_documents(self, documents):
        """Costruisce graph da documenti ERP"""
        
        for doc in documents:
            # Estrai entità
            entities = await self.extract_entities(doc)
            
            # Aggiungi nodi al grafo
            for entity in entities:
                self.graph.add_node(
                    entity['name'],
                    type=entity['type'],
                    description=entity['description'],
                    source_doc=doc['id']
                )
            
            # Estrai relazioni
            relations = await self.extract_relations(doc, entities)
            
            # Aggiungi edges
            for rel in relations:
                self.graph.add_edge(
                    rel['source'],
                    rel['target'],
                    type=rel['type'],
                    weight=rel.get('weight', 1.0),
                    properties=rel.get('properties', {})
                )
        
        # Calcola metriche del grafo
        self.compute_graph_metrics()
        
        return self.graph
```

---

## NanoGraphRAG Implementation

### Query e Retrieval
```python
class GraphRetriever:
    def __init__(self, graph, embedder):
        self.graph = graph
        self.embedder = embedder
        
    async def retrieve(self, query, method='hybrid'):
        """Retrieval multi-strategia da knowledge graph"""
        
        # 1. Entity Recognition nella query
        query_entities = await self.identify_query_entities(query)
        
        # 2. Subgraph extraction
        if method == 'local':
            subgraph = self.local_neighborhood(query_entities, max_hops=2)
        elif method == 'path':
            subgraph = self.path_based_retrieval(query_entities)
        elif method == 'hybrid':
            local = self.local_neighborhood(query_entities)
            paths = self.path_based_retrieval(query_entities)
            subgraph = self.merge_subgraphs(local, paths)
        
        # 3. Rank nodes per relevance
        ranked_nodes = self.pagerank_scoring(subgraph, query_entities)
        
        # 4. Costruisci contesto narrativo
        context = self.build_narrative_context(ranked_nodes, subgraph)
        
        return context
    
    def local_neighborhood(self, entities, max_hops=2):
        """Estrae neighborhood locale delle entità"""
        subgraph = nx.Graph()
        
        for entity in entities:
            if entity in self.graph:
                # BFS fino a max_hops
                for node in nx.single_source_shortest_path_length(
                    self.graph, entity, cutoff=max_hops
                ):
                    subgraph.add_node(node, **self.graph.nodes[node])
                    
                    # Aggiungi edges
                    for neighbor in self.graph.neighbors(node):
                        if neighbor in subgraph:
                            subgraph.add_edge(
                                node, neighbor,
                                **self.graph.edges[node, neighbor]
                            )
        
        return subgraph
```

---

## Hybrid RAG Approach

### Combinare Vector e Graph RAG
```python
class HybridRAG:
    def __init__(self, vector_store, graph_store):
        self.vector_rag = VectorRAG(vector_store)
        self.graph_rag = GraphRAG(graph_store)
        self.fusion_model = FusionModel()
        
    async def query(self, user_query, strategy='adaptive'):
        """Query ibrida vector + graph"""
        
        # Analizza tipo di query
        query_type = self.analyze_query_type(user_query)
        
        if strategy == 'adaptive':
            if query_type == 'factual':
                # Usa principalmente vector search
                vector_weight, graph_weight = 0.8, 0.2
            elif query_type == 'relational':
                # Usa principalmente graph search
                vector_weight, graph_weight = 0.3, 0.7
            else:
                # Bilanciato
                vector_weight, graph_weight = 0.5, 0.5
        
        # Retrieval parallelo
        vector_context = await self.vector_rag.retrieve(user_query)
        graph_context = await self.graph_rag.retrieve(user_query)
        
        # Fusion dei contesti
        fused_context = self.fusion_model.fuse(
            vector_context, 
            graph_context,
            weights=(vector_weight, graph_weight)
        )
        
        # Genera risposta
        response = await self.generate_response(user_query, fused_context)
        
        return {
            'answer': response,
            'vector_sources': vector_context['sources'],
            'graph_paths': graph_context['paths'],
            'fusion_weights': (vector_weight, graph_weight)
        }
```

---

## Valutazione RAG

### Metriche di valutazione
```python
class RAGEvaluator:
    def __init__(self):
        self.metrics = {
            'retrieval': RetrievalMetrics(),
            'generation': GenerationMetrics(),
            'end_to_end': EndToEndMetrics()
        }
    
    def evaluate_retrieval(self, queries, ground_truth):
        """Valuta qualità del retrieval"""
        results = {
            'precision_at_k': [],
            'recall_at_k': [],
            'mrr': [],  # Mean Reciprocal Rank
            'ndcg': []  # Normalized Discounted Cumulative Gain
        }
        
        for query, true_docs in zip(queries, ground_truth):
            retrieved = self.retriever.retrieve(query)
            
            # Calcola metriche
            results['precision_at_k'].append(
                self.precision_at_k(retrieved, true_docs, k=5)
            )
            results['recall_at_k'].append(
                self.recall_at_k(retrieved, true_docs, k=10)
            )
            results['mrr'].append(
                self.mean_reciprocal_rank(retrieved, true_docs)
            )
            
        return {k: np.mean(v) for k, v in results.items()}
    
    def evaluate_generation(self, questions, answers, ground_truth):
        """Valuta qualità delle risposte generate"""
        return {
            'bleu': self.calculate_bleu(answers, ground_truth),
            'rouge': self.calculate_rouge(answers, ground_truth),
            'bert_score': self.calculate_bert_score(answers, ground_truth),
            'factual_accuracy': self.check_factual_accuracy(answers, ground_truth),
            'hallucination_rate': self.detect_hallucinations(answers)
        }
```

---

## Metriche RAG Avanzate

### RAGAS Framework
```python
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
    context_relevancy,
    answer_similarity,
    answer_correctness
)

class RAGASEvaluator:
    def __init__(self):
        self.metrics = [
            faithfulness,        # Risposta fedele al contesto?
            answer_relevancy,    # Risposta pertinente alla domanda?
            context_precision,   # Contesto preciso?
            context_recall,      # Contesto completo?
            context_relevancy,   # Contesto rilevante?
            answer_similarity,   # Similarità con ground truth
            answer_correctness   # Correttezza fattuale
        ]
    
    async def evaluate_rag_pipeline(self, test_dataset):
        """Valutazione completa pipeline RAG"""
        results = evaluate(
            dataset=test_dataset,
            metrics=self.metrics,
            llm=self.eval_llm,
            embeddings=self.embeddings
        )
        
        # Analisi dettagliata
        analysis = {
            'overall_score': results.aggregate_score(),
            'metric_breakdown': results.to_dict(),
            'weak_points': self.identify_weaknesses(results),
            'recommendations': self.generate_recommendations(results)
        }
        
        return analysis
```

---

## Deployment e Produzione

### Architettura Production-Ready
```python
# docker-compose.yml per RAG system
"""
version: '3.8'
services:
  api:
    build: ./api
    environment:
      - VECTOR_DB_URL=http://vectordb:8000
      - GRAPH_DB_URL=bolt://neo4j:7687
      - REDIS_URL=redis://redis:6379
    depends_on:
      - vectordb
      - neo4j
      - redis
      
  vectordb:
    image: qdrant/qdrant
    volumes:
      - ./data/qdrant:/qdrant/storage
      
  neo4j:
    image: neo4j:latest
    environment:
      - NEO4J_AUTH=neo4j/password
    volumes:
      - ./data/neo4j:/data
      
  redis:
    image: redis:alpine
    command: redis-server --appendonly yes
    
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
"""

class ProductionRAG:
    def __init__(self):
        self.cache = RedisCache()
        self.rate_limiter = RateLimiter()
        self.monitor = PrometheusMonitor()
        self.load_balancer = LoadBalancer()
```

---

## Deployment e Produzione

### Scalabilità e Performance
```python
class ScalableRAGService:
    def __init__(self):
        self.connection_pool = ConnectionPool(
            min_connections=10,
            max_connections=100
        )
        self.cache_layer = MultiLevelCache()
        self.async_processor = AsyncProcessor()
        
    async def handle_request(self, query, user_context):
        """Gestione richieste con caching e async"""
        
        # Check cache
        cache_key = self.generate_cache_key(query, user_context)
        cached_result = await self.cache_layer.get(cache_key)
        
        if cached_result:
            self.monitor.record_cache_hit()
            return cached_result
            
        # Process request
        async with self.rate_limiter.acquire(user_context['user_id']):
            # Parallel retrieval
            vector_task = asyncio.create_task(
                self.vector_retriever.retrieve_async(query)
            )
            graph_task = asyncio.create_task(
                self.graph_retriever.retrieve_async(query)
            )
            
            vector_results, graph_results = await asyncio.gather(
                vector_task, graph_task
            )
            
            # Generate response
            response = await self.generate_with_timeout(
                query, vector_results, graph_results,
                timeout=30  # seconds
            )
            
            # Cache result
            await self.cache_layer.set(
                cache_key, response, 
                ttl=3600  # 1 hour
            )
            
            return response
```

---

## Case Study: ERP/CRM RAG System

### Requisiti del sistema
- **Domini**: Finance, HR, Sales, Operations
- **Utenti**: 1000+ concurrent users
- **Documenti**: 100k+ technical docs, procedures, tickets
- **Languages**: Italiano, Inglese
- **SLA**: <2s latency, 99.9% uptime

### Architettura implementata
```python
class EnterpriseERPRAG:
    def __init__(self):
        # Multi-tenant configuration
        self.tenant_manager = TenantManager()
        
        # Domain-specific models
        self.domain_models = {
            'finance': FinanceRAG(),
            'hr': HumanResourcesRAG(),
            'sales': SalesRAG(),
            'operations': OperationsRAG()
        }
        
        # Security layer
        self.auth_manager = AuthenticationManager()
        self.permission_checker = PermissionChecker()
        
        # Monitoring
        self.analytics = AnalyticsEngine()
```

---

## Case Study: Implementation Details

### 1. Document Processing Pipeline
```python
class ERPDocumentPipeline:
    def __init__(self):
        self.ocr_engine = TesseractOCR()  # Per PDF scansionati
        self.table_extractor = TableExtractor()  # Per dati strutturati
        self.multilingual_processor = MultilingualProcessor()
        
    async def process_erp_document(self, document):
        """Pipeline completa per documenti ERP"""
        
        # 1. Estrazione contenuto
        if document.is_scanned:
            content = await self.ocr_engine.extract(document)
        else:
            content = await self.extract_native(document)
            
        # 2. Estrazione tabelle e dati strutturati
        tables = await self.table_extractor.extract_tables(document)
        
        # 3. Processamento multilingue
        detected_lang = self.detect_language(content)
        if detected_lang != 'en':
            content_en = await self.translate(content, to='en')
            # Mantieni entrambe le versioni
            
        # 4. Chunking specializzato per tipo
        if document.type == 'procedure':
            chunks = self.chunk_procedure(content)
        elif document.type == 'technical_spec':
            chunks = self.chunk_technical(content)
        else:
            chunks = self.smart_chunk(content)
            
        # 5. Metadata enrichment
        enriched = self.enrich_with_erp_metadata(chunks, document)
        
        return enriched
```

---

## Case Study: Query Processing

### 2. Intelligent Query Router
```python
class ERPQueryRouter:
    def __init__(self):
        self.intent_classifier = IntentClassifier()
        self.entity_recognizer = ERPEntityRecognizer()
        self.access_controller = AccessController()
        
    async def route_query(self, query, user_context):
        """Routing intelligente basato su intent e permessi"""
        
        # 1. Classificazione intent
        intent = await self.intent_classifier.classify(query)
        # Esempi: 'configuration', 'troubleshooting', 'reporting', 'integration'
        
        # 2. Estrazione entità ERP
        entities = await self.entity_recognizer.extract(query)
        # Esempi: modules, transactions, workflows, users
        
        # 3. Check permessi utente
        allowed_modules = self.access_controller.get_allowed_modules(
            user_context['role'],
            user_context['department']
        )
        
        # 4. Filtra risultati per accesso
        search_filters = {
            'modules': {'$in': allowed_modules},
            'access_level': {'$lte': user_context['clearance_level']},
            'language': user_context.get('preferred_language', 'it')
        }
        
        # 5. Selezione strategia retrieval
        if intent == 'troubleshooting':
            strategy = 'graph_first'  # Priorità a relazioni
        elif intent == 'configuration':
            strategy = 'vector_first'  # Priorità a documenti
        else:
            strategy = 'hybrid'
            
        return {
            'intent': intent,
            'entities': entities,
            'filters': search_filters,
            'strategy': strategy
        }
```

---

## Laboratorio Pratico - Parte 1

### Esercizio 1: Setup completo RAG System
```python
# Template progetto
"""
erp_rag_project/
├── config/
│   ├── settings.yaml
│   └── prompts/
├── src/
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── document_processor.py
│   │   └── chunking_strategies.py
│   ├── retrieval/
│   │   ├── vector_retriever.py
│   │   ├── graph_retriever.py
│   │   └── hybrid_retriever.py
│   ├── generation/
│   │   ├── prompt_builder.py
│   │   └── response_generator.py
│   └── evaluation/
│       └── metrics.py
├── data/
│   ├── documents/
│   └── test_sets/
└── notebooks/
    └── experiments.ipynb
"""

# Implementazione base
class ERPRAGSystem:
    def __init__(self, config_path="config/settings.yaml"):
        self.config = self.load_config(config_path)
        self.setup_components()
        
    def setup_components(self):
        # Vector store
        self.vector_store = ChromaDB(
            collection_name=self.config['vector_db']['collection'],
            embedding_model=self.config['embedding']['model']
        )
        
        # Graph store
        self.graph_store = Neo4j(
            uri=self.config['graph_db']['uri'],
            auth=(self.config['graph_db']['user'], 
                  self.config['graph_db']['password'])
        )
        
        # LLM
        self.llm = OpenAI(
            model=self.config['llm']['model'],
            temperature=self.config['llm']['temperature']
        )
```

---

## Laboratorio Pratico - Parte 1

### Esercizio 2: Implementare GraphRAG con NanoGraphRAG
```python
import asyncio
from nano_graphrag import GraphRAG, QueryParam
from nano_graphrag._utils import wrap_embedding_func
import numpy as np

class ERPNanoGraphRAG:
    def __init__(self):
        # Custom embedding function per italiano
        self.embedding_func = wrap_embedding_func(
            self.create_multilingual_embeddings
        )
        
        # Inizializza GraphRAG
        self.graph_rag = GraphRAG(
            working_dir="./erp_graph_rag_data",
            embedding_func=self.embedding_func,
            llm_model_func=self.custom_llm_func,
            llm_model_name="gpt-4",
            llm_model_max_token_size=8192,
            llm_model_max_async=4,
            enable_llm_cache=True
        )
    
    def create_multilingual_embeddings(self, texts):
        """Embeddings ottimizzati per italiano/inglese"""
        # Usa sentence-transformers per multilingue
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
        embeddings = model.encode(texts)
        return embeddings.tolist()
    
    async def index_erp_documents(self, documents):
        """Indicizza documenti ERP nel graph"""
        # Prepara documenti
        formatted_docs = []
        for doc in documents:
            formatted_docs.append({
                'content': doc['text'],
                'metadata': {
                    'source': doc['source'],
                    'module': doc['module'],
                    'doc_type': doc['type'],
                    'language': doc.get('language', 'it')
                }
            })
        
        # Inserisci nel GraphRAG
        await self.graph_rag.insert(formatted_docs)
        
        print(f"Indicizzati {len(documents)} documenti")
        return True
```

---

## Laboratorio Pratico - Parte 1

### Esercizio 3: Query Avanzate con GraphRAG
```python
class AdvancedGraphQuery:
    def __init__(self, graph_rag):
        self.graph_rag = graph_rag
        
    async def query_with_context(self, question, user_context):
        """Query GraphRAG con contesto utente"""
        
        # Prepara parametri query
        query_params = QueryParam(
            mode="hybrid",  # local, global, o hybrid
            only_need_context=False,
            response_type="comprehensive",
            top_k=10,
            max_tokens=2000
        )
        
        # Aggiungi filtri basati su contesto utente
        if user_context.get('department'):
            query_params.filter = {
                'module': user_context['department']
            }
        
        # Esegui query
        result = await self.graph_rag.aquery(
            question,
            param=query_params
        )
        
        # Post-process risultato
        enhanced_result = self.enhance_with_metadata(result)
        
        return enhanced_result
    
    def enhance_with_metadata(self, result):
        """Arricchisce risultato con metadata e spiegazioni"""
        return {
            'answer': result,
            'confidence': self.calculate_confidence(result),
            'sources': self.extract_sources(result),
            'related_topics': self.find_related_topics(result),
            'visualization': self.create_graph_visualization(result)
        }
    
    def create_graph_visualization(self, result):
        """Crea visualizzazione del subgraph utilizzato"""
        import pyvis
        from pyvis.network import Network
        
        net = Network(notebook=True, height="500px", width="100%")
        
        # Aggiungi nodi e edges dal risultato
        # (implementazione specifica per il risultato GraphRAG)
        
        return net.generate_html()
```

---

## Laboratorio Pratico - Parte 2

### Esercizio 4: Sistema RAG Completo per ERP
```python
# main.py - Sistema completo
import asyncio
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

app = FastAPI(title="ERP RAG System")

# Models
class Query(BaseModel):
    question: str
    user_id: str
    department: str
    language: str = "it"

class RAGResponse(BaseModel):
    answer: str
    sources: list
    confidence: float
    processing_time: float

# Sistema RAG globale
rag_system = None

@app.on_event("startup")
async def startup_event():
    global rag_system
    rag_system = await initialize_rag_system()

async def initialize_rag_system():
    """Inizializza sistema RAG completo"""
    system = HybridERPRAG()
    
    # Carica documenti iniziali
    await system.load_initial_documents()
    
    # Costruisci knowledge graph
    await system.build_knowledge_graph()
    
    # Riscalda cache
    await system.warm_up_cache()
    
    return system

@app.post("/query", response_model=RAGResponse)
async def query_endpoint(query: Query):
    """Endpoint principale per query RAG"""
    try:
        start_time = time.time()
        
        # Autenticazione e autorizzazione
        user_permissions = await verify_user_permissions(query.user_id)
        
        # Processo query
        result = await rag_system.process_query(
            question=query.question,
            user_context={
                'user_id': query.user_id,
                'department': query.department,
                'language': query.language,
                'permissions': user_permissions
            }
        )
        
        # Calcola tempo di elaborazione
        processing_time = time.time() - start_time
        
        return RAGResponse(
            answer=result['answer'],
            sources=result['sources'],
            confidence=result['confidence'],
            processing_time=processing_time
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "vector_db": await rag_system.check_vector_db(),
        "graph_db": await rag_system.check_graph_db(),
        "cache": await rag_system.check_cache()
    }
```

---

## Laboratorio Pratico - Parte 2

### Esercizio 5: Testing e Valutazione
```python
# test_rag_system.py
import pytest
import asyncio
from dataclasses import dataclass
from typing import List, Dict

@dataclass
class TestCase:
    question: str
    expected_modules: List[str]
    expected_entities: List[str]
    min_confidence: float
    
class RAGSystemTester:
    def __init__(self, rag_system):
        self.rag = rag_system
        self.test_cases = self.load_test_cases()
        
    def load_test_cases(self):
        """Carica test cases per ERP/CRM"""
        return [
            TestCase(
                question="Come configurare i permessi utente nel modulo HR?",
                expected_modules=["HR", "Security"],
                expected_entities=["permessi", "utente", "ruolo"],
                min_confidence=0.8
            ),
            TestCase(
                question="Qual è il processo di approvazione fatture?",
                expected_modules=["Finance", "Workflow"],
                expected_entities=["fattura", "approvazione", "workflow"],
                min_confidence=0.85
            ),
            TestCase(
                question="Come integrare il CRM con Outlook?",
                expected_modules=["CRM", "Integration"],
                expected_entities=["Outlook", "email", "sincronizzazione"],
                min_confidence=0.75
            )
        ]
    
    async def run_comprehensive_tests(self):
        """Esegue suite completa di test"""
        results = {
            'retrieval_accuracy': [],
            'response_quality': [],
            'performance': [],
            'robustness': []
        }
        
        for test_case in self.test_cases:
            # Test retrieval accuracy
            retrieval_score = await self.test_retrieval_accuracy(test_case)
            results['retrieval_accuracy'].append(retrieval_score)
            
            # Test response quality
            quality_score = await self.test_response_quality(test_case)
            results['response_quality'].append(quality_score)
            
            # Test performance
            perf_metrics = await self.test_performance(test_case)
            results['performance'].append(perf_metrics)
            
        # Test robustness
        robustness_score = await self.test_robustness()
        results['robustness'] = robustness_score
        
        return self.generate_report(results)
    
    async def test_retrieval_accuracy(self, test_case):
        """Testa accuratezza del retrieval"""
        result = await self.rag.query(test_case.question)
        
        # Verifica moduli trovati
        found_modules = set(doc['module'] for doc in result['sources'])
        expected_modules = set(test_case.expected_modules)
        module_recall = len(found_modules & expected_modules) / len(expected_modules)
        
        # Verifica entità estratte
        found_entities = set(result.get('entities', []))
        expected_entities = set(test_case.expected_entities)
        entity_recall = len(found_entities & expected_entities) / len(expected_entities)
        
        return {
            'module_recall': module_recall,
            'entity_recall': entity_recall,
            'overall_score': (module_recall + entity_recall) / 2
        }
```

---

## Laboratorio Pratico - Parte 2

### Esercizio 6: Ottimizzazione e Monitoring
```python
# monitoring.py
from prometheus_client import Counter, Histogram, Gauge
import logging

# Metriche Prometheus
query_counter = Counter('rag_queries_total', 'Totale query RAG')
query_latency = Histogram('rag_query_duration_seconds', 'Latenza query RAG')
active_users = Gauge('rag_active_users', 'Utenti attivi')
cache_hit_rate = Gauge('rag_cache_hit_rate', 'Cache hit rate')

class RAGMonitor:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.metrics_buffer = []
        
    async def monitor_query(self, query_func, *args, **kwargs):
        """Wrapper per monitorare query"""
        query_counter.inc()
        
        with query_latency.time():
            try:
                result = await query_func(*args, **kwargs)
                
                # Log dettagliato
                self.log_query_details({
                    'query': kwargs.get('query', ''),
                    'latency': query_latency._value.get(),
                    'sources_count': len(result.get('sources', [])),
                    'confidence': result.get('confidence', 0),
                    'cache_hit': result.get('from_cache', False)
                })
                
                return result
                
            except Exception as e:
                self.logger.error(f"Query failed: {str(e)}")
                raise
    
    def analyze_performance(self):
        """Analizza performance del sistema"""
        analysis = {
            'avg_latency': np.mean([m['latency'] for m in self.metrics_buffer]),
            'p95_latency': np.percentile([m['latency'] for m in self.metrics_buffer], 95),
            'cache_effectiveness': self.calculate_cache_effectiveness(),
            'popular_queries': self.identify_popular_queries(),
            'bottlenecks': self.identify_bottlenecks()
        }
        
        return analysis
    
    def generate_optimization_suggestions(self, analysis):
        """Genera suggerimenti per ottimizzazione"""
        suggestions = []
        
        if analysis['avg_latency'] > 2.0:
            suggestions.append({
                'issue': 'High average latency',
                'suggestion': 'Consider implementing query result caching',
                'priority': 'high'
            })
            
        if analysis['cache_effectiveness'] < 0.3:
            suggestions.append({
                'issue': 'Low cache hit rate',
                'suggestion': 'Analyze query patterns and pre-warm cache',
                'priority': 'medium'
            })
            
        return suggestions
```

---

## Progetto Finale

### Requisiti del progetto
Costruire un sistema RAG completo per gestione knowledge base ERP/CRM che:

1. **Ingestion**: processi documenti multipli formati (PDF, Word, Excel)
2. **Dual Retrieval**: implementi sia vector che graph retrieval
3. **Multi-tenant**: supporti diversi dipartimenti con permessi
4. **Multilingue**: gestisca italiano e inglese
5. **API REST**: esponga endpoints per integrazione
6. **Monitoring**: includa dashboard metriche
7. **Testing**: suite test automatizzati

### Struttura consegna
```
progetto_rag_finale/
├── README.md
├── requirements.txt
├── docker-compose.yml
├── src/
│   ├── api/
│   ├── core/
│   ├── retrieval/
│   └── evaluation/
├── tests/
├── docs/
└── notebooks/
```

---

## Criteri di Valutazione

### Valutazione progetto (100 punti)
1. **Architettura (20 punti)**
   - Design modulare e scalabile
   - Best practices implementate
   - Documentazione architetturale

2. **Implementazione RAG (30 punti)**
   - Vector retrieval efficace
   - Graph retrieval funzionante
   - Hybrid approach bilanciato

3. **Qualità Codice (20 punti)**
   - Clean code principles
   - Test coverage >80%
   - Error handling robusto

4. **Performance (15 punti)**
   - Latenza <2s per query
   - Throughput >100 QPS
   - Ottimizzazioni implementate

5. **Features Avanzate (15 punti)**
   - Multilingua
   - Caching intelligente
   - Monitoring/Analytics

---

## Best Practices Riepilogo

### Do's ✅
- **Chunk intelligentemente**: usa boundaries semantici
- **Embeddings appropriati**: multilingui per contenuto misto
- **Metadata ricchi**: facilitano filtering e ranking
- **Cache aggressivo**: riduci latenza e costi
- **Monitor costantemente**: identifica bottlenecks
- **Test exhaustively**: copri edge cases

### Don'ts ❌
- **No chunking fisso**: adatta alla struttura documento
- **No single retrieval**: combina approcci
- **No hardcoded prompts**: usa template configurabili
- **No sync operations**: async per scalabilità
- **No monoliths**: microservizi per flessibilità

---

## Risorse e Riferimenti

### Documentazione
- [LangChain RAG](https://python.langchain.com/docs/use_cases/question_answering/)
- [LlamaIndex Guide](https://docs.llamaindex.ai/en/stable/understanding/rag.html)
- [RAGAS Metrics](https://docs.ragas.io/)
- [NanoGraphRAG](https://github.com/gusye1234/nano-graphrag)

### Papers fondamentali
- "Retrieval-Augmented Generation" (Lewis et al., 2020)
- "Dense Passage Retrieval" (Karpukhin et al., 2020)
- "REALM: Retrieval-Augmented LM" (Guu et al., 2020)
- "GraphRAG: Unlocking LLM discovery" (Microsoft, 2024)

### Tools e frameworks
- Vector DBs: Pinecone, ChromaDB, Weaviate, Qdrant
- Graph DBs: Neo4j, ArangoDB, Amazon Neptune
- Frameworks: LangChain, LlamaIndex, Haystack
- Evaluation: RAGAS, TruLens, Arize Phoenix

---

## Conclusione del Corso

### Competenze acquisite
✅ Comprensione profonda architetture LLM
✅ Mastery del Prompt Engineering
✅ Gestione VectorDB e embeddings
✅ Implementazione sistemi RAG completi
✅ GraphRAG e approcci ibridi
✅ Deployment e monitoring produzione

### Prossimi passi
1. Completare progetto finale
2. Sperimentare con nuove tecniche
3. Contribuire a progetti open source
4. Applicare in contesti reali
5. Continuous learning!

---

## Q&A Session

### Domande frequenti
- Come scegliere tra RAG e fine-tuning?
- Quando usare GraphRAG vs Vector RAG?
- Come gestire documenti multilingua?
- Best practices per security in RAG?
- Come ridurre costi in produzione?

### Contatti e supporto continuo
- Slack del corso: #rag-erp-crm
- Email: docente@corso-rag.it
- GitHub: github.com/corso-rag-erp-crm
- Office hours: su appuntamento

---

## Grazie e Arrivederci!

### Il vostro viaggio RAG è appena iniziato! 🚀

> "The best way to predict the future is to build it"
> - Alan Kay

### Ricordate:
- Sperimentate sempre
- Misurate tutto
- Condividete conoscenza
- Stay curious!

### Buona fortuna con i vostri progetti RAG! 💪