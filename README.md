# RAG Chatbot — Deep Dive

A production-oriented reference on Retrieval-Augmented Generation: architecture, system design, tradeoffs, and engineering decisions.

---

## Table of Contents

1. [What is RAG?](#what-is-rag)
2. [Why RAG Over Fine-Tuning?](#why-rag-over-fine-tuning)
3. [Full System Architecture](#full-system-architecture)
4. [Offline Pipeline — Indexing](#offline-pipeline--indexing)
5. [Online Pipeline — Query Time](#online-pipeline--query-time)
6. [System Design Considerations](#system-design-considerations)
7. [Accuracy vs Latency Tradeoffs](#accuracy-vs-latency-tradeoffs)
8. [Failure Modes and How to Handle Them](#failure-modes-and-how-to-handle-them)
9. [Advanced RAG Patterns](#advanced-rag-patterns)
10. [When to Use RAG](#when-to-use-rag)
11. [Final Summary](#final-summary)

---

## What is RAG?

![RAG Architecture](https://softwarediagrams.com/assets/generated/diagrams/2025/07/20/rag-architectures-humanloop/1.webp)

Retrieval-Augmented Generation (RAG) is an architectural pattern that augments a Large Language Model (LLM) with a dynamic, queryable knowledge base at inference time.

A standard LLM is frozen after training. It cannot access new information, cannot reason over your private documents, and will hallucinate when asked questions beyond its training distribution. RAG solves this by treating retrieval as a first-class step in the generation pipeline.

The core idea:

```
User Query
    ↓
Search a knowledge base for relevant context
    ↓
Inject that context into the LLM's prompt
    ↓
LLM generates a grounded response
```

RAG was introduced in the 2020 paper *"Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"* (Lewis et al., Facebook AI). It has since become the dominant pattern for building AI systems over proprietary or frequently updated data.

---

## Why RAG Over Fine-Tuning?

Fine-tuning and RAG are often compared. Understanding when to use each requires understanding what they solve.

| Dimension | Fine-Tuning | RAG |
|---|---|---|
| Updates knowledge | No (frozen post-training) | Yes (update the vector DB) |
| Cost | High (GPU hours, data prep) | Low (index documents once) |
| Hallucination risk | Still present | Significantly reduced |
| Transparency | Black box | Retrievable sources |
| Custom data | Baked into weights | Stored externally |
| Latency | Low (no retrieval step) | Slightly higher |
| Best for | Style/behavior changes | Knowledge-intensive tasks |

**Rule of thumb:**
- Use fine-tuning when you want the model to *behave* differently (tone, format, reasoning style).
- Use RAG when you want the model to *know* something it doesn't know from training.
- Use both when you need domain-specific behavior AND accurate knowledge.

---

## Full System Architecture

A production RAG system has two distinct pipelines:

```
OFFLINE PIPELINE (runs once or on schedule)
─────────────────────────────────────────────────────────────────
Raw Data → Loader → Cleaner → Chunker → Embedder → Vector DB

ONLINE PIPELINE (runs on every user query)
─────────────────────────────────────────────────────────────────
User Query → Query Embedder → Vector Search → Re-ranker
    → Prompt Builder → LLM → Response → (optional) Citation Layer
```

These pipelines are intentionally decoupled. The offline pipeline is a batch job. The online pipeline must be low-latency.

---

## Offline Pipeline — Indexing

### 1. Data Loading

Before any processing, raw data must be loaded from its source. Common sources:

- **PDFs** — research papers, internal docs, contracts
- **Web pages** — crawled HTML, markdown exports
- **Databases** — SQL tables exported as text
- **APIs** — Notion, Confluence, Slack exports
- **Code repositories** — source files, docstrings

Each source type requires a different loader. Libraries like LangChain and LlamaIndex provide loaders for most common formats. Custom loaders are often necessary for enterprise systems.

**Engineering consideration:** Loading is often I/O bound. Use async loading where possible. For large corpora (millions of documents), use a distributed ingestion queue (e.g., Kafka, Celery).

---

### 2. Data Cleaning and Preprocessing

Raw text is noisy. Ingesting it directly into a vector database degrades retrieval quality.

Common cleaning steps:

- Strip HTML tags, headers, footers, page numbers
- Normalize unicode and encoding issues
- Deduplicate identical or near-identical passages
- Remove boilerplate (legal disclaimers, cookie notices)
- Detect and handle tables, code blocks, and structured data separately
- Language detection and filtering if multilingual

**Why this matters:** An embedding model cannot distinguish signal from noise. Dirty text produces noisy embeddings, which produce irrelevant retrievals, which produce hallucinated answers. Garbage in, garbage out.

---

### 3. Chunking

Documents must be broken into smaller units before embedding. This is called chunking. It is one of the most impactful and underappreciated decisions in a RAG pipeline.

**Why chunk at all?**

- Embedding models have token limits (typically 512–8192 tokens)
- Smaller chunks allow more precise retrieval
- LLM context windows have limits — you cannot inject an entire document

**Chunking strategies:**

#### Fixed-Size Chunking
Split every N tokens regardless of content.

```
[token_1 ... token_512] [token_513 ... token_1024] ...
```

Pros: Simple, predictable.  
Cons: Breaks sentences, paragraphs, and logical units mid-thought.

#### Sentence-Based Chunking
Split on sentence boundaries.

```
"RAG improves accuracy. It does so by retrieving context."
→ ["RAG improves accuracy.", "It does so by retrieving context."]
```

Pros: Preserves grammatical units.  
Cons: Sentences alone may lack sufficient context.

#### Paragraph / Section-Based Chunking
Split on natural document structure (paragraphs, headings, sections).

Pros: Preserves semantic coherence.  
Cons: Variable chunk sizes; large sections may exceed limits.

#### Recursive Chunking
Attempt to split on paragraphs first, fall back to sentences, fall back to fixed-size. Used by LangChain's `RecursiveCharacterTextSplitter`.

Pros: Best balance of structure and size control.  
Cons: Slightly more complex to implement.

#### Semantic Chunking
Use an embedding model to detect where topic shifts occur and split there.

Pros: Highly coherent chunks.  
Cons: Computationally expensive; slower indexing.

---

### 4. Chunk Overlap

When splitting text, context can be lost at boundaries. Overlap addresses this by repeating a portion of the previous chunk at the start of the next.

**Without overlap:**
```
Chunk 1: "The rate hike was driven by"
Chunk 2: "persistent inflation in the housing sector."
```
If the query asks about the cause of the rate hike, neither chunk alone contains the full answer.

**With overlap (e.g., 20% overlap):**
```
Chunk 1: "The rate hike was driven by persistent inflation"
Chunk 2: "persistent inflation in the housing sector."
```
Now both chunks contain enough context to be retrieved and understood.

**Overlap tradeoff:**

| More Overlap | Less Overlap |
|---|---|
| Better context preservation | Smaller index size |
| More redundant retrievals | Less redundant results |
| Higher storage cost | Possible boundary artifacts |

Typical overlap: 10–20% of chunk size.

---

### 5. Embedding

Embedding is the process of converting text into a dense numerical vector that encodes its semantic meaning.

```
"What is the capital of France?" → [0.231, -0.847, 0.112, ..., 0.034]  (768 or 1536 dimensions)
```

Texts with similar meanings produce vectors that are geometrically close to each other in high-dimensional space.

**How embeddings work (simplified):**

Transformer-based models (BERT, sentence-transformers, OpenAI Ada) process text and output a fixed-size vector from their final hidden layer. The model is trained specifically to place semantically similar texts near each other using contrastive learning.

**Embedding model choices:**

| Model | Dimensions | Speed | Quality | Cost |
|---|---|---|---|---|
| `text-embedding-3-small` (OpenAI) | 1536 | Fast | High | Paid |
| `text-embedding-3-large` (OpenAI) | 3072 | Moderate | Very High | Paid |
| `all-MiniLM-L6-v2` (local) | 384 | Very Fast | Good | Free |
| `bge-large-en` (local) | 1024 | Moderate | High | Free |
| `nomic-embed-text` (local) | 768 | Fast | High | Free |

**Critical rule:** The embedding model used at indexing time must be the same model used at query time. Mixing models produces incoherent similarity scores.

---

### 6. Vector Database Storage

Embeddings and their associated metadata are stored in a vector database optimized for similarity search.

**What gets stored per chunk:**

```json
{
  "id": "doc_47_chunk_12",
  "embedding": [0.231, -0.847, 0.112, ...],
  "text": "The rate hike was driven by persistent inflation in the housing sector.",
  "metadata": {
    "source": "fed_report_2024.pdf",
    "page": 14,
    "section": "Monetary Policy",
    "ingested_at": "2024-11-01T09:22:00Z"
  }
}
```

The embedding is what gets searched. The text and metadata get returned and injected into the prompt.

**Vector database options:**

| Database | Hosting | Best For |
|---|---|---|
| FAISS | Local / self-hosted | Prototyping, offline use |
| Chroma | Local / self-hosted | Small to medium scale |
| Pinecone | Managed cloud | Production, high scale |
| Weaviate | Self-hosted / cloud | Hybrid search, complex filtering |
| Qdrant | Self-hosted / cloud | High performance, metadata filtering |
| pgvector | PostgreSQL extension | Teams already on Postgres |
| Redis VSS | Self-hosted / cloud | Low latency, caching combined |

---

## Online Pipeline — Query Time

### 1. Query Embedding

The user's query is converted to an embedding using the same model used during indexing.

```
"What caused the 2024 rate hike?" → [0.198, -0.812, 0.143, ...]
```

**Query preprocessing (optional but impactful):**

- **Query expansion:** Generate multiple reformulations of the query and retrieve for each, then merge results.
- **HyDE (Hypothetical Document Embeddings):** Ask the LLM to generate a hypothetical answer to the query, embed that, and use it for retrieval. The hypothesis tends to be closer in embedding space to real answers than the raw question.
- **Query normalization:** Lowercase, remove filler words, fix typos before embedding.

---

### 2. Vector Search (Retrieval)

The query embedding is compared against all stored embeddings using a similarity metric.

**Similarity metrics:**

- **Cosine similarity** — measures the angle between vectors. Most common. Range: [-1, 1]. Higher = more similar.
- **Dot product** — faster than cosine when vectors are normalized.
- **Euclidean distance (L2)** — measures absolute distance. Less common for text.

**Exact vs Approximate search:**

| Method | Accuracy | Speed | Scale |
|---|---|---|---|
| Exact (brute force) | 100% | Slow at scale | <100K vectors |
| ANN (HNSW, IVF, etc.) | ~95–99% | Very fast | Millions of vectors |

Approximate Nearest Neighbor (ANN) algorithms trade a small amount of recall for dramatic speed improvements. HNSW (Hierarchical Navigable Small World) is the most common algorithm, used in FAISS, Weaviate, and Qdrant.

**Top-K parameter:**

| K too small | K too large |
|---|---|
| May miss relevant context | Injects irrelevant context |
| Faster LLM inference | Slower LLM inference |
| Lower token cost | Higher token cost |
| Risk: incomplete answers | Risk: confused or distracted LLM |

Typical values: K = 3–10. Tune empirically based on your data.

---

### 3. Re-Ranking

Vector search retrieves by semantic similarity, but similarity is not the same as relevance. Re-ranking adds a second pass that scores retrieved chunks more precisely.

**Two-stage retrieval:**

```
Stage 1 (fast):     Vector search → retrieve top-50 candidates
Stage 2 (accurate): Cross-encoder re-ranker → select top-5 from candidates
```

Cross-encoders compare the query and document together (full attention), which is far more accurate than bi-encoder cosine similarity. But they are too slow to run over an entire vector database. Running them over a small candidate set (top-50) is fast.

**Re-ranking models:**

- `cross-encoder/ms-marco-MiniLM-L-6-v2` (local, fast)
- Cohere Rerank API (managed, high quality)
- `bge-reranker-large` (local, high quality)

**Tradeoff:**

| Without Re-ranking | With Re-ranking |
|---|---|
| Lower latency | +50–200ms latency |
| Sometimes irrelevant top results | Significantly more relevant results |
| Lower cost | Slightly higher cost |

---

### 4. Prompt Construction

Retrieved chunks are injected into the LLM's prompt as context.

**Basic prompt template:**

```
You are a helpful assistant. Answer the question based only on the provided context.
If the answer is not in the context, say "I don't know."

Context:
---
[Chunk 1 text]
---
[Chunk 2 text]
---
[Chunk 3 text]
---

Question: {user_query}

Answer:
```

**Prompt engineering considerations:**

- **Instruction clarity:** Explicitly tell the model to use only the provided context. Without this, it may mix retrieved context with parametric memory.
- **Context ordering:** Models attend better to content at the beginning and end of the context window ("lost in the middle" problem). Place the most relevant chunks first.
- **Citation prompting:** Ask the model to cite which chunk its answer comes from. Improves traceability and user trust.
- **Negative space instructions:** Tell the model what to do when it doesn't know. Reduces hallucination on unanswerable queries.

---

### 5. LLM Generation

**LLM choice factors:**

| Factor | Consideration |
|---|---|
| Context window | Must fit query + all retrieved chunks + response |
| Speed | Affects end-to-end latency |
| Cost | Per-token pricing adds up at scale |
| Quality | Larger models reason better over noisy context |
| Hosting | API (OpenAI, Anthropic) vs self-hosted (Llama, Mistral) |

**Streaming:** For user-facing applications, always stream the LLM response. Users tolerate latency far better when they see tokens appearing progressively rather than waiting for the full response.

---

## System Design Considerations

### Scalability

**Indexing at scale:**

- Use a distributed job queue (Celery, Ray, Spark) for ingestion
- Parallelize embedding generation with GPU acceleration
- Batch API calls to embedding providers to reduce cost and increase throughput
- Use incremental indexing — only re-embed documents that have changed

**Query at scale:**

- Deploy the retriever behind a load balancer
- Cache frequent queries (Redis or in-memory LRU)
- Use read replicas for the vector database
- Pre-warm the ANN index in memory

---

### Metadata Filtering (Hybrid Search)

Pure vector search retrieves by semantic similarity but ignores structure. Metadata filtering lets you add hard constraints.

**Example:** A user asks "What did the Q3 2024 earnings report say about margins?" You want to restrict retrieval to documents tagged `year=2024`, `quarter=Q3`, `type=earnings_report`.

```python
results = vector_db.query(
    embedding=query_embedding,
    top_k=5,
    filters={"year": 2024, "quarter": "Q3", "type": "earnings_report"}
)
```

**Hybrid search** combines vector similarity with keyword (BM25) search:

```
final_score = α × vector_score + (1 - α) × bm25_score
```

Hybrid search outperforms pure vector search on queries with specific entities, proper nouns, or exact terms that may not be well-represented in embedding space.

---

### Caching

Three layers of caching can dramatically reduce latency and cost:

| Cache Layer | What it Stores | Benefit |
|---|---|---|
| Query cache | Full (query → response) pairs | Eliminates all processing for repeated queries |
| Embedding cache | (text → embedding) | Skips API call for re-queried text |
| Retrieval cache | (query → retrieved chunks) | Skips vector search for repeated queries |

Use semantic similarity-based caching (e.g., GPTCache) to cache not just exact matches but near-duplicate queries.

---

### Observability and Evaluation

**What to log:**

- User query
- Retrieved chunk IDs and similarity scores
- Re-ranked scores
- Final prompt sent to LLM
- LLM response
- Latency at each stage
- User feedback (thumbs up/down, corrections)

**Evaluation metrics:**

| Metric | What it Measures |
|---|---|
| Retrieval Recall@K | Were relevant documents in the top K? |
| MRR (Mean Reciprocal Rank) | How high was the first relevant result? |
| Answer Faithfulness | Does the answer contradict the retrieved context? |
| Answer Relevance | Does the answer address the question? |
| Context Precision | What fraction of retrieved context was actually used? |

Tools: RAGAS, TruLens, LangSmith, Arize AI.

---

## Accuracy vs Latency Tradeoffs

This is the central engineering tension in RAG. Every design decision sits somewhere on this spectrum.

```
HIGH ACCURACY                              LOW LATENCY
      |------------------------------------------|

More chunks (K=10)          ←→         Fewer chunks (K=3)
Re-ranking on               ←→         No re-ranking
Large embedding model       ←→         Small embedding model
Exact ANN search            ←→         Approximate ANN (HNSW)
Large LLM (GPT-4)           ←→         Small LLM (GPT-3.5, Llama-3-8B)
No caching                  ←→         Aggressive caching
Semantic chunking           ←→         Fixed-size chunking
HyDE query expansion        ←→         Raw query embedding
Hybrid search               ←→         Vector-only search
```

**Typical latency breakdown for a production RAG call:**

| Step | Typical Latency |
|---|---|
| Query embedding | 20–80ms |
| Vector search (ANN) | 5–50ms |
| Re-ranking (optional) | 50–200ms |
| Prompt construction | <5ms |
| LLM generation (first token) | 200–800ms |
| LLM generation (full response) | 1–10s |
| **Total (no re-ranking)** | **~300–1000ms** |
| **Total (with re-ranking)** | **~400–1200ms** |

**Optimization strategies by priority:**

1. **Stream the LLM response** — biggest perceived latency improvement, zero accuracy cost.
2. **Cache at query level** — eliminates cost and latency for repeated queries entirely.
3. **Use ANN over exact search** — 10–100x speedup with <5% accuracy loss at scale.
4. **Tune K** — reduce from 10 to 5 if your chunks are high quality. Half the context = faster LLM inference.
5. **Use a smaller LLM for simple queries** — route simple factual queries to a fast small model; route complex synthesis to a large model.
6. **Batch embedding generation** — critical for indexing pipelines.

---

## Failure Modes and How to Handle Them

### Retrieval Failure
The correct document exists in the index but is not retrieved.

**Causes:** Poor chunking, wrong embedding model, K too small, query poorly phrased.

**Fixes:** Increase K, improve chunking strategy, add query expansion, use hybrid search.

---

### Hallucination Despite Retrieval
The LLM ignores retrieved context and generates from parametric memory.

**Causes:** Prompt not explicit enough, retrieved context contradicts model's prior knowledge, context is too long.

**Fixes:** Strengthen prompt instructions, reduce context size, use a model with better instruction following.

---

### Irrelevant Retrieval
Retrieved chunks are semantically similar to the query but not actually useful.

**Causes:** K too high, no re-ranking, poor metadata filtering.

**Fixes:** Add re-ranking, reduce K, add metadata filters, improve chunk quality.

---

### Stale Knowledge
The vector database contains outdated documents.

**Causes:** No ingestion refresh schedule.

**Fixes:** Implement delta ingestion (detect and re-index changed documents), add `ingested_at` metadata, set document TTLs.

---

### Context Window Overflow
The total prompt exceeds the LLM's context limit.

**Causes:** K too high, chunks too large, long system prompt.

**Fixes:** Reduce K, reduce chunk size, use a model with a larger context window, implement context compression (summarize retrieved chunks before injecting).

---

## Advanced RAG Patterns

### Parent-Child Chunking
Index small child chunks for precise retrieval, but inject their larger parent chunk into the prompt for full context.

```
Parent: Full paragraph (512 tokens) — injected into prompt
  └── Child: First sentence (64 tokens) — used for retrieval
  └── Child: Second sentence (64 tokens) — used for retrieval
```

Combines precision of small-chunk retrieval with context richness of large-chunk injection.

---

### Self-Query Retrieval
Use an LLM to parse the user's natural language query into structured filters plus a semantic search query.

```
User: "Find documents about AI regulation published after 2023"
    ↓ LLM parses into:
{
  "semantic_query": "AI regulation",
  "filters": { "year": { "$gte": 2024 } }
}
```

---

### Agentic / Multi-Hop RAG
For complex questions that require reasoning over multiple documents, use an agent that performs multiple retrieval steps.

```
Query: "Compare the revenue growth of Company A and Company B in 2024"
    Step 1: Retrieve Company A 2024 financials
    Step 2: Retrieve Company B 2024 financials
    Step 3: Synthesize comparison
```

---

### Corrective RAG (CRAG)
After retrieval, score the relevance of retrieved documents. If relevance is low, trigger a fallback (web search, broader retrieval, or "I don't know").

---

## When to Use RAG

**Use RAG when:**
- Your data is proprietary or not in the LLM's training set
- Your data changes frequently (news, internal docs, product catalogs)
- You need citations and source traceability
- Hallucination is unacceptable in your domain (legal, medical, finance)
- You have large volumes of documents too large to fine-tune on

**Do not use RAG when:**
- The task requires behavioral changes (use fine-tuning instead)
- Ultra-low latency is required (<100ms end-to-end)
- Your knowledge is static and small enough to fit in a prompt
- You need the model to internalize complex reasoning patterns across many examples

---

## Final Summary

RAG is not a single algorithm — it is a system design pattern. Its quality is determined by the sum of every decision: how you chunk, what you embed, how you retrieve, whether you re-rank, and how you prompt.

```
Retrieve → Augment → Generate
```

The retrieval step is the most impactful and least understood component. A poorly indexed knowledge base cannot be rescued by a better LLM. Invest in chunking, embedding model selection, and retrieval evaluation before optimizing generation.

The fundamental tradeoff in RAG is accuracy vs latency. Every added layer of sophistication (re-ranking, hybrid search, query expansion, larger models) improves answer quality at the cost of response time. Production systems must decide where on this spectrum to sit based on their latency SLA and quality requirements.

At its core, RAG transforms a language model into a reasoning engine over external knowledge — accurate, current, and traceable.






To run the code:
install the dependencies from requirements.
Create your data folder and add your data to the folder
(Create a virtual env if in ubuntu)
Execute
