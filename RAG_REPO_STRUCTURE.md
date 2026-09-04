# RAG Specialization Repository Structure

A progressive, hands-on repository for learning **Retrieval-Augmented Generation (RAG)** from the fundamentals to production-grade systems.

The repository is organized into **7 levels**. Each folder represents one stage of the specialization and contains theory, examples, experiments, evaluation, and a dedicated `README.md`.

---

# Repository Tree

```text
rag-specialization/
│
├── README.md
├── .gitignore
├── .env.example
├── pyproject.toml
├── docker-compose.yml
├── Makefile
│
├── shared/
│   ├── data/
│   ├── models/
│   ├── prompts/
│   ├── utils/
│   ├── config/
│   └── evaluation/
│
├── 01-naive-rag/
│   ├── README.md
│   ├── theory/
│   ├── notebooks/
│   ├── src/
│   ├── examples/
│   ├── tests/
│   └── data/
│
├── 02-advanced-rag/
│   ├── README.md
│   ├── chunking/
│   ├── retrieval/
│   ├── hybrid-search/
│   ├── reranking/
│   ├── query-transformations/
│   ├── metadata-filtering/
│   ├── context-compression/
│   ├── evaluation/
│   └── examples/
│
├── 03-modular-rag/
│   ├── README.md
│   ├── routing/
│   ├── multi-retriever/
│   ├── sql-rag/
│   ├── graph-rag/
│   ├── web-rag/
│   ├── api-rag/
│   ├── multimodal-rag/
│   ├── examples/
│   └── tests/
│
├── 04-adaptive-rag/
│   ├── README.md
│   ├── query-classification/
│   ├── dynamic-retrieval/
│   ├── corrective-rag/
│   ├── self-rag/
│   ├── multi-hop-rag/
│   ├── fallback-strategies/
│   ├── examples/
│   └── evaluation/
│
├── 05-agentic-rag/
│   ├── README.md
│   ├── agents/
│   ├── tools/
│   ├── planning/
│   ├── memory/
│   ├── iterative-retrieval/
│   ├── reflection/
│   ├── verification/
│   ├── examples/
│   └── tests/
│
├── 06-multi-agent-rag/
│   ├── README.md
│   ├── supervisor/
│   ├── research-agent/
│   ├── retrieval-agent/
│   ├── sql-agent/
│   ├── web-agent/
│   ├── graph-agent/
│   ├── verification-agent/
│   ├── synthesis-agent/
│   ├── workflows/
│   └── examples/
│
└── 07-production-rag/
    ├── README.md
    ├── api/
    ├── inference/
    ├── retrieval-infrastructure/
    ├── observability/
    ├── evaluation/
    ├── security/
    ├── caching/
    ├── deployment/
    ├── load-testing/
    ├── docker/
    ├── kubernetes/
    └── examples/
```

---

# Root README.md

Your root `README.md` should explain the full journey.

```md
# RAG Specialization

A practical repository for mastering Retrieval-Augmented Generation from a basic vector-search pipeline to production multi-agent RAG systems.

## Roadmap

1. Naive RAG
2. Advanced RAG
3. Modular RAG
4. Adaptive RAG
5. Agentic RAG
6. Multi-Agent RAG
7. Production RAG

## Main Open-Source Stack

- Python
- Ollama for local development
- vLLM or SGLang for production inference
- Qdrant for vector/hybrid retrieval
- PostgreSQL + pgvector for relational retrieval
- Redis for cache/state
- LlamaIndex for RAG pipelines
- LangGraph for agent orchestration
- FastAPI for APIs
- Ragas for evaluation
- OpenTelemetry + Prometheus + Grafana for observability
- Docker for packaging

## Learning Rule

Each level contains:

- Theory
- Minimal implementation
- Framework implementation
- Experiments
- Evaluation
- Tests
- A small project

Do not move to the next level until you can explain why the current system succeeds or fails.
```

---

# Shared Folder

The `shared/` directory contains reusable assets.

```text
shared/
├── data/
│   ├── sample.pdf
│   ├── faq.md
│   └── documents.jsonl
├── models/
│   └── model_config.yaml
├── prompts/
│   ├── rag_prompt.txt
│   ├── query_rewrite_prompt.txt
│   └── verifier_prompt.txt
├── utils/
│   ├── loaders.py
│   ├── logging.py
│   └── text.py
├── config/
│   ├── settings.py
│   └── retrieval.yaml
└── evaluation/
    ├── questions.jsonl
    └── ground_truth.jsonl
```

Use the same small dataset across levels so you can compare retrieval quality as the architecture becomes more advanced.

---

# Level 1 — Naive RAG

## Goal

Understand the complete basic RAG pipeline without hiding the fundamentals.

```text
Document
  ↓
Load
  ↓
Chunk
  ↓
Embed
  ↓
Vector Store
  ↓
Retrieve Top-K
  ↓
Prompt
  ↓
LLM
  ↓
Answer
```

## Folder

```text
01-naive-rag/
├── README.md
├── theory/
│   ├── embeddings.md
│   ├── cosine_similarity.md
│   ├── chunking.md
│   └── vector_search.md
├── notebooks/
│   ├── 01_embeddings.ipynb
│   ├── 02_similarity.ipynb
│   ├── 03_vector_search.ipynb
│   └── 04_first_rag.ipynb
├── src/
│   ├── ingest.py
│   ├── chunk.py
│   ├── embed.py
│   ├── retrieve.py
│   └── generate.py
├── examples/
│   ├── simple_rag.py
│   ├── rag_with_qdrant.py
│   └── rag_with_ollama.py
├── tests/
│   ├── test_chunking.py
│   └── test_retrieval.py
└── data/
    └── sample_docs/
```

## README.md Template

```md
# Level 1 — Naive RAG

## Objective

Build a RAG system from first principles.

## Topics

- Documents and loaders
- Text cleaning
- Chunking
- Embeddings
- Cosine similarity
- Vector databases
- Top-K retrieval
- Prompt augmentation
- Local LLM generation

## Stack

- Python
- Ollama
- Qdrant
- Sentence Transformers or Ollama embeddings

## Exercises

1. Embed two sentences and calculate cosine similarity.
2. Build a tiny in-memory vector search implementation.
3. Store document chunks in Qdrant.
4. Retrieve Top-3 chunks.
5. Send retrieved context to a local LLM.
6. Return an answer with source IDs.

## Mini Project

Build a local PDF question-answering assistant.

## Success Criteria

You can explain:

- Why embeddings work.
- Why chunk size changes retrieval quality.
- What Top-K means.
- Why the LLM can still hallucinate even when retrieval works.
```

## Example

```python
query = "What is the refund period?"

query_vector = embed(query)
results = vector_store.search(query_vector, limit=3)

context = "\n\n".join(item.text for item in results)

prompt = f"""
Answer using only the context below.

Context:
{context}

Question:
{query}
"""

answer = llm.generate(prompt)
print(answer)
```

---

# Level 2 — Advanced RAG

## Goal

Improve retrieval quality and learn how to diagnose retrieval failures.

## Folder

```text
02-advanced-rag/
├── README.md
├── chunking/
│   ├── fixed_size.py
│   ├── recursive.py
│   ├── semantic.py
│   └── parent_child.py
├── retrieval/
│   ├── dense.py
│   ├── sparse.py
│   └── top_k_experiments.py
├── hybrid-search/
│   ├── bm25_vector.py
│   └── rrf.py
├── reranking/
│   ├── cross_encoder.py
│   └── bge_reranker.py
├── query-transformations/
│   ├── query_rewrite.py
│   ├── multi_query.py
│   ├── hyde.py
│   └── step_back.py
├── metadata-filtering/
│   └── filters.py
├── context-compression/
│   └── compressor.py
├── evaluation/
│   ├── recall_at_k.py
│   ├── mrr.py
│   └── ndcg.py
└── examples/
    └── advanced_pipeline.py
```

## README.md Template

```md
# Level 2 — Advanced RAG

## Objective

Move from "vector search works" to "retrieval is measurable and tunable".

## Topics

- Fixed vs semantic chunking
- Parent-child retrieval
- Dense retrieval
- Sparse/BM25 retrieval
- Hybrid search
- Reciprocal Rank Fusion
- Query rewriting
- Multi-query retrieval
- HyDE
- Metadata filtering
- Reranking
- Context compression
- Retrieval evaluation

## Experiments

Run every experiment on the same evaluation dataset.

Compare:

- chunk size 256 vs 512 vs 1024
- overlap 0 vs 50 vs 100
- dense only vs sparse only vs hybrid
- Top-5 vs Top-20 before reranking
- without reranker vs with reranker

## Metrics

- Recall@K
- Precision@K
- MRR
- NDCG
- Answer faithfulness

## Mini Project

Build a high-quality documentation assistant with hybrid retrieval and reranking.

## Success Criteria

You can identify whether a bad answer was caused by:

1. Ingestion
2. Chunking
3. Embeddings
4. Retrieval
5. Reranking
6. Context construction
7. Generation
```

## Example Hybrid Pipeline

```python
dense_results = dense_retriever.search(query, k=20)
sparse_results = bm25_retriever.search(query, k=20)

fused_results = reciprocal_rank_fusion(
    dense_results,
    sparse_results,
)

reranked = reranker.rank(
    query=query,
    documents=fused_results[:30],
)

context = reranked[:5]
```

---

# Level 3 — Modular RAG

## Goal

Learn that not every question should go to the same retriever.

## Architecture

```text
                    Query
                      ↓
                    Router
       ┌──────────────┼──────────────┐
       ↓              ↓              ↓
   Vector DB         SQL            Web
       ↓              ↓              ↓
   Documents       Tables          Search
       └──────────────┼──────────────┘
                      ↓
                  Synthesis
```

## Folder

```text
03-modular-rag/
├── README.md
├── routing/
│   ├── rule_router.py
│   └── llm_router.py
├── multi-retriever/
│   ├── vector_retriever.py
│   └── retriever_fusion.py
├── sql-rag/
│   ├── schema.py
│   ├── text_to_sql.py
│   └── sql_guardrails.py
├── graph-rag/
│   ├── entity_extraction.py
│   ├── graph_builder.py
│   └── graph_retrieval.py
├── web-rag/
│   ├── search.py
│   └── page_extraction.py
├── api-rag/
│   └── tool_api.py
├── multimodal-rag/
│   ├── image_retrieval.py
│   └── table_retrieval.py
└── examples/
    └── modular_rag.py
```

## README.md Template

```md
# Level 3 — Modular RAG

## Objective

Build a RAG system that uses different retrieval systems depending on the query.

## Topics

- Query routing
- Multiple vector collections
- SQL RAG
- Knowledge graph retrieval
- Web retrieval
- API retrieval
- Multimodal retrieval
- Result fusion

## Routing Example

User: "What is our refund policy?"
→ Vector RAG

User: "How many orders were completed last week?"
→ SQL

User: "What changed in the latest Python release?"
→ Web search

User: "Who reports to the CTO?"
→ Graph retrieval

## Mini Project

Build an enterprise assistant that can answer questions from:

- company documents
- PostgreSQL
- APIs
- a knowledge graph

## Success Criteria

You can explain why vector search is not the correct solution for every data source.
```

## Router Example

```python
route = router.classify(query)

if route == "documents":
    result = vector_rag(query)
elif route == "sql":
    result = sql_rag(query)
elif route == "graph":
    result = graph_rag(query)
elif route == "web":
    result = web_rag(query)
```

---

# Level 4 — Adaptive RAG

## Goal

Make retrieval dynamic instead of always following a fixed pipeline.

## Folder

```text
04-adaptive-rag/
├── README.md
├── query-classification/
│   └── classifier.py
├── dynamic-retrieval/
│   ├── dynamic_top_k.py
│   └── retrieval_policy.py
├── corrective-rag/
│   └── crag.py
├── self-rag/
│   └── self_rag.py
├── multi-hop-rag/
│   ├── planner.py
│   └── subquestion_retrieval.py
├── fallback-strategies/
│   ├── web_fallback.py
│   └── retry.py
├── examples/
│   └── adaptive_pipeline.py
└── evaluation/
    └── adaptive_eval.py
```

## README.md Template

```md
# Level 4 — Adaptive RAG

## Objective

Let the system decide how much retrieval is required and what to do when retrieval quality is poor.

## Topics

- Query complexity classification
- Retrieval/no-retrieval decisions
- Dynamic Top-K
- Corrective RAG (CRAG)
- Self-RAG
- Multi-hop retrieval
- Search fallback
- Confidence-based routing

## Example Decision Policy

Simple conversational question
→ no retrieval

Known factual question
→ one retrieval step

Complex analytical question
→ multi-query + reranking

Weak retrieved evidence
→ retry / rewrite / web fallback

Multi-hop question
→ generate subquestions and retrieve repeatedly

## Mini Project

Build a RAG system that automatically selects its retrieval strategy.

## Success Criteria

You can explain why more retrieval is not always better.
```

## Example

```python
complexity = classify_query(query)

if complexity == "none":
    return llm(query)

if complexity == "simple":
    return standard_rag(query)

if complexity == "complex":
    return multi_query_rag(query)

if complexity == "multi_hop":
    return multi_hop_rag(query)
```

---

# Level 5 — Agentic RAG

## Goal

Give an agent control over retrieval and tool usage.

## Architecture

```text
User
 ↓
Agent
 ↓
Plan
 ↓
Choose Tool
 ├─ Vector Search
 ├─ SQL
 ├─ Web
 ├─ API
 └─ Graph
 ↓
Observe
 ↓
Reason
 ↓
Retrieve Again?
 ↓
Final Answer
```

## Folder

```text
05-agentic-rag/
├── README.md
├── agents/
│   ├── rag_agent.py
│   └── state.py
├── tools/
│   ├── vector_tool.py
│   ├── sql_tool.py
│   ├── web_tool.py
│   └── graph_tool.py
├── planning/
│   └── planner.py
├── memory/
│   ├── short_term.py
│   └── long_term.py
├── iterative-retrieval/
│   └── loop.py
├── reflection/
│   └── reflection.py
├── verification/
│   ├── source_checker.py
│   └── answer_verifier.py
├── examples/
│   └── research_agent.py
└── tests/
    └── test_agent.py
```

## README.md Template

```md
# Level 5 — Agentic RAG

## Objective

Move from fixed workflows to an agent that can plan, retrieve, inspect evidence, and retry.

## Topics

- Tool calling
- Agent state
- Planning
- Iterative retrieval
- Memory
- Reflection
- Verification
- Citation checking
- Stop conditions

## Tool Set

- vector_search(query)
- sql_query(question)
- web_search(query)
- graph_search(entity)
- get_document(document_id)

## Agent Loop

1. Understand the task.
2. Decide whether retrieval is needed.
3. Select a tool.
4. Inspect the result.
5. Decide whether evidence is sufficient.
6. Retrieve again if required.
7. Verify claims.
8. Produce a cited answer.

## Mini Project

Build an autonomous research assistant using LangGraph.

## Success Criteria

The agent can recover from an unsuccessful retrieval rather than immediately hallucinating an answer.
```

## Example Agent State

```python
class AgentState(TypedDict):
    question: str
    plan: list[str]
    retrieved_documents: list[dict]
    tool_history: list[dict]
    answer: str | None
    verified: bool
```

---

# Level 6 — Multi-Agent RAG

## Goal

Split complex work across specialized agents.

## Architecture

```text
                         User
                           ↓
                       Supervisor
                           ↓
       ┌───────────────┬───┴────┬───────────────┐
       ↓               ↓        ↓               ↓
 Research Agent   Retrieval   SQL Agent     Web Agent
                       Agent
       │               │        │               │
       └───────────────┴────────┴───────────────┘
                           ↓
                    Verification Agent
                           ↓
                     Synthesis Agent
                           ↓
                        Answer
```

## Folder

```text
06-multi-agent-rag/
├── README.md
├── supervisor/
│   └── supervisor.py
├── research-agent/
│   └── agent.py
├── retrieval-agent/
│   └── agent.py
├── sql-agent/
│   └── agent.py
├── web-agent/
│   └── agent.py
├── graph-agent/
│   └── agent.py
├── verification-agent/
│   └── agent.py
├── synthesis-agent/
│   └── agent.py
├── workflows/
│   ├── sequential.py
│   ├── parallel.py
│   └── supervisor_graph.py
└── examples/
    └── company_research_system.py
```

## README.md Template

```md
# Level 6 — Multi-Agent RAG

## Objective

Design systems where specialized agents collaborate on complex retrieval and reasoning tasks.

## Topics

- Supervisor architecture
- Specialized agents
- Agent communication
- Shared state
- Parallel execution
- Sequential execution
- Handoffs
- Conflict resolution
- Verification
- Final synthesis

## Recommended Agents

### Supervisor
Routes tasks and controls the workflow.

### Retrieval Agent
Searches vector and hybrid indexes.

### SQL Agent
Queries structured data.

### Web Agent
Retrieves fresh external information.

### Graph Agent
Retrieves entity relationships.

### Verification Agent
Checks whether claims are supported.

### Synthesis Agent
Combines validated results into the final answer.

## Mini Project

Build a multi-agent business research assistant.

## Success Criteria

You can explain when multi-agent architecture is useful and when it creates unnecessary complexity.
```

## Example Workflow

```python
results = await asyncio.gather(
    retrieval_agent.run(task),
    sql_agent.run(task),
    web_agent.run(task),
)

verified = verification_agent.run(results)
answer = synthesis_agent.run(verified)
```

---

# Level 7 — Production RAG

## Goal

Deploy, evaluate, secure, observe, scale, and operate a RAG system reliably.

## Production Architecture

```text
                         Client
                            ↓
                     Nginx / Traefik
                            ↓
                         FastAPI
                            ↓
                         LangGraph
                            ↓
             ┌──────────────┼──────────────┐
             ↓              ↓              ↓
          Qdrant        PostgreSQL       Redis
             ↓
       Hybrid Retrieval
             ↓
          Reranker
             ↓
          LiteLLM
             ↓
        vLLM / SGLang
             ↓
        Open Model GPU

Observability:
OpenTelemetry → Prometheus → Grafana

Evaluation:
Ragas + custom retrieval metrics
```

## Folder

```text
07-production-rag/
├── README.md
├── api/
│   ├── main.py
│   ├── routes.py
│   └── schemas.py
├── inference/
│   ├── ollama_client.py
│   ├── vllm_client.py
│   └── litellm_config.yaml
├── retrieval-infrastructure/
│   ├── qdrant.py
│   ├── postgres.py
│   └── redis.py
├── observability/
│   ├── telemetry.py
│   ├── prometheus.yml
│   └── grafana/
├── evaluation/
│   ├── ragas_eval.py
│   ├── retrieval_eval.py
│   └── regression_suite.py
├── security/
│   ├── auth.py
│   ├── permissions.py
│   ├── document_acl.py
│   └── prompt_injection.py
├── caching/
│   ├── semantic_cache.py
│   └── response_cache.py
├── deployment/
│   ├── docker-compose.yml
│   └── production.env.example
├── load-testing/
│   ├── locustfile.py
│   └── scenarios.md
├── docker/
│   ├── api.Dockerfile
│   └── worker.Dockerfile
├── kubernetes/
│   ├── api-deployment.yaml
│   ├── qdrant.yaml
│   └── ingress.yaml
└── examples/
    └── production_app/
```

## README.md Template

```md
# Level 7 — Production RAG

## Objective

Turn the previous experiments into a secure, observable, scalable production service.

## Topics

### API
- FastAPI
- request validation
- streaming responses
- async execution

### Inference
- Ollama for local development
- vLLM or SGLang for GPU production
- LiteLLM gateway
- model routing

### Retrieval Infrastructure
- Qdrant
- PostgreSQL
- Redis
- backups
- collection/index migrations

### Evaluation
- Recall@K
- MRR
- NDCG
- context precision
- context recall
- faithfulness
- answer relevance
- regression testing

### Security
- authentication
- authorization
- document-level ACLs
- tenant isolation
- prompt-injection defense
- secret management

### Observability
- logs
- traces
- metrics
- retrieval latency
- reranking latency
- generation latency
- token usage
- failure rate

### Performance
- caching
- batching
- async I/O
- connection pooling
- inference optimization
- load testing

### Deployment
- Docker
- Docker Compose
- reverse proxy
- TLS
- CI/CD
- Kubernetes only when scaling requires it

## Mini Project

Deploy a complete self-hosted RAG API using:

- FastAPI
- LangGraph
- Qdrant
- PostgreSQL
- Redis
- LiteLLM
- vLLM
- local embedding model
- local reranker
- Ragas
- OpenTelemetry
- Prometheus
- Grafana

## Success Criteria

You can answer:

- How accurate is retrieval?
- Which component causes failures?
- What is the p95 latency?
- How many concurrent users can the system support?
- What happens when Qdrant is unavailable?
- How are private documents protected?
- How do you deploy a new embedding model without corrupting the index?
- How do you detect quality regressions?
```

---

# Suggested Technology Progression

Do not introduce every technology on day one.

```text
LEVEL 1
Python
Ollama
Qdrant

LEVEL 2
+ BM25 / sparse retrieval
+ reranker
+ evaluation metrics

LEVEL 3
+ PostgreSQL
+ pgvector
+ APIs
+ graph database

LEVEL 4
+ LangGraph
+ adaptive routing

LEVEL 5
+ tools
+ memory
+ agent state

LEVEL 6
+ multiple agents
+ supervisor orchestration

LEVEL 7
Ollama → vLLM / SGLang
+ LiteLLM
+ Redis
+ FastAPI
+ OpenTelemetry
+ Prometheus
+ Grafana
+ Ragas
+ Docker
```

---

# Project Progression

Each level should finish with one practical project.

| Level | Project |
|---|---|
| 1 | Local PDF chatbot |
| 2 | Hybrid-search documentation assistant |
| 3 | Enterprise assistant using Docs + SQL + APIs |
| 4 | Adaptive RAG that changes retrieval strategy |
| 5 | Autonomous research agent |
| 6 | Multi-agent company research platform |
| 7 | Self-hosted production RAG service |

---

# Evaluation Dataset Structure

Create one reusable evaluation dataset early.

```text
shared/evaluation/
├── questions.jsonl
├── expected_sources.jsonl
├── expected_answers.jsonl
└── difficult_queries.jsonl
```

Example `questions.jsonl`:

```json
{"id":"q001","question":"What is the refund period?"}
{"id":"q002","question":"Which section describes enterprise refunds?"}
```

Example `expected_sources.jsonl`:

```json
{"id":"q001","document_ids":["refund_policy_2026"]}
{"id":"q002","document_ids":["enterprise_policy_2026"]}
```

This lets you compare Level 1 against Level 2 and prove that advanced techniques actually improved retrieval.

---

# Recommended README Structure for Every Level

Every level README should follow the same format:

```md
# Level X — Name

## Objective

## Architecture

## Concepts

## Technologies

## Folder Structure

## Setup

## Examples

## Experiments

## Evaluation

## Mini Project

## Common Failure Modes

## What I Learned

## Checklist

## Next Level
```

---

# Level Completion Checklist

Before moving to the next folder, complete:

```text
[ ] Read the theory
[ ] Run the minimal implementation
[ ] Build the framework implementation
[ ] Add unit tests
[ ] Run retrieval experiments
[ ] Measure quality
[ ] Document failures
[ ] Build the mini project
[ ] Update README with lessons learned
[ ] Commit results
```

---

# Final Repository Philosophy

The repository should not become a collection of copied notebooks.

Each level should answer three questions:

1. **How does this architecture work?**
2. **When should I use it?**
3. **How do I measure whether it is better?**

Your progression should be:

```text
Naive RAG
   ↓
Understand retrieval
   ↓
Advanced RAG
   ↓
Improve retrieval
   ↓
Modular RAG
   ↓
Choose the correct data source
   ↓
Adaptive RAG
   ↓
Choose the correct strategy dynamically
   ↓
Agentic RAG
   ↓
Let an agent control retrieval
   ↓
Multi-Agent RAG
   ↓
Coordinate specialized agents
   ↓
Production RAG
   ↓
Evaluate + secure + observe + scale
```

The end goal is not simply to know how to connect an LLM to a vector database. The end goal is to be able to **design, diagnose, evaluate, deploy, and operate complete RAG systems**.
