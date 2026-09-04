# Gap Analysis — What This Repo Covers, What It Doesn't (Sept 2026)

> Triggered by a direct request to review every concept in the repo and deep-search for
> architectures or techniques the 7-level roadmap doesn't cover. This is a **research document**,
> grounded in real, cited sources (web search run September 2026) — not a memory dump. Two of the
> gaps found are significant enough to get their own new level docs:
> **[Level 8 — Reasoning Strategies](./08-reasoning-strategies/README.md)** (Chain-/Tree-/Graph-of-Thought)
> and **[Level 9 — Knowledge-Augmented Generation](./09-knowledge-augmented-generation/README.md)**
> (KAG). Both **started structure-only, like every level did before it got built, and are now
> fully implemented and executed** — see each one's own Evaluation section for its real, measured
> result. This document is the "why," those two are (now) the "what happened."
>
> A follow-up request asked to check a much longer, explicitly-named list of ~45 "RAG types"
> against this repo directly — see **[`RAG_TAXONOMY_COVERAGE.md`](./RAG_TAXONOMY_COVERAGE.md)**
> for that full item-by-item audit (and the HyDE correction above, which it caught). Every finding
> from both documents is turned into a concrete, ordered build checklist in
> **[`TASK.md`](./TASK.md)**.
>
> [Back to roadmap](./README.md)

---

## 1. What the existing 7 levels actually cover

A quick, honest inventory (see each level's own README for the real, measured results):

| Level | Retrieval architecture | Reasoning / control flow |
|---|---|---|
| 1 — Naive RAG | Flat chunking, dense embeddings, top-k vector search | None — single retrieve-then-generate pass |
| 2 — Advanced RAG | + BM25/dense hybrid, cross-encoder reranking | None |
| 3 — Modular RAG | + routing to SQL / graph / web / API / multimodal, RRF fusion | Rule- or LLM-based router picks one backend |
| 4 — Adaptive RAG | + dynamic top-k, query classification | + CRAG (grade-then-retry), Self-RAG (self-critique), multi-hop decomposition |
| 5 — Agentic RAG | + 5 callable tools | + hand-rolled ReAct loop, planning, memory, iterative retrieval, reflection |
| 6 — Multi-Agent RAG | Same tool surface, split across specialists | + supervisor routing, parallel/sequential workflows, verification, synthesis |
| 7 — Production RAG | + Qdrant/Postgres/Redis, ACL pre-filtering | + auth, caching, prompt-injection defense, observability, hand-rolled Ragas eval |

This is a real, coherent progression — but it optimizes one axis (**how retrieval is structured
and orchestrated**) and mostly treats **how the model reasons over what it retrieved** as a
solved, single-shot step ("stuff context into a prompt, generate once"). It also treats
**knowledge representation** as either flat chunks or a simple LLM-extracted entity/relation
graph (Levels 3, 5, 6) — never a graph disciplined by an explicit schema or a logical solver.
Those two blind spots are exactly where the deep search below turned up the most.

---

## 2. Deep search findings, by category

### A. Reasoning strategies — the axis this repo doesn't touch → [Level 8](./08-reasoning-strategies/README.md)

| Technique | What it is | Relevant here because |
|---|---|---|
| **Chain-of-Thought (CoT)** | Prompt the model to produce intermediate reasoning steps before its final answer, instead of answering directly. | Every generation step in Levels 1-7 is effectively zero-shot — none of them explicitly elicit intermediate reasoning over the retrieved context. |
| **Tree-of-Thought (ToT)** ([Yao et al., 2023](https://arxiv.org/abs/2305.10601)) | Model reasoning as a **search over a tree** of candidate "thoughts," each scored by a state evaluator, with backtracking when a branch looks unpromising. Explicitly designed to fix CoT's inability to backtrack. | None of Levels 4-6's retry/reflection loops actually explore *multiple candidate reasoning paths in parallel* — they retry linearly (CRAG's rewrite-and-retry, the ReAct loop's next action) rather than branching and comparing. |
| **Graph-of-Thoughts (GoT)** ([Besta et al., AAAI 2024](https://dl.acm.org/doi/10.1609/aaai.v38i16.29720)) | Generalizes ToT further: reasoning steps are nodes in an arbitrary **directed graph**, so branches can also *merge* and *aggregate*, not just split. | The natural next step past this repo's Level 6 multi-agent synthesis pattern — several parallel findings merging into one, but generalized to reasoning steps instead of whole agents. |
| **Hierarchical GoT (HGoT)** and **Knowledge-GoT (KGoT)** — 2025-2026 extensions | HGoT decomposes a query into sub-questions across a multilayered graph with citation-aware voting, explicitly for **retrieval-augmented factuality**. KGoT persists a tool-enhanced knowledge graph across reasoning steps, integrating live retrieval and code execution. | These are the RAG-specific descendants of GoT — closer to what a "Level 8" should actually teach than the generic algorithmic-puzzle framing most ToT/GoT demos use. |
| **Think-on-Graph 2.0** | Knowledge-guided retrieval-augmented reasoning directly over a graph, for "deep and faithful" LLM reasoning. | Bridges category A (reasoning strategies) and category B (KAG) below — worth naming in both. |

**A directly relevant tension, not hidden**: every one of these techniques multiplies LLM calls
(ToT/GoT can cost 5-30x a single CoT pass). Level 7's own load test found **generation latency is
already this repo's binding bottleneck** (24s median at just 5 concurrent users, CPU-bound
Ollama). Layering ToT/GoT on top of that without also solving Level 7's inference-scaling problem
would make it dramatically worse — a real trade-off Level 8 should measure, not assume away.

### B. Knowledge-graph-centric architectures beyond this repo's simple graph-rag → [Level 9](./09-knowledge-augmented-generation/README.md)

| Framework | What it actually does | vs. this repo's Level 3/5/6 graph-rag |
|---|---|---|
| **KAG** ([Liang et al., 2024](https://arxiv.org/abs/2409.13731), Ant Group + Zhejiang University, ACM WWW 2025) | A **schema-constrained** knowledge graph (built on Ant's OpenSPG engine) with **mutual indexing** between KG nodes and source chunks, queried through a **logical-form-guided hybrid reasoning engine** that routes across four operator types: retrieval, KG reasoning, language reasoning, and numerical calculation. Reports +19.6% F1 on 2WikiMultihopQA and +33.5% on HotpotQA over strong RAG baselines; deployed in Ant's own E-Government (91.6% precision) and E-Health Q&A products. | This repo's graph-rag is an **unconstrained** LLM entity/relation extraction into a plain `networkx` graph, queried by simple fact lookup — no schema, no logical-form solver, no numerical/temporal reasoning path, and (per [Level 3's own README](./03-modular-rag/README.md#common-failure-modes)) no coreference resolution. KAG is the rigorous version of the same idea. |
| **Microsoft GraphRAG** | Extracts entities/relations, runs **community detection** (Leiden algorithm) over the graph, then generates natural-language summaries per community for hierarchical, **global sensemaking** queries ("what are the main themes in this corpus?") — a different problem than the point-lookup questions this repo's graph-rag answers. | Complementary, not redundant: this repo's graph-rag and KAG both answer specific-fact questions; Microsoft GraphRAG answers corpus-wide summarization questions neither currently attempts. |
| **LightRAG** | A lighter-weight alternative to Microsoft GraphRAG: **dual-level retrieval** (low-level specific-entity keys + high-level topic keys) without the community-summarization layer — cheaper and faster, closer in spirit to this repo's existing graph-rag but more structured. | The natural "cheaper middle ground" to mention alongside KAG in Level 9, for cost-conscious readers. |

### C. Retrieval-quality techniques — corrected after a follow-up taxonomy review, then built

> **Correction (original):** this section originally listed HyDE as uncovered. That was wrong —
> Level 2 already implements it for real (`02-advanced-rag/query-transformations/hyde.py`),
> executed in `notebooks/05_query_transformations.ipynb` with a documented real finding ("HyDE
> solves a query everything else misses"). Level 2 also already covers parent-child/small-to-big
> chunking, multi-query, query rewriting, step-back prompting, metadata filtering, and context
> compression — a fuller, evidence-checked cross-reference against a much longer list of named RAG
> "types" is in **[`RAG_TAXONOMY_COVERAGE.md`](./RAG_TAXONOMY_COVERAGE.md)**.
>
> **Update (since built):** the three techniques below — genuinely absent when this document was
> first written, verified by directly grepping the repo, not assumed — have since been implemented
> as part of Level 2's "eight additions from the taxonomy review" pass:
> `chunking/raptor.py`, `retrieval/late_interaction.py`, `chunking/contextual_enrichment.py`, all
> real, tested (17 passing tests across the three), and run against real data with documented
> findings in [Level 2's own README](./02-advanced-rag/README.md#eight-additions-from-the-taxonomy-review)
> — including a genuine limitation RAPTOR's clustering hit on a same-domain sample. This section is
> kept below as a historical record of what the gap analysis originally found, not as a current
> "still missing" list — check `02-advanced-rag/README.md`'s own Checklist for current status, the
> same lesson this repo's READMEs keep landing on: check the actual code, don't reason from memory
> of what a repo "probably" has.

| Technique | What it is | Real measured effect (from research) |
|---|---|---|
| **RAPTOR** | Recursively cluster and summarize chunks into a tree, so retrieval can pull a fine-grained chunk *or* a high-level summary depending on the question's scope. | +20% absolute accuracy on QuALITY with GPT-4, per the technique's own reported benchmarks. |
| **ColBERT / late interaction** | Token-level embeddings compared via "late interaction" (MaxSim) instead of one dense vector per chunk — finer-grained matching than this repo's single-vector dense retrieval (Levels 1-7 all use one embedding per chunk). | Consistently outperforms both BM25 and single-vector dense retrieval in head-to-head benchmarks. |
| **Contextual Retrieval** (Anthropic) | Prepend an LLM-generated one- or two-sentence summary of *where a chunk sits in its source document* before embedding/indexing it — fixes the "chunk lost its surrounding context" problem this repo's flat chunking (every level) never addresses. Distinct from Level 2's `context-compression/`, which trims context at *generation* time, not indexing time. | Reported 67% reduction in retrieval failure rate when combined with reranking. |

All three were real, cheap-to-prototype additions to **Level 2** specifically — no new
infrastructure needed beyond what it already had — and that is exactly what happened.

### D. A technique that directly targets Level 7's own measured bottleneck

| Technique | What it is | Why this repo should care |
|---|---|---|
| **Speculative RAG / predictive prefetching** | Start retrieval (or even draft generation) speculatively, before the full query is finalized, or prefetch likely-next context asynchronously to hide latency. | Level 7's load test found generation, not infrastructure, is the bottleneck (16-40s at 5 concurrent users). Speculative/prefetch techniques are a genuine mitigation path Level 7 doesn't explore — its own fix (batched GPU inference via vLLM) was reference-only, never run. Worth flagging as the most *directly evidenced* gap in the whole repo. |

### E. Multimodal RAG beyond captions — confirms a limitation this repo already disclosed

Level 3's own README already flags its multimodal retrieval as **caption-based, not visual**
(keys off a figure's `"Figure N:"` caption text, not the image's pixels) and names true
CLIP-style visual embedding as a deliberate scope boundary. Deep search confirms this is a live
research area, not a solved problem this repo happened to skip: current 2025-2026 work
(multimodal knowledge graphs spanning text + visual regions, video-temporal retrieval,
robotic-affordance retrieval) is still actively pushing past caption/OCR-based approaches. No new
level proposed here — Level 3's own disclosure already covers this honestly — but it's worth
naming as a confirmed, not just suspected, gap.

### F. Security beyond prompt-injection — a different attack surface than Level 7 covers

Level 7 defends against prompt injection **in a live query or retrieved passage at query time**
(`security/prompt_injection.py`). A different, related attack defends against **corpus
poisoning** — an attacker planting malicious documents into the index *ahead of time* so they get
retrieved and trusted later. Recent work (RAGPart/RAGMask-style defenses) targets this at the
retrieval layer specifically. Not proposed as a new level here (narrower, security-research scope
rather than an architecture pattern), but worth a line in Level 7's own future-work list.

### G. Frontier techniques — real, but a heavier lift than this repo's "run it for real on a laptop" philosophy supports

- **RL-trained retrieval policies** (e.g., Search-R1-style agents that *learn*, via reinforcement
  learning, when and what to search, rather than following a hand-rolled ReAct prompt like Level
  5's agent) — genuinely different paradigm (learned policy vs. prompted policy), but needs RL
  training infrastructure this repo has deliberately avoided everywhere else (Ollama + hand-rolled
  logic, no training runs). Logged here, not built.
- **Federated RAG** (retrieval across fragmented, non-centralized corpora/edge nodes) — real,
  but an infrastructure/systems problem more than a RAG-architecture one. Logged, not built.

---

## 3. What actually became new level docs, and why

Of everything above, two categories earned a full structure-only level doc, matching exactly how
every other level in this repo started (folders + README + mermaid, no implementation until it's
actually studied):

1. **[Level 8 — Reasoning Strategies](./08-reasoning-strategies/README.md)** (CoT / ToT / GoT,
   plus the RAG-specific HGoT/KGoT/Think-on-Graph 2.0 extensions) — because it's an entire axis
   ("how does the model reason over what it retrieved") this repo's 7 levels never touch, it's
   cheap to prototype (no new infrastructure), and it has a real, evidenced tension with Level 7's
   own measured latency bottleneck worth teaching explicitly.
2. **[Level 9 — Knowledge-Augmented Generation (KAG)](./09-knowledge-augmented-generation/README.md)**
   — because it's a rigorous, published, benchmarked evolution of exactly the graph-rag pattern
   this repo already built three times (Levels 3, 5, 6) in a deliberately simplified form, making
   a direct, honest "simple graph-rag vs. schema-constrained KAG-style" comparison possible on the
   same kind of question.

Category C (RAPTOR, ColBERT, Contextual Retrieval) and the rest are logged here as backlog
candidates rather than built out — genuinely worth adding to Level 2 specifically (see
[`RAG_TAXONOMY_COVERAGE.md`](./RAG_TAXONOMY_COVERAGE.md) for exactly where), but this pass
focused on the two gaps with the clearest architectural distinctiveness and the most direct
connection to what this repo has already measured.

---

## Sources

- [Tree of Thoughts: Deliberate Problem Solving with Large Language Models (Yao et al., 2023)](https://arxiv.org/abs/2305.10601)
- [Graph of Thoughts: Solving Elaborate Problems with Large Language Models (Besta et al., AAAI 2024)](https://dl.acm.org/doi/10.1609/aaai.v38i16.29720)
- [Graph-of-Thoughts overview, HGoT/KGoT/Think-on-Graph 2.0 (Emergent Mind)](https://www.emergentmind.com/topics/graph-of-thoughts-got-4b78edd3-5791-45f5-81f7-74eb602e13fc)
- [KAG: Boosting LLMs in Professional Domains via Knowledge Augmented Generation (Liang et al., 2024)](https://arxiv.org/abs/2409.13731)
- [KAG (Knowledge Augmented Generation): A Step Beyond RAG](https://umeey.medium.com/kag-knowledge-augmented-generation-a-step-beyond-rag-a86925694a01)
- [RAG vs KAG: Comparison and Differences in GenAI Knowledge Systems](https://www.plainconcepts.com/rag-vs-kag/)
- [How Would Microsoft GraphRAG Work Alongside a Graph Database?](https://memgraph.com/blog/how-microsoft-graphrag-works-with-graph-databases)
- [Understanding LightRAG: A New Era in RAG](https://blog.nashtechglobal.com/understanding-lightrag-a-new-era-in-rag/)
- [Under the covers with LightRAG: Extraction (Neo4j)](https://neo4j.com/blog/developer/under-the-covers-with-lightrag-extraction/)
- [20 Advanced RAG Types to Know in 2026 (Turing Post)](https://www.turingpost.com/p/ragtypes)
- [Agentic Retrieval-Augmented Generation: A Survey on Agentic RAG (Singh et al., 2501.09136)](https://arxiv.org/abs/2501.09136)
- [Comprehensive Comparison of RAG Methods Across Multi-Domain Conversational QA](https://arxiv.org/pdf/2602.09552)
- [From BM25 to Corrective RAG: Benchmarking Retrieval Strategies](https://arxiv.org/pdf/2604.01733)
