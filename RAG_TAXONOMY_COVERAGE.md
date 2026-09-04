# RAG Taxonomy Coverage — Cross-Referenced Against the Actual Code

> A follow-up request asked to check a long, explicitly-named list of ~45 "RAG types" against this
> repo directly. This document is that audit: every item checked against the **real files**, not
> against memory of what a level's README claims — a distinction that mattered, see the
> correction below. Related: [`GAP_ANALYSIS.md`](./GAP_ANALYSIS.md) (the earlier, narrower deep
> search that produced Levels 8-9) and [Level 8](./08-reasoning-strategies/README.md) /
> [Level 9](./09-knowledge-augmented-generation/README.md) themselves. Every gap below is turned
> into a concrete, ordered build checklist in **[`TASK.md`](./TASK.md)**.
>
> [Back to roadmap](./README.md)

---

## A correction, made honestly

`GAP_ANALYSIS.md` originally claimed HyDE was uncovered. **That was wrong** — this repo already
implements it for real: `02-advanced-rag/query-transformations/hyde.py`, executed in
`notebooks/05_query_transformations.ipynb`, with a documented real finding ("HyDE solves a query
everything else misses"). That error came from reasoning about what a level "probably" has
instead of re-checking the actual files before writing. This document exists specifically to not
repeat that: every verdict below was checked with `grep`/`cat` against the real repo, not assumed.
`GAP_ANALYSIS.md` has been corrected to match.

---

## Method

For each named technique: does a real, checked-in file implement it, in which level, and — if
genuinely absent — which existing level is the natural place to add it (per the request: extend
an existing level where one fits, rather than manufacturing a new level for everything).

Verdicts: **✅ covered** (real code, and usually a real executed notebook) · **🔶 partial**
(covered in spirit by an adjacent pattern, or a narrower version exists) · **❌ gap** (nothing in
the repo does this).

---

## Retrieval mechanism

| Type | Verdict | Where |
|---|---|---|
| Naive / Vector / Semantic RAG | ✅ | Level 1 — dense embeddings, top-k vector search |
| Sparse / Keyword RAG (BM25) | ✅ | Level 2 — `hybrid-search/` |
| Hybrid RAG | ✅ | Level 2 — BM25 + dense fused via RRF |
| Reranking RAG | ✅ | Level 2 — `reranking/cross_encoder.py`, `bge_reranker.py` |
| Ensemble RAG | ✅ | Level 2's hybrid fusion + Level 3's `multi-retriever/retriever_fusion.py` (RRF across named collections, e.g. document chunks + graph text at once) |
| **Late-interaction retrieval (ColBERT)** | **❌ gap** | Token-level embeddings + MaxSim scoring instead of one vector per chunk — every level uses single-vector dense retrieval. **→ add to Level 2**, alongside the existing `hybrid-search/` and `retrieval/` folders, as a third retrieval mechanism to compare against dense/BM25/hybrid on the same real eval set Level 2 already tracks. |

## Query processing

| Type | Verdict | Where |
|---|---|---|
| Query Rewriting RAG | ✅ | Level 2 — `query-transformations/query_rewrite.py` (+ Level 4/5's retry-rewriting on insufficient evidence) |
| Multi-Query / Query Expansion RAG | ✅ | Level 2 — `query-transformations/multi_query.py` (+ Level 4's "complex → multi-query fusion, top_k=8") |
| HyDE RAG | ✅ | Level 2 — `query-transformations/hyde.py` (see correction above) |
| Step-Back Prompting *(not in the source list, found while checking)* | ✅ | Level 2 — `query-transformations/step_back.py` |
| **Self-Query RAG** (LLM auto-derives semantic query + structured filters from one NL question) | **❌ gap** | Level 2's `metadata-filtering/filters.py` only supports a manually-written predicate — nothing extracts filters from the question itself. **→ add to Level 2**, as `metadata-filtering/self_query.py`: an LLM call that splits "AI papers after 2024 by OpenAI" into `{semantic: "AI papers", filters: {year: ">2024", org: "OpenAI"}}`, feeding the existing `filtered_search()`. |
| **Conversational RAG** (multi-turn query rewriting that resolves pronouns/follow-ups against chat history) | **❌ gap** | Checked specifically — Level 3's "coreference" mention is about *graph entity* coreference ("Jakob" vs "Jakob Uszkoreit"), a different problem. No level rewrites a query using prior conversation turns. **→ add to Level 2**, as `query-transformations/conversational_rewrite.py`, the same category as the existing `query_rewrite.py` (an LLM rewrites the query before retrieval), just conditioned on chat history instead of on retrieval-quality feedback. |

## Document / index structure

| Type | Verdict | Where |
|---|---|---|
| Parent-Child / Small-to-Big RAG | ✅ | Level 2 — `chunking/parent_child.py` |
| Metadata-filtered RAG | ✅ | Level 2 — `metadata-filtering/filters.py` (post-filter; Level 7's Qdrant ACL filter is the pre-filter version of the same idea, at index level) |
| Context compression | ✅ | Level 2 — `context-compression/compressor.py` (generation-time trimming) |
| **Hierarchical RAG** (index at doc → section → paragraph → chunk, retrieve progressively) | **❌ gap** | No level indexes at more than one granularity. **→ add to Level 2's `chunking/`**, as `hierarchical.py`. |
| **RAPTOR** (recursive clustering + summarization into a retrieval tree) | **❌ gap** | Same folder as above — **→ Level 2's `chunking/`**, as `raptor.py`. Natural to build alongside the Hierarchical addition; they solve the same underlying problem (retrieve at the right granularity) at different levels of sophistication. |
| **Contextual Retrieval** (Anthropic-style: prepend an LLM-written "where this chunk sits in its document" note *before* embedding) | **❌ gap** | Distinct from `context-compression/` (that trims context at generation time; this enriches it at indexing time — checked, nothing does the latter). **→ add to Level 2's `chunking/`**, as `contextual_enrichment.py`, run as a preprocessing step before `embed.py`. |
| **Long-Context RAG** (skip chunking/retrieval entirely; feed a small-enough corpus straight into a large context window) | **❌ gap** | No level runs this comparison. **→ add to Level 2's `evaluation/`** as an explicit baseline arm — directly answers "when does retrieval even help" on the same real eval set Level 2 already scores Recall@K/MRR/NDCG against. |

## Reasoning / control flow

| Type | Verdict | Where |
|---|---|---|
| Recursive RAG (iterative retrieve → reason → retrieve again) | ✅ | Level 5 — `iterative-retrieval/loop.py` |
| Multi-Hop RAG | ✅ | Level 4 — `multi-hop-rag/` (+ Level 5's iterative loop for the agentic version) |
| Corrective RAG (CRAG) | ✅ | Level 4 |
| Self-RAG | ✅ | Level 4 |
| Adaptive RAG | ✅ | Level 4 (the whole level) |
| Agentic RAG | ✅ | Level 5 (the whole level) |
| Multi-Agent RAG | ✅ | Level 6 (the whole level) |
| Verification RAG | ✅ | Level 5 — `verification/source_checker.py`, `answer_verifier.py`; Level 6 — verification agent |

Chain-/Tree-/Graph-of-Thought are not in the source list for this pass but were already covered
by the prior deep search — see [Level 8](./08-reasoning-strategies/README.md).

## Knowledge representation

| Type | Verdict | Where |
|---|---|---|
| Graph RAG / Knowledge-Graph RAG | ✅ | Levels 3, 5, 6 — LLM-extracted entity/relation graph, `networkx`, fact lookup (a real, disclosed simplification — see [Level 9](./09-knowledge-augmented-generation/README.md) for the rigorous, schema-constrained evolution of the same idea) |

## Data source

| Type | Verdict | Where |
|---|---|---|
| SQL / Database RAG | ✅ | Level 3 — `sql-rag/` (reused in Levels 5, 6 against two more real SQL databases) |
| API RAG | ✅ | Level 3 — `api-rag/` (arXiv's public API) |
| Web RAG | ✅ | Levels 3-6 — live DuckDuckGo search + page extraction |
| Router RAG | ✅ | Level 3 — `routing/` (rule- and LLM-based) |
| **Real-Time RAG** (fresh, frequently-changing data) | 🔶 **partial** | Web-rag and api-rag already hit a *live* external source on every query, which functionally satisfies "always current" — but that's pull-based (fresh per request), not push-based (a persistent feed/webhook updating the index as new data arrives). The gap is narrow: a subscription-driven ingestion pipeline, not a new retrieval pattern. Lower priority; if built, **Level 7** (`retrieval-infrastructure/` already owns live index writes). |
| Federated RAG (query fragmented, non-centralized corpora across systems/orgs) | ❌ gap, **logged as backlog, not recommended to build here** | Already flagged in [`GAP_ANALYSIS.md` § G](./GAP_ANALYSIS.md#g-frontier-techniques--real-but-a-heavier-lift-than-this-repos-run-it-for-real-on-a-laptop-philosophy-supports) — a distributed-systems problem more than a RAG-architecture one, and this repo's Level 3/6 routing across SQL+graph+web+API from one process already covers the *architectural* lesson (route to the right backend) without needing real cross-organization infrastructure. |

## Modality

| Type | Verdict | Where |
|---|---|---|
| Document RAG | ✅ | Every level, broadly |
| Multimodal RAG | 🔶 **partial, already disclosed** | Level 3 — `multimodal-rag/`, explicitly caption-based, not visual (its own README names this as "this level's one genuine simplification, not a hidden one") |
| **Vision RAG** (true pixel-level vision-language embeddings, e.g. CLIP, so uncaptioned images/scanned pages are retrievable) | **❌ gap** | This is Level 3's own already-named limitation. **→ add to Level 3's `multimodal-rag/`**, as `vision_embedding.py`, the direct fix for the caption-dependency problem its own `image_retrieval.py` docstring describes. |
| Table RAG (dedicated semantic + schema-aware retrieval over ad-hoc CSV/Excel) | 🔶 partial | Split across two existing patterns — Level 3's caption-based `table_retrieval.py` (PDF tables) and Level 3/5/6's SQL-RAG (structured databases) — but no single module does schema-inference-plus-semantic-search over an arbitrary spreadsheet. Low priority; the two adjacent patterns cover the real use cases this repo's datasets present. |

## User-specific / enterprise

| Type | Verdict | Where |
|---|---|---|
| Memory RAG | ✅ | Level 5 — `memory/short_term.py`, `long_term.py` |
| Permission-aware / Secure RAG | ✅ | Level 7 — `security/` (auth, permissions, document ACL, prompt-injection defense) |
| Citation-Aware RAG | 🔶 partial | Level 7 already returns real `Source(doc_id, title, score)` objects per answer, and its faithfulness eval checks *per-claim* support against context (`production_eval/ragas_eval.py`) — real evidence-checking exists, just not strict inline citation markers (`[1]`, `[2]`) inserted into the answer text itself. Low priority. |
| **Personalized RAG** (ranking/content shaped by user preference or role, not just binary visibility) | **❌ gap** | Level 7's `security/permissions.py` and `document_acl.py` decide *whether* a user can see a document, never *how retrieval ranks* differently per user beyond that. **→ add to Level 7's `security/`**, as `personalization.py` — same query, two different `user_id`s, genuinely different (not just filtered) ranked results. |
| **Temporal RAG** (filter/retrieve by document validity date — "what was true as of date X") | **❌ gap** | A specific, valuable shape of metadata filtering nothing currently demonstrates. **→ add to Level 2's `metadata-filtering/`**, as `temporal.py` (same mechanism as `filters.py`, a date-range predicate) — with a one-line operational note in **Level 7** that a real deployment would need document *versions*, not just a `date` field, to answer this against a changing corpus. |

## Production architecture concepts

| Concept | Verdict | Where |
|---|---|---|
| Production-grade RAG architecture | ✅ | Level 7 (the whole level) |
| Advanced Agentic Enterprise RAG | ✅ | Level 6 (agents) + Level 7 (production hardening) combined |
| **Streaming responses** *(not in the source list — a self-consistency check against this repo's own original plan)* | **❌ gap** | `07-production-rag`'s original scaffold README explicitly named "streaming responses" under API concepts to cover; the real build never implemented one (`grep -rn "StreamingResponse\|text/event-stream"` returns nothing in the whole repo). **→ add to Level 7's `api/`** — stream the generation token-by-token over the existing `/query` route (or a new `/query/stream`), directly relevant given Level 7's own load test found generation latency is the dominant cost users wait through. |

---

## Summary — what actually gets added, and where

**7 additions to [Level 2](./02-advanced-rag/README.md)** (all fit its existing folders, no new
infrastructure): ColBERT/late-interaction (`retrieval/` or `hybrid-search/`), self-query
(`metadata-filtering/`), conversational query rewriting (`query-transformations/`), hierarchical
indexing + RAPTOR + contextual retrieval (`chunking/`), and a long-context baseline
(`evaluation/`).

**2 additions to existing levels**: true vision embeddings → [Level 3](./03-modular-rag/README.md)
(`multimodal-rag/`, fixing its own disclosed gap); personalization → [Level 7](./07-production-rag/README.md)
(`security/`).

**1 cross-cutting operational note**: temporal filtering's mechanism belongs in Level 2, but
using it against a *live, changing* corpus is a Level 7 concern — noted in both places.

**1 self-consistency fix**: streaming responses, planned in Level 7's original scaffold, never
built — → [Level 7](./07-production-rag/README.md) (`api/`).

**Nothing here needs a new numbered level.** Every genuine gap found by this pass extends a level
that already exists and already has the right folder for it — a different outcome than
`GAP_ANALYSIS.md`'s CoT/ToT/GoT and KAG findings, which were distinct enough architecturally to
warrant Levels 8 and 9. Both documents are accurate simultaneously: some gaps are new axes
(new levels), most are missing techniques along axes this repo already built (extend the level).

None of the additions above are implemented yet — this document is the audit and the plan, not
the build, consistent with how every other addition to this repo started. See
[`TASK.md`](./TASK.md) for these turned into a concrete, ordered, file-by-file checklist.
