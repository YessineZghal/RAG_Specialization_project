# Missing to Complete — Gap Review Against RAG-Anything (Sept 2026)

> Triggered by a direct request to review **[HKUDS/RAG-Anything](https://github.com/HKUDS/RAG-Anything)**
> (arXiv:2510.12323) — a real, published multimodal RAG framework — and identify what this repo is
> missing relative to it. Same practice as [`GAP_ANALYSIS.md`](./GAP_ANALYSIS.md) and
> [`RAG_TAXONOMY_COVERAGE.md`](./RAG_TAXONOMY_COVERAGE.md): grounded in what the reference project
> actually does (its README and source, not a memory of what a "multimodal RAG" probably has), and
> honest about which gaps are real engineering debt versus a deliberate scope boundary this repo
> has held since Level 1 (Ollama-only, run-it-on-a-laptop, hand-roll the mechanism before adopting
> a framework).
>
> **Update:** a follow-up pass built every item below marked "closed" — real code, real tests, real
> verification against live data, not stubs. Two items were deliberately **not** built, exactly as
> this document originally recommended: audio/video ingestion and the full cross-modal knowledge
> graph, both flagged below as disproportionate scope for this repo. See each level's own README
> (linked in the table) for the full walkthrough and real measured output of what was built.

---

## What RAG-Anything actually is

A five-stage pipeline built **on top of LightRAG** ("simple and fast RAG," not reimplemented):
document parsing (MinerU / Docling / PaddleOCR) → content routed to modality-specific processors
→ a **multimodal knowledge graph** with cross-modal relationships → hybrid vector+graph retrieval
→ query answering, optionally VLM-enhanced. It ingests PDFs, Office formats (DOC/DOCX/PPT/PPTX/
XLS/XLSX via LibreOffice), images, plain text/Markdown, and — as an emerging capability — audio
and video via transcription/keyframe extraction. Four query modes are inherited from LightRAG:
`hybrid` (vector+graph fusion), `local` (local graph traversal), `global` (global knowledge
patterns / community-level), and `naive` (direct similarity search). No quantitative benchmark
numbers are published in the README itself; the technical claims are backed by the arXiv paper.

---

## A. True multimodal knowledge graph — this repo has never built one

Every graph this repo has built (Levels 3, 5, 6, and 9's schema-constrained KAG graph) is
**text-only**: entities and relations extracted from prose, nothing else. RAG-Anything's graph
does **cross-modal entity extraction and relationship mapping** — an image, a table, and the
paragraph that references them can all become linked nodes in the *same* graph, with weighted
relevance preserving document hierarchy.

This repo's own multimodal work (`03-modular-rag/multimodal-rag/`) is disclosed as
**caption-based, not visual** — `image_retrieval.py` and `table_retrieval.py` key off a figure's
`"Figure N:"` caption text, and index that caption alongside the surrounding prose in the *same*
flat vector store as everything else. There is no graph connecting "this image" to "this table" to
"this paragraph" as distinct, typed, linked nodes anywhere in this repo. `GAP_ANALYSIS.md`'s own
Section E already named this precisely ("multimodal knowledge graphs spanning text + visual
regions... is still actively pushing past caption/OCR-based approaches") without a concrete
reference implementation to point at — RAG-Anything is exactly that reference.

**Where it would fit:** a genuine addition to Level 3 (`multimodal-rag/`) or a new Level 9-adjacent
extension, not a small patch — this is a different graph-construction mechanism, not a tweak to
the existing one.

## B. Specialized modality processors — this repo has two of four, and both are shallow

| Modality | RAG-Anything | This repo |
|---|---|---|
| Images | Context-aware VLM captioning + spatial relationship extraction | `vision_embedding.py`: one caption per image via a local vision model, no spatial/relationship extraction — and disclosed as environment-limited (the pulled vision model couldn't load against the running Ollama server; see [Level 3's Common Failure Modes](./03-modular-rag/README.md#common-failure-modes)) |
| Tables | Statistical pattern recognition on tabular data | `table_retrieval.py`: caption-keyed retrieval only, no statistical analysis of the table's actual cell contents |
| Equations | LaTeX parsing + conceptual mapping | **None** — confirmed by grepping the whole repo for `latex`/`equation`: zero matches anywhere |
| Custom/plugin types | Generic handler for arbitrary content types | **None** — no plugin or extension mechanism for a new content type exists in any level |

Equations are a genuine, clean gap: not disclosed anywhere as an intentional boundary (unlike the
vision-model limitation, which is), simply never attempted.

## C. Query-time VLM analysis vs. this repo's index-time-only captioning

RAG-Anything's `vlm_enhanced` query mode calls a vision-language model **at query time** on the
specific images retrieved for that question — the same image can get a different analysis
depending on what's actually being asked about it. This repo's Level 3 only ever describes an
image **once, at ingestion time** (`describe_images()`), and that one fixed caption is all any
future query ever sees. A question asking about a specific visual detail the original caption
didn't happen to mention has no path to get it, in this repo's current design.

**Where it would fit:** Level 3's `multimodal-rag/`, as a genuinely new query-time step — not
achievable by editing `vision_embedding.py`'s existing ingestion-time function.

## D. Document format coverage — this repo handles a fraction of what RAG-Anything does

Confirmed by grep across the whole repo: zero handling anywhere for `.docx`, `.pptx`, `.xlsx`, or
any audio/video format. This repo's actual ingestion surface, across all 9 levels, is: PDF
(Levels 1, 3), plain web pages (Level 3's `web-rag/`), and three SQL databases (Levels 3, 5, 6) —
plus whatever gets pasted as prompt text. RAG-Anything additionally covers the entire Office suite
(via LibreOffice conversion) and an emerging audio/video pipeline (transcription + keyframe
analysis).

**Not necessarily worth closing in full** — Office-format and audio/video ingestion is a real
engineering lift (LibreOffice as a system dependency, an ASR model, video keyframe extraction)
disproportionate to what any single level here is trying to teach. Office-document parsing
specifically (DOCX/PPTX/XLSX → the same chunking pipeline Level 1 already has) is the one piece of
this that's cheap enough to be a legitimate Level 1 or Level 3 addition; audio/video is a heavier
lift closer to `GAP_ANALYSIS.md` Section G's "real but heavier than this repo's laptop philosophy
supports" category.

## E. "Global" / community-level query mode — the same gap this repo already named, now with a second reference implementation

Level 9's own README already compares this repo's graph-rag against Microsoft GraphRAG's
community-detection-plus-summarization layer and names it as something this repo has never built
(`| Query handling | ... | Community-level summary retrieval, tuned for *global* sensemaking
queries |` in the comparison table). RAG-Anything's `global` query mode is the same underlying
idea inherited from LightRAG. This is not a new gap — it is the same one, now confirmed as a
real, load-bearing feature in a second independent published system, not a one-off design choice
Microsoft happened to make.

**Where it would fit:** exactly where `GAP_ANALYSIS.md` Section B already said — a genuine
addition to Level 9's comparison, not a new level.

## F. Two operational features this repo has explicitly disclosed as missing, now with a concrete pattern to follow

- **Reprocessing/reindexing after a model or storage-backend change.** Level 7's own Success
  Criteria already states this plainly: *"How do you deploy a new embedding model without
  corrupting the index? Not exercised in this level."* RAG-Anything's "force multimodal
  reprocessing" flag for storage-backend migrations is a concrete, real pattern for exactly this
  operational gap — worth closing in Level 7 specifically, since that's the level with an actual
  live index to migrate.
- **Batch ingestion.** This repo's only ingestion path with a defined API (`07-production-rag`'s
  `POST /admin/ingest`) takes one document per call — confirmed by reading its handler directly.
  RAG-Anything supports batch processing of multiple documents in one call. A real, small,
  closeable gap in the same file.

## G. Direct content-list insertion (parser-bypass)

RAG-Anything accepts pre-extracted content directly, skipping its own parsing stage entirely, for
callers who already have structured content from elsewhere. This repo's ingestion functions always
parse from a raw source (PDF bytes, a URL, a DB connection) — there is no "I already have the
chunks, just index them" entry point anywhere. A minor but real API-flexibility gap, most relevant
to Level 7's `/admin/ingest` if it were extended.

---

## What is *not* a gap — deliberate scope differences, not oversights

- **No hosted-model dependency.** RAG-Anything's VLM integration is a user-provided callback, and
  its own examples call OpenAI's `gpt-4o` — this repo's Ollama-only boundary (stated at every
  level since Level 1) is a considered choice, not something RAG-Anything does "better." The two
  projects are optimizing for different things: RAG-Anything for capability breadth, this repo for
  "understand the mechanism, run it for real on a laptop with one local model."
- **No published quantitative benchmark to cite.** RAG-Anything's own README states no numbers —
  they live in the arXiv paper this document doesn't have access to verify directly, so none are
  quoted here. Consistent with this repo's own standing rule: never reuse a number that wasn't
  independently reproduced (see every level's Evaluation section, and `GAP_ANALYSIS.md`'s explicit
  refusal to reuse the KAG paper's own reported 19.6%/33.5% figures in Level 9).
- **Building on LightRAG rather than from scratch** is RAG-Anything's own choice, the opposite of
  this repo's "hand-roll the mechanism before adopting a framework" philosophy (stated explicitly
  in Levels 3, 5, 6, 8, and 9's own READMEs) — not something to imitate, a different pedagogical
  goal entirely.

---

## Summary table

| Gap | Status | Where it landed | Real, verified result |
|---|---|---|---|
| Cross-modal knowledge graph | **Not built** (as recommended) | — | Large, disproportionate lift — left as a real, disclosed gap, not attempted |
| Equation/LaTeX handling | **Closed** | [Level 3 `multimodal-rag/equation_retrieval.py`](./03-modular-rag/README.md#three-additions-from-the-rag-anything-gap-review) | 3 real equations found in the actual PDF, 7 false positives correctly rejected, real symbol-gloss concepts attached |
| Statistical table analysis | **Closed** | [Level 3 `multimodal-rag/table_statistics.py`](./03-modular-rag/README.md#three-additions-from-the-rag-anything-gap-review) | Real min/max/mean over the actual paper's BLEU/F1 scores across its 4 real tables |
| Query-time VLM re-analysis | **Closed** | [Level 3 `multimodal-rag/query_time_vision.py`](./03-modular-rag/README.md#three-additions-from-the-rag-anything-gap-review) | Real dispatch logic verified offline; inherits the same disclosed vision-model environment limitation as `vision_embedding.py` |
| Office-format ingestion (DOCX/PPTX/XLSX) | **Closed** | [Level 1 `src/ingest.py`](./01-naive-rag/README.md) | 3 real committed Office sample files, all three formats parse correctly end-to-end, including a real docx table |
| Audio/video ingestion | **Not built** (as recommended) | — | Heavier than this repo's laptop philosophy supports — left as a real, disclosed gap |
| "Global"/community-level query mode | **Closed** | [Level 9 `indexing/community_summary.py` + `reasoning-engine/global_op.py`](./09-knowledge-augmented-generation/README.md#a-fifth-operator-from-the-rag-anything-gap-review-global) | 11 real communities detected in the actual cached KAG graph, each with a real, genuinely on-topic LLM-generated summary |
| Reindex-on-model-change | **Closed** | [Level 7 `POST /admin/reindex`](./07-production-rag/README.md#two-additions-from-the-rag-anything-gap-review) | Verified live against the real Qdrant stack: original collection's point count unchanged after a non-activated reindex; `/health` and `/query` both correctly reflected the swap after activation |
| Batch ingestion API | **Closed** | [Level 7 `POST /admin/ingest/batch`](./07-production-rag/README.md#two-additions-from-the-rag-anything-gap-review) | Verified live: one batched embed call and one batched upsert call for a real multi-document request |
| Direct content-list insertion (parser bypass) | **Closed** (folded into batch ingestion above) | Level 7 `/admin/ingest/batch` | Each item in a batch call is already-extracted text with no parsing step |

49 new offline tests were added across Levels 1, 3, 7, and 9 for everything marked "Closed" above
— full repository suite: **523 tests passing** (474 right before this pass, which already
included Level 8's own separately-added mini project; 461 right after Level 9 was first built,
before either addition).

---

## Sources

- [RAG-Anything (HKUDS)](https://github.com/HKUDS/RAG-Anything) — repository and README, fetched directly
- [RAG-Anything technical report (arXiv:2510.12323)](https://arxiv.org/abs/2510.12323) — referenced by the README, not independently verified for the specific numbers it reports
- [`GAP_ANALYSIS.md`](./GAP_ANALYSIS.md) — this repo's own prior gap analysis, cross-referenced above (Sections B and E already named two of these gaps independently)
- [`03-modular-rag/README.md`](./03-modular-rag/README.md#common-failure-modes) — this repo's own disclosed multimodal limitation
- [`07-production-rag/README.md`](./07-production-rag/README.md#success-criteria) — this repo's own disclosed reindexing gap
