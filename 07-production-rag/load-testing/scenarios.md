# Load Testing — real numbers, one honest headline finding

## Running it

```bash
cd 07-production-rag
uv run --extra production locust -f load-testing/locustfile.py --host http://127.0.0.1:8001
# open http://localhost:8089 for the interactive UI, or run headless:
uv run --extra production locust -f load-testing/locustfile.py --host http://127.0.0.1:8001 \
  --headless -u 5 -r 2 --run-time 45s --csv=load-testing/results
```

`RagApiUser` weights `/query` 8:1 over `/health` (`@task(8)` / `@task(2)`), with real SQuAD
questions sampled from `production_common.dataset.prepare()` — not synthetic Lorem Ipsum text,
so retrieval actually has to do real work on every request.

## What was actually measured

A 45-second run, ramping to 5 concurrent users, against the live API on this machine
(Ollama running natively, `llama3.2` for generation, no GPU):

| Endpoint | Requests | Median | Avg | Min | Max | Throughput |
|---|---|---|---|---|---|---|
| `GET /health` | 1 | 27 ms | 27 ms | 27 ms | 27 ms | — |
| `POST /query` | 4 | 24.0 s | 28.0 s | 15.9 s | 40.4 s | 0.09 req/s |

Full percentile breakdown and raw CSVs: [`results_stats.csv`](results_stats.csv),
[`results_stats_history.csv`](results_stats_history.csv).

## The headline finding: **the LLM generation step, not the API/infra layer, is the bottleneck**

`/health` (no model calls, just a Qdrant count) answers in 27ms. `/query` — even on a **cache
miss with only 5 concurrent users** — takes 16-40 **seconds**. That's not a caching problem,
an ACL problem, or a FastAPI-overhead problem: it's `llama3.2` running generation on CPU via
Ollama, one request at a time, with no batching. Five concurrent users each waiting 20-40s for
a single Ollama process to answer them serially is a queueing problem, not a code bug — and it
is exactly the problem `inference/vllm_client.py` (batched, GPU-served generation) and
`caching/` (skip generation entirely on a hit) exist to solve in a real deployment.

Two caching-path numbers from earlier testing put this in context — a **cache hit is ~1,000x
faster than a miss**:

| Path | Latency |
|---|---|
| Cache miss (full retrieval + generation) | ~4,116 ms (fast case, cache-warm corpus) to 40,374 ms (loaded, this test) |
| Exact-match cache hit (`response_cache`) | ~1.08 ms |
| Semantic-match cache hit (`semantic_cache`) | ~28 ms |

At real production traffic (repeated/paraphrased questions are common), the two-tier cache is
what keeps p50 latency low — not the model. This load test's 5 concurrent, mostly-unique-question
users are close to a worst case (every request is a genuine cache miss), which is precisely why
it surfaces the generation bottleneck instead of hiding behind cache hits.

## What this means for the Kubernetes manifests (`../kubernetes/`)

The `HorizontalPodAutoscaler` in `api-deployment.yaml` scales on **CPU utilization** — which is
the *API process's* CPU, not Ollama's. Scaling `rag-api` pods without also scaling (or moving to
GPU-served `vllm_client.py`-style) the inference backend just creates more processes queued
behind the same slow, single generation bottleneck. A real production rollout of this system
would need either: (a) a GPU-served, batched inference backend behind LiteLLM
(`inference/litellm_config.yaml`), or (b) aggressive caching + a strict per-user rate limit, to
keep tail latency bounded under concurrent load. This load test is the evidence for that
recommendation, not a guess.

## Scaling this test further

45 seconds / 5 users was deliberately kept short — this environment has one Ollama process
serving one model on CPU, so a longer or higher-concurrency run doesn't reveal new behavior,
it just queues more requests behind the same bottleneck (confirmed by watching Ollama's own
process during a longer manual run: consistently at ~100% of one CPU core, requests processed
one at a time). Against a GPU-served or horizontally-scaled inference backend, a realistic next
step would be a sustained 5-minute run ramping to 50+ users to find the actual saturation point
of the *infrastructure* layer (Qdrant, Postgres, Redis, the FastAPI process itself) once
generation is no longer the dominant term.
