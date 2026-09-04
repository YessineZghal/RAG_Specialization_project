"""Metrics + tracing setup — real Prometheus counters/histograms the API
actually updates on every request (scraped by `deployment/docker-compose.yml`'s
Prometheus service, see `prometheus.yml`), and a minimal OpenTelemetry
tracer for FastAPI's automatic request spans.
"""

from __future__ import annotations

from prometheus_client import Counter, Histogram

REQUEST_COUNT = Counter(
    "rag_requests_total", "Total number of RAG API requests", ["endpoint", "status"]
)
REQUEST_LATENCY = Histogram(
    "rag_request_latency_seconds", "End-to-end request latency", ["endpoint"]
)
RETRIEVAL_LATENCY = Histogram("rag_retrieval_latency_seconds", "Vector search latency")
GENERATION_LATENCY = Histogram("rag_generation_latency_seconds", "LLM generation latency")
CACHE_HITS = Counter("rag_cache_hits_total", "Cache hits", ["cache_type"])
CACHE_MISSES = Counter("rag_cache_misses_total", "Cache misses", ["cache_type"])


def setup_tracing(app) -> None:
    """Instrument a FastAPI app with OpenTelemetry auto-tracing. Spans are
    exported to the console by default (no Jaeger/Tempo collector running
    in this repo's local stack) -- swap `ConsoleSpanExporter` for an OTLP
    exporter pointed at a real collector in an actual deployment.
    """
    from opentelemetry import trace
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter

    provider = TracerProvider()
    provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
    trace.set_tracer_provider(provider)
    FastAPIInstrumentor.instrument_app(app)
