"""The naive RAG pipeline, end to end.

    Document -> Load -> Chunk -> Embed -> Vector Store -> Retrieve Top-K
             -> Prompt -> LLM -> Answer

`RAGPipeline` wires together `ingest`, `chunk`, `embed`, `retrieve`, and
`generate` behind three calls: build an index, ask a question, save/load
the index. Every dependency is injected (embedder, generator, vector
store), so tests can swap in the `ExtractiveGenerator` and a fake embedder
to run the whole pipeline with no network access and no Ollama.
"""

from __future__ import annotations

import logging
from pathlib import Path

from tqdm import tqdm

from .chunk import chunk_documents
from .config import settings
from .embed import Embedder, get_embedder
from .generate import Generator, get_generator
from .ingest import load_from_directory, load_from_hf_dataset
from .retrieve import InMemoryVectorStore, VectorStore, get_vector_store
from .schema import Document, EmbeddedChunk, RagAnswer

logger = logging.getLogger(__name__)


class RAGPipeline:
    def __init__(
        self,
        embedder: Embedder | None = None,
        generator: Generator | None = None,
        vector_store: VectorStore | None = None,
        chunk_size: int = settings.chunk_size,
        chunk_overlap: int = settings.chunk_overlap,
        top_k: int = settings.top_k,
    ) -> None:
        # NB: explicit `is None` checks, not `x or default()` — a valid
        # injected dependency that happens to be falsy (e.g. a vector store
        # with __len__() == 0) must not be silently replaced.
        self.embedder = embedder if embedder is not None else get_embedder()
        self.generator = generator if generator is not None else get_generator()
        self.vector_store = vector_store if vector_store is not None else get_vector_store()
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k = top_k

    # -- Indexing -------------------------------------------------------

    def build_index(self, documents: list[Document], batch_size: int = 32) -> int:
        """Chunk, embed, and index `documents`. Returns the number of chunks indexed."""
        chunks = chunk_documents(documents, self.chunk_size, self.chunk_overlap)
        logger.info("Chunked %d documents into %d chunks.", len(documents), len(chunks))

        for start in tqdm(range(0, len(chunks), batch_size), desc="Embedding chunks"):
            batch = chunks[start : start + batch_size]
            vectors = self.embedder.embed([c.text for c in batch])
            self.vector_store.add(
                [EmbeddedChunk(chunk=c, vector=v) for c, v in zip(batch, vectors, strict=True)]
            )
        return len(chunks)

    @classmethod
    def from_hf_dataset(cls, limit: int | None = None, **kwargs) -> "RAGPipeline":
        """Convenience constructor: build an index from the open HF dataset."""
        pipeline = cls(**kwargs)
        documents = load_from_hf_dataset(limit=limit)
        pipeline.build_index(documents)
        return pipeline

    @classmethod
    def from_directory(cls, directory: Path | str | None = None, **kwargs) -> "RAGPipeline":
        """Convenience constructor: build an index from local files."""
        pipeline = cls(**kwargs)
        documents = load_from_directory(directory)
        pipeline.build_index(documents)
        return pipeline

    # -- Querying ---------------------------------------------------------

    def ask(self, question: str, top_k: int | None = None) -> RagAnswer:
        if len(self.vector_store) == 0:
            raise RuntimeError(
                "The vector store is empty — call build_index(...) (or load a saved "
                "index) before calling ask()."
            )

        query_vector = self.embedder.embed_one(question)
        sources = self.vector_store.search(query_vector, top_k=top_k or self.top_k)
        answer_text = self.generator.generate(question, sources)
        return RagAnswer(question=question, answer=answer_text, sources=sources)

    # -- Persistence (in-memory store only) --------------------------------

    def save_index(self, directory: Path | str = settings.index_dir) -> None:
        if not isinstance(self.vector_store, InMemoryVectorStore):
            raise NotImplementedError("save_index() only supports the in-memory vector store.")
        self.vector_store.save(directory)

    @classmethod
    def load_index(cls, directory: Path | str = settings.index_dir, **kwargs) -> "RAGPipeline":
        pipeline = cls(vector_store=InMemoryVectorStore.load(directory), **kwargs)
        return pipeline
