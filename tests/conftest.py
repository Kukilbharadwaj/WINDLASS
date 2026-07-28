"""Shared pytest fixtures.

Every fixture here is offline and deterministic. The whole suite runs with no
network access and no API keys, which is only possible because Windlass ships
dependency-free implementations of every essential component.
"""

from __future__ import annotations

import pytest

from windlass.core.config import reset_settings
from windlass.core.registry import REGISTRY
from windlass.core.types import Chunk, Document


@pytest.fixture(autouse=True)
def _clean_settings():
    """Reset process-wide settings around every test.

    Settings are a global singleton. Without this, a test that calls
    ``configure()`` leaks into every test that runs afterwards.
    """
    reset_settings()
    yield
    reset_settings()


@pytest.fixture
def registry():
    """Yield the global registry, restoring it afterwards.

    Registering a component is a global side effect; this snapshots and restores
    the registry so tests that register custom components stay isolated.
    """
    snapshot = REGISTRY.snapshot()
    yield REGISTRY
    REGISTRY.restore(snapshot)


@pytest.fixture
def fake_llm():
    """A scripted model returning a single fixed answer."""
    from windlass.providers.llm.fake import FakeLLM

    return FakeLLM(responses=["A scripted answer."])


@pytest.fixture
def embedder():
    """A deterministic, dependency-free embedding model."""
    from windlass.providers.embeddings.hash import HashEmbedder

    return HashEmbedder(dimensions=128)


@pytest.fixture
def store(embedder):
    """An empty in-memory vector store matching ``embedder``."""
    from windlass.providers.vectordb.memory import InMemoryVectorStore

    return InMemoryVectorStore(dimensions=embedder.dimension())


@pytest.fixture
def documents():
    """Three short documents on clearly distinct topics."""
    return [
        Document(
            content=(
                "Paris is the capital of France. The Eiffel Tower stands on the "
                "Champ de Mars and was completed in 1889."
            ),
            source="france.txt",
            metadata={"topic": "geography", "year": 2024},
        ),
        Document(
            content=(
                "Python is a high-level programming language. It emphasises "
                "readability and supports multiple programming paradigms."
            ),
            source="python.txt",
            metadata={"topic": "programming", "year": 2023},
        ),
        Document(
            content=(
                "Retrieval augmented generation combines a retriever with a "
                "language model so answers are grounded in source documents."
            ),
            source="rag.txt",
            metadata={"topic": "ai", "year": 2025},
        ),
    ]


@pytest.fixture
def chunks(documents, embedder):
    """Embedded chunks derived from :func:`documents`."""
    from windlass.providers.chunkers.recursive import RecursiveChunker

    produced = RecursiveChunker(chunk_size=200, overlap=20).chunk(documents)
    for chunk, vector in zip(produced, embedder.embed([c.content for c in produced]), strict=True):
        chunk.embedding = vector
    return produced


@pytest.fixture
def populated_store(store, chunks) -> object:
    """An in-memory store pre-loaded with :func:`chunks`."""
    store.add(chunks)
    return store


@pytest.fixture
def sample_chunk() -> Chunk:
    """A single embedded chunk."""
    return Chunk(
        content="Windlass is a modular AI application framework.",
        embedding=[0.1] * 8,
        metadata={"source": "readme.md"},
    )


@pytest.fixture
def text_corpus(tmp_path):
    """A temporary directory holding a small mixed-format corpus."""
    (tmp_path / "notes.txt").write_text(
        "Meeting notes: the launch is scheduled for March.", encoding="utf-8"
    )
    (tmp_path / "guide.md").write_text(
        "---\ntitle: Setup Guide\ntags: [docs, setup]\n---\n"
        "# Setup Guide\n\nInstall the package, then run the migration.",
        encoding="utf-8",
    )
    (tmp_path / "data.csv").write_text("name,role\nada,engineer\ngrace,admiral\n", encoding="utf-8")
    (tmp_path / "records.json").write_text(
        '[{"text": "first record", "id": 1}, {"text": "second record", "id": 2}]',
        encoding="utf-8",
    )
    return tmp_path
