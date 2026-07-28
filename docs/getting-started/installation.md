# Installation

## Requirements

Python 3.12 or newer.

## The core install

```bash
pip install windlass
```

That pulls in four packages: `pydantic`, `pydantic-settings`, `httpx` and `typing-extensions`. Nothing else — no torch, no LangChain, no vector database driver.

Verify it:

```bash
windlass doctor
```

You should see a health report ending in `Everything looks healthy.`, including a smoke test that runs a complete RAG pipeline offline.

## Extras

Everything beyond the core is opt-in. Install only what you use.

### LLM providers

| Command | Provides |
|---|---|
| `pip install "windlass[openai]"` | OpenAI, Azure OpenAI, and any OpenAI-compatible gateway |
| `pip install "windlass[anthropic]"` | Anthropic Claude |
| `pip install "windlass[gemini]"` | Google Gemini |
| `pip install "windlass[groq]"` | Groq |
| *(none needed)* | **Ollama** — speaks HTTP, uses the core `httpx` |

### Feature groups

| Command | Provides |
|---|---|
| `pip install "windlass[agent]"` | LangGraph agent runtime |
| `pip install "windlass[rag]"` | Embeddings, vector stores, document loaders, tokenisers |
| `pip install "windlass[embeddings]"` | `sentence-transformers` for local embeddings |
| `pip install "windlass[vectordb]"` | FAISS, ChromaDB and Pinecone |
| `pip install "windlass[faiss]"` | FAISS only |
| `pip install "windlass[chroma]"` | ChromaDB only |
| `pip install "windlass[pinecone]"` | Pinecone only |
| `pip install "windlass[loaders]"` | PDF, DOCX, PPTX, XLSX, HTML, YouTube |
| `pip install "windlass[ocr]"` | Image OCR (also needs the Tesseract binary) |
| `pip install "windlass[audio]"` | Audio transcription via faster-whisper |
| `pip install "windlass[mcp]"` | FastMCP client |
| `pip install "windlass[evaluation]"` | RAGAS and DeepEval |
| `pip install "windlass[guardrails]"` | NVIDIA NeMo Guardrails |
| `pip install "windlass[observability]"` | LangSmith and Langfuse |
| `pip install "windlass[pii]"` | Microsoft Presidio for NER-based PII detection |
| `pip install "windlass[cache]"` | Persistent disk cache |

### Combining

Extras compose. Ask for exactly what your application needs:

```bash
pip install "windlass[agent,mcp,openai]"
pip install "windlass[rag,anthropic,observability]"
```

### Everything

```bash
pip install "windlass[full]"
```

This is large — it includes torch by way of `sentence-transformers`. Use it for exploration; pin the specific extras you need for a deployment.

## What happens when an extra is missing

Nothing crashes at import time, because nothing is imported until it is used. When you *do* use it, you get an actionable error rather than a traceback from somebody else's package:

```pycon
>>> Windlass.vectordb("pinecone", dimensions=1536)
MissingDependencyError: The Pinecone vector store is not installed.

Hint: Run:

    pip install "windlass[pinecone]"
```

Windlass never lets a raw `ImportError` from an optional dependency reach your code. If you see one, that is a bug worth reporting.

## Checking what you have

```bash
windlass doctor      # extras, credentials, defaults, and an end-to-end smoke test
windlass info        # a one-line summary
windlass list        # every registered component; uninstalled ones are marked
```

`windlass list` marks components whose extra is missing, so you can see at a glance what a given install can actually do:

```
vectordb
  chroma            (not installed)
  faiss             (not installed)
  memory
  pinecone          (not installed)
```

## Credentials

Windlass reads the conventional provider environment variables, so an existing setup needs no changes:

```bash
export OPENAI_API_KEY=sk-...
export ANTHROPIC_API_KEY=sk-ant-...
export GOOGLE_API_KEY=...
export GROQ_API_KEY=...
export PINECONE_API_KEY=...
export LANGFUSE_PUBLIC_KEY=... LANGFUSE_SECRET_KEY=...
```

A `.env` file in the working directory is read automatically. See [Configuration](configuration.md) for the full precedence order.

## Development install

```bash
git clone https://github.com/windlass/windlass && cd windlass
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -e ".[dev]"
pre-commit install
pytest
```

The whole test suite runs offline in a few seconds. See [Testing](../guides/testing.md).

## Docker

```dockerfile
FROM python:3.12-slim

# Install extras in their own layer so application changes do not re-resolve them.
RUN pip install --no-cache-dir "windlass[openai,rag,observability]"

WORKDIR /app
COPY . .
CMD ["python", "-m", "myapp"]
```

If you use local embeddings, bake the model into the image so the first request does not pay for a download:

```dockerfile
RUN python -c "from sentence_transformers import SentenceTransformer; \
    SentenceTransformer('BAAI/bge-small-en-v1.5')"
```

## Upgrading

Windlass follows semantic versioning. Before 1.0, minor versions may change public API; the [changelog](https://github.com/windlass/windlass/blob/main/CHANGELOG.md) records every break, and the [migration guide](../guides/migration.md) explains how to move.
