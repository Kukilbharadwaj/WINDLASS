# Intermediate: a production RAG service

Take the beginner's bot and make it something you would put behind a load balancer: multi-tenant, guarded, observable, measured, and served over HTTP.

**You need:** the [beginner tutorial](beginner.md), and:

```bash
pip install "windlass[openai,rag,observability]" fastapi uvicorn
```

---

## 1. Isolate tenants

A shared index that leaks one customer's documents into another's answers is the failure that ends a product. Two mechanisms, used together.

**Tag at ingestion:**

```python
rag.ingest(f"./corpora/{tenant}", metadata={"tenant": tenant})
```

**Filter at retrieval:**

```python
answer = rag.ask(question, filters={"tenant": tenant})
```

Filters are pushed down to stores that support them and applied in-process by those that do not, with identical semantics either way.

!!! danger "Make the filter impossible to forget"
    A missing filter is a data breach, not a bug. Never let one be optional in your own code:

    ```python
    def ask_for_tenant(question: str, tenant: str) -> RAGAnswer:
        if not tenant:
            raise ValueError("tenant is required")
        return rag.ask(question, filters={"tenant": tenant}, thread_id=f"t:{tenant}")
    ```

    Stronger still, give each tenant its own namespace or collection so a filtering bug cannot cross the boundary at all:

    ```python
    Windlass.rag().vectordb("pinecone", collection="docs", namespace=tenant)
    ```

---

## 2. Redact PII before it is indexed

Once personal data is embedded, it is very hard to remove — deleting a row does not undo the copies in your backups. Ingestion is the last point of real control:

```python
rag = (
    Windlass.rag()
    .preprocessor("clean", min_length=100)
    .preprocessor("dedup", threshold=0.92)
    .preprocessor("pii", action="redact")
)
```

`dedup` matters more than it looks: real corpora are full of the same policy PDF exported three times. Duplicates waste embedding spend and, worse, crowd the top-k so one document fills the whole context window.

---

## 3. Guard both directions

```python
rag = (
    Windlass.rag()
    .guardrails(
        injection=True,          # instruction-override attempts
        secrets=True,            # leaked credentials in output
        pii=True,
        on_violation="redact",
    )
)
```

To reject malicious input outright while still masking PII, compose two policies:

```python
policy = (
    Windlass.guardrail("rules", pii=True, secrets=True, injection=False,
                      on_violation="redact")
    & Windlass.guardrail("rules", pii=False, secrets=False, injection=True,
                        on_violation="block")
)
rag = Windlass.rag().guardrails(policy)
```

---

## 4. Configure once, centrally

```python
# app/config.py
from windlass import configure

def setup() -> None:
    configure(
        default_llm="openai",
        default_embedding="huggingface",
        default_vectordb="pinecone",
        temperature=0.0,
        request_timeout=45.0,
        max_concurrency=16,
        retry={"attempts": 4, "max_delay": 30},
        cache={"enabled": True, "backend": "disk", "ttl": 86_400},
        project="docs-qa-prod",
    )
```

Caching embeddings is always safe — they are deterministic — and it makes re-ingesting a mostly-unchanged corpus nearly free.

---

## 5. Build once at startup

Construction is not free: it loads an embedding model, opens a store connection, checks credentials. Do it during startup, not on the first request.

```python
# app/pipeline.py
from functools import lru_cache

from windlass import Windlass

from app.config import setup


@lru_cache(maxsize=1)
def pipeline():
    setup()
    rag = (
        Windlass.rag()
        .llm("gpt-4o-mini", temperature=0)
        .embedding("huggingface", model="BAAI/bge-small-en-v1.5")
        .vectordb("pinecone", collection="docs")
        .chunker("markdown")
        .retriever("hybrid", weights=[0.6, 0.4])
        .preprocessor("clean").preprocessor("dedup").preprocessor("pii", action="redact")
        .guardrails(injection=True, pii=True, on_violation="redact")
        .observe("langfuse")
        .top_k(8)
        .max_context_tokens(10_000)
        .strict()
    )
    rag.build()          # surface configuration errors now, not on request one
    return rag
```

---

## 6. Serve it

```python
# app/main.py
from contextlib import asynccontextmanager

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from windlass import GuardrailViolation, WindlassError
from app.pipeline import pipeline


@asynccontextmanager
async def lifespan(app: FastAPI):
    pipeline()                      # warm the model and the store
    yield
    pipeline().close()


app = FastAPI(title="Docs Q&A", lifespan=lifespan)


class Question(BaseModel):
    question: str
    thread_id: str | None = None


@app.post("/ask")
async def ask(body: Question, x_tenant: str = Header(...)):
    try:
        answer = await pipeline().aask(
            body.question,
            filters={"tenant": x_tenant},
            thread_id=body.thread_id or f"tenant:{x_tenant}",
        )
    except GuardrailViolation as exc:
        raise HTTPException(422, detail=str(exc.message)) from exc
    except WindlassError as exc:
        raise HTTPException(502, detail=str(exc.message)) from exc

    return {
        "answer": answer.answer,
        "sources": answer.sources,
        "tokens": answer.usage.total_tokens,
        "latency_ms": round(answer.latency_ms),
    }


@app.post("/ask/stream")
async def ask_stream(body: Question, x_tenant: str = Header(...)):
    async def tokens():
        async for token in pipeline().astream_ask(
            body.question, filters={"tenant": x_tenant}
        ):
            yield token

    return StreamingResponse(tokens(), media_type="text/plain")


@app.get("/health")
async def health():
    return {"status": "ok", "chunks": await pipeline().build().acount()}
```

```bash
uvicorn app.main:app --workers 4
```

Note that the async endpoint uses `aask`. The blocking `ask` would also work — Windlass detects the running loop and dispatches to a background thread — but calling the coroutine directly avoids the hop.

!!! warning "One embedding model per worker"
    Four uvicorn workers means four copies of the embedding model in memory. Size the container accordingly, or move embeddings to an API-backed provider.

---

## 7. Ingest out of band

Ingestion does not belong on the request path.

```python
# scripts/ingest.py
import argparse

from app.pipeline import pipeline


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tenant", required=True)
    parser.add_argument("--path", required=True)
    args = parser.parse_args()

    rag = pipeline()
    count = rag.ingest(
        args.path,
        metadata={"tenant": args.tenant, "ingested_by": "batch"},
        show_progress=True,
    )
    print(f"Indexed {count} chunks for {args.tenant}")


if __name__ == "__main__":
    main()
```

Re-running it is safe: chunk ids are content hashes, so unchanged content upserts.

---

## 8. Gate quality in CI

```python
# tests/test_quality.py
import json
import pathlib

import pytest

from app.pipeline import pipeline

GOLDEN = json.loads(pathlib.Path("tests/golden.json").read_text())


@pytest.mark.network
def test_quality_has_not_regressed():
    report = pipeline().evaluate(
        GOLDEN, metrics=["faithfulness", "answer_relevancy", "context_precision_lexical"]
    )
    print(report)
    assert report.summary["faithfulness"] >= 0.85
    assert report.summary["answer_relevancy"] >= 0.80


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("What is the refund window?", "30 days"),
        ("Who approves releases?", "release manager"),
    ],
)
def test_retrieval_finds_the_right_passage(question, expected):
    hits = pipeline().search(question, k=5)
    assert any(expected in h.chunk.content for h in hits)
```

The retrieval tests are deterministic and free — run them on every commit. The evaluation gate costs tokens; run it nightly and on release branches.

`golden.json` is twenty real questions with reference answers. Twenty is enough to catch a regression; a thousand is a research project.

---

## 9. Watch it in production

```python
.observe("langfuse")
```

The three things to actually alert on:

| Signal | Where | Why |
|---|---|---|
| `no_context` rate | `answer.metadata["no_context"]` | Strict mode refusing means retrieval is failing |
| p95 latency by stage | trace spans | Tells you whether the model or the store slowed down |
| tokens per request | `answer.usage.total_tokens` | Cost regressions are silent otherwise |

```python
if answer.metadata.get("no_context"):
    metrics.increment("rag.no_context", tags={"tenant": tenant})
```

---

## 10. Deploy

```dockerfile
FROM python:3.12-slim

RUN pip install --no-cache-dir "windlass[openai,rag,observability]" \
                                fastapi "uvicorn[standard]"

# Bake the embedding model in so request one does not pay for a download.
RUN python -c "from sentence_transformers import SentenceTransformer; \
    SentenceTransformer('BAAI/bge-small-en-v1.5')"

WORKDIR /app
COPY app app
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
```

Checklist before you ship:

- [ ] Tenant filtering enforced in one place, never optional
- [ ] PII redacted at ingestion
- [ ] `.strict()` on
- [ ] Guardrails on both stages
- [ ] Tracing configured, `flush()` on shutdown
- [ ] Retrieval tests in CI; evaluation gate nightly
- [ ] Ingestion runs out of band
- [ ] Secrets from the environment, never from code
- [ ] `windlass doctor` clean in the built image

---

## Next

- [Advanced](advanced.md) — multi-agent systems, human-in-the-loop, custom graphs.
- [Performance](../guides/performance.md) — where the time and money go.
- [Best practices](../guides/best-practices.md).
