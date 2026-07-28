# 05 — Production RAG

The example to copy for a real project. Everything a deployable RAG service needs, in about a hundred lines.

## Run

```bash
pip install "windlass[openai,rag]"
export OPENAI_API_KEY=sk-...
python examples/05_production_rag/main.py
```

Without a key it falls back to the offline providers, so you can read the structure and see the ingestion behaviour without spending anything.

## What it shows

| Section | Concern |
|---|---|
| `setup()` | Configuration in one place: timeouts, retries, concurrency, caching |
| `pipeline()` | Built once at startup, `lru_cache`d, `.build()` called eagerly |
| Ingestion | Cleaning, deduplication, PII redaction — **before** anything is embedded |
| Query | Metadata filtering, the basis of multi-tenancy |
| Out of scope | Strict mode plus a score floor: refuse rather than invent |
| Persistence | Save and reload, so a restart does not re-index |
| Evaluation | The quality gate that belongs in CI |
| Wiring | `describe()` — what is actually deployed |

## The decisions worth copying

**PII redaction at ingestion, not at query time.**

```python
.preprocessor("pii", action="redact")
```

Once personal data is embedded, deleting a row does not undo the copies in your backups. Ingestion is the last point of genuine control. The example corpus contains an email address and a phone number; look at what is actually stored.

**Deduplication.**

```python
.preprocessor("dedup", threshold=0.9)
```

The corpus deliberately contains the same policy twice under different filenames. Real corpora always do. Duplicates waste embedding spend and crowd the top-k so one document fills the whole context window.

**Strict mode with a score floor.**

```python
.strict().min_score(0.015)
```

`strict()` alone rarely fires: dense retrieval returns its nearest neighbours however far away they are, so the result set is almost never empty. `min_score` is what makes "found nothing relevant" mean what you expect.

**Build eagerly.**

```python
rag.build()   # surface configuration errors now, not on request one
```

Construction loads an embedding model, opens connections and checks credentials. A missing API key should fail your deploy, not your first user's request.

**Guardrails on both stages.**

```python
.guardrails(injection=True, secrets=True, pii=True, on_violation="redact")
```

Injection detection matters most on *retrieved* text — in a RAG system, that is where injected instructions actually arrive.

## Turning it into a service

The pipeline function is already the right shape for a web app:

```python
from fastapi import FastAPI, Header

app = FastAPI()

@app.post("/ask")
async def ask(question: str, x_tenant: str = Header(...)):
    answer = await pipeline().aask(question, filters={"tenant": x_tenant})
    return {"answer": answer.answer, "sources": answer.sources}
```

Note `aask` rather than `ask` — the blocking form works inside a running loop, but calling the coroutine directly avoids a thread hop.

The [intermediate tutorial](../../docs/tutorials/intermediate.md) builds this out fully.

## Turning the gate into a test

```python
# tests/test_quality.py
def test_quality_has_not_regressed():
    report = pipeline().evaluate(GOLDEN_SET, metrics=["faithfulness", "answer_relevancy"])
    assert report.summary["faithfulness"] >= 0.85
```

Cheap lexical metrics can run on every commit. LLM-judged ones cost tokens — run those nightly and on release branches.

Twenty golden questions is enough to catch a regression. A thousand is a research project.

## Deployment checklist

- [ ] Tenant filter enforced in one place, never optional
- [ ] PII redacted at ingestion
- [ ] `.strict()` and `.min_score()` both set
- [ ] Guardrails on both stages
- [ ] Tracing configured, flushed on shutdown
- [ ] Ingestion runs out of band
- [ ] Embedding model baked into the image
- [ ] `windlass doctor` clean in the built image
- [ ] Alerts on `no_context` rate, p95 latency and tokens per request
