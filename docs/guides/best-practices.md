# Best practices

Opinions earned from the failure modes that actually occur.

---

## RAG

### Look at what was retrieved before you touch the prompt

Most RAG failures are retrieval failures wearing a generation costume. If the right chunk is not in `answer.contexts`, no amount of prompt engineering will produce a correct answer.

```python
answer = rag.ask("What is the refund window?")
for hit in answer.contexts:
    print(f"{hit.score:.3f}  {hit.retriever:<12}  {hit.chunk.content[:70]}")
```

### Tune in this order

Earlier steps have far more leverage than later ones:

1. **Chunking** — wrong size or wrong boundaries is the most common root cause.
2. **Hybrid retrieval** — especially with identifiers, codes or product names in the corpus.
3. **`top_k` and the context budget** — cheap to try, immediately measurable.
4. **The embedding model** — `bge-small` → `bge-m3` is a real jump. Requires re-ingesting.
5. **Contextual enrichment** — genuine gains, paid once at ingestion.
6. **The prompt** — last, not first.

### Match the chunker to the content

| Content | Chunker |
|---|---|
| Prose, articles, reports | `recursive` |
| Docs, wikis, READMEs | `markdown` — the heading path is nearly free context |
| Source code | `code` |
| Long-form where topics shift | `semantic` |
| Mixed, and you want precision *and* context | `parent_child` |
| A hard context budget | `token` |

### Default to hybrid retrieval

```python
rag.retriever("hybrid")
```

Embeddings handle paraphrase; BM25 finds `E1042`. If your corpus has any identifiers, codes, SKUs or function names, dense-only search will miss them and you will not notice until a user does.

### Turn on strict mode

```python
rag.strict()
```

The single most effective anti-hallucination setting. It removes the model's *opportunity* to invent rather than asking it not to.

### Redact PII at ingestion

```python
rag.preprocessor("pii", action="redact")
```

Once personal data is embedded, deleting a row does not undo the copies in your backups. Ingestion is the last point of genuine control.

### Deduplicate

```python
rag.preprocessor("dedup", threshold=0.9)
```

Real corpora are full of the same document exported three times. Duplicates waste embedding spend and, worse, crowd the top-k so one document fills the whole context window.

### Never let a tenant filter be optional

```python
def ask(question: str, tenant: str) -> RAGAnswer:
    if not tenant:
        raise ValueError("tenant is required")
    return rag.ask(question, filters={"tenant": tenant})
```

A missing filter is a data breach, not a bug. Better still, give each tenant its own namespace so a filtering bug cannot cross the boundary at all.

---

## Agents

### Few tools, well described

Five well-named tools beat twenty overlapping ones. The description is prompt text — the model chooses on it — so write it like a prompt.

```python
# Weak
@tool
def search(q: str) -> list: ...

# Strong
@tool
def search_support_tickets(query: str, status: str = "open") -> list[dict]:
    """Search the support ticket system by free text.

    Args:
        query: Words to search for in ticket titles and bodies.
        status: Filter by "open", "closed" or "all".
    """
```

### Return what the model can use

A tool returning a forty-field object teaches the model to guess. Return the three fields that matter.

```python
return [{"id": t.id, "title": t.title, "status": t.status} for t in results]
```

### Gate consequential actions

```python
@tool(requires_approval=True)
def issue_refund(order_id: str, amount_cents: int) -> dict: ...
```

Anything that spends money, sends a message or deletes data.

### Least privilege beats clever filtering

`search_products(query)` scoped to one index is safer than `run_sql(query)`. Prompt injection is real, and the strongest mitigation is that the injected instruction has nothing dangerous to reach.

### Set budgets

```python
agent.max_iterations(15)

@tool(timeout=30.0)
def slow_thing(): ...
```

The difference between a bad afternoon and a bad quarter.

### Treat retrieved text as untrusted

Documents and tool output can contain instructions.

```python
agent.guardrails(injection=True)
```

### Specialise before you add the thirtieth tool

One agent with thirty tools reasons badly. Split into specialists with a supervisor, and give each its own model — research is high-volume and wants a cheap model; synthesis wants a strong one.

---

## Reliability

### Let the retry policy do its job

```python
configure(retry={"attempts": 4, "max_delay": 30})
```

Only transient failures retry — rate limits, timeouts, 5xx, connection resets. A 401 fails immediately, because the next attempt would fail identically. Do not add your own retry loop on top; you will multiply the two.

### Set timeouts everywhere

```python
configure(request_timeout=45.0)
```

The default is 60 seconds. Local models loading for the first time need more; interactive paths usually want less.

### Bound your concurrency

```python
configure(max_concurrency=16)
```

Unbounded fan-out over a few hundred prompts is the fastest way to get rate limited. Windlass bounds every batch operation; this is the ceiling.

### Handle the errors that mean something

```python
from windlass import AuthenticationError, GuardrailViolation, WindlassError, RateLimitError

try:
    answer = rag.ask(question)
except GuardrailViolation as exc:
    return refuse(exc.rule)                 # 422 — user error
except AuthenticationError:
    alert_ops()                             # your problem, not theirs
    raise
except RateLimitError as exc:
    return retry_later(exc.retry_after)     # 429
except WindlassError as exc:
    log.error("windlass failed", extra=exc.context)
    return degraded_response()
```

`exc.context` carries structured detail — provider, model, ids — that is worth putting in your logs.

---

## Cost

### Cache embeddings

```python
configure(cache={"enabled": True, "backend": "disk", "ttl": 86_400})
```

Embeddings are deterministic, so caching them is always safe. This makes re-ingesting a mostly-unchanged corpus nearly free.

### Right-size the model per job

Research, classification, metadata extraction and summarisation rarely need your best model. Synthesis and final answers usually do.

```python
researcher = Windlass.agent().llm("gpt-4o-mini")
writer     = Windlass.agent().llm("claude-sonnet-4-5")
```

### Watch the context budget

```python
rag.max_context_tokens(8000)
```

`top_k=20` on 2000-character chunks is 40k characters of prompt on every request. Measure whether it actually improves answers before paying for it.

### Track tokens as a metric

```python
metrics.histogram("rag.tokens", answer.usage.total_tokens, tags={"tenant": tenant})
```

Cost regressions are silent otherwise.

---

## Testing

### Test retrieval, not the model's wording

```python
def test_finds_the_refund_policy():
    hits = rag.search("refund window")
    assert any("30 days" in h.chunk.content for h in hits)
```

Retrieval is yours and it is deterministic. The model's phrasing is neither.

### Use the fakes

```python
from windlass.testing import fake_rag, fake_agent, call

rag = fake_rag(["Refunds are available within 30 days."], answers=["30 days."])
agent = fake_agent(["", "Found it."], tools=[search],
                   tool_calls=[[call("search", query="refunds")], []])
```

No network, no keys, no flakiness.

### Gate quality in CI

```python
report = rag.evaluate(GOLDEN_SET, metrics=["faithfulness", "f1"])
assert report.summary["faithfulness"] >= 0.85
```

Twenty golden questions turn "it feels worse since Tuesday" into a red build.

### Reset settings between tests

```python
@pytest.fixture(autouse=True)
def clean_settings():
    reset_settings()
    yield
    reset_settings()
```

---

## Structure

### Build once, at startup

```python
@lru_cache(maxsize=1)
def pipeline():
    rag = Windlass.rag()...
    rag.build()          # surface configuration errors now, not on request one
    return rag
```

Construction loads an embedding model, opens connections and checks credentials. That belongs in startup, not on the request path.

### Configure centrally

```python
# app/config.py
def setup():
    configure(default_llm="openai", temperature=0.0, ...)
```

One place to change a model, one place to audit.

### Keep the chain readable

The builder chain *is* your architecture documentation. Someone should be able to read it top to bottom and know what the system does. If it has grown past twenty lines, that is a signal to split into named factory functions.

### Promote repeated Level 3 into a plugin

If you reach past the abstraction the same way three times, write a component and register it. Then it is Level 2 for you and everyone else on your team.

---

## Deployment

- [ ] Secrets from the environment, never from code
- [ ] `windlass doctor` clean in the built image
- [ ] Embedding models baked into the image
- [ ] Tracing configured; `flush()` on shutdown
- [ ] Ingestion runs out of band, not on the request path
- [ ] Tenant isolation enforced in exactly one place
- [ ] `.strict()` on
- [ ] Guardrails on both stages
- [ ] Alerts on the `no_context` rate, p95 latency and tokens per request
