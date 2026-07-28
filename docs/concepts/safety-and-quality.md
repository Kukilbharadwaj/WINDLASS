# Guardrails, evaluation and observability

Three concerns that decide whether an AI system is deployable, and that teams usually bolt on after something goes wrong.

---

## Guardrails

Guardrails inspect what goes into the model and what comes out. Windlass runs them at two stages and offers three verdicts.

```python
rag = Windlass.rag().guardrails()
agent = Windlass.agent().guardrails(pii=True, on_violation="redact")
```

### Verdicts

| `on_violation` | Behaviour |
|---|---|
| `block` | Raise `GuardrailViolation`. The default |
| `redact` | Rewrite the content and continue. The right default for PII |
| `warn` | Log and let it through. Use while calibrating a new policy in production |
| `allow` | Detect and record only |

### The rule guardrail

Dependency-free, no model call, microseconds per kilobyte. Reach for it first.

```python
rag.guardrails(
    pii=True,                     # email, phone, SSN, cards, IBAN, IP, passports
    injection=True,               # instruction-override signatures
    secrets=True,                 # leaked API keys, tokens, private keys
    banned_words=["competitor"],  # word-boundary matched
    max_length=50_000,
    on_violation="redact",
)
```

```python
>>> guard = Windlass.guardrail(on_violation="redact")
>>> guard.validate("mail ada@example.com or call 555-123-4567")
'mail [EMAIL] or call [PHONE]'
```

Card numbers are Luhn-checked, so order ids and phone strings do not fire as credit cards.

### Prompt injection

Five signature families are detected: instruction override, role hijack, system-prompt exfiltration, delimiter injection and encoded payloads.

```python
>>> Windlass.guardrail().check("Ignore all previous instructions.").allowed
False
```

!!! warning "Pattern matching catches known shapes, not novel ones"
    Treat this as defence in depth alongside least-privilege tool design and human approval for consequential actions — not as a solution to prompt injection. The strongest mitigation remains that an injected instruction has nothing dangerous to reach.

**Run injection checks on retrieved documents, not just user input.** In a RAG system, that is where injected instructions actually arrive.

### NeMo Guardrails

For conversational policy that regexes cannot express — topical rails, Colang dialogue flows, model-based jailbreak detection:

```bash
pip install "windlass[guardrails]"
```

```python
agent.guardrails("nemo", config_path="./rails")
```

NeMo rails involve extra model calls. Run the rule guardrail first for the cheap deterministic checks and reserve NeMo for policy that genuinely needs a model.

### Composing

```python
policy = (
    Windlass.guardrail("rules", pii=True, on_violation="redact")
    & Windlass.guardrail("rules", injection=True, on_violation="block")
)
agent.guardrails(policy)
```

Each guardrail sees what the previous one produced, so the injection detector inspects already-masked text.

### PII at ingestion, not just at query time

Once personal data is embedded into an index it is very hard to remove — "delete the row" does not undo the copies in your backups. Redact at ingestion, where you still have a single point of control:

```python
rag.preprocessor("pii", action="redact")
```

---

## Evaluation

Evaluation answers "is this actually any good?", and it is what teams skip until something breaks.

```python
report = rag.evaluate(
    [
        {"question": "What is the refund window?", "reference": "30 days"},
        {"question": "Who approves releases?", "reference": "The release manager"},
    ],
    metrics=["faithfulness", "answer_relevancy", "f1"],
)
print(report)
```

```
EvaluationReport(2 samples, pass_rate=100.0%)
  answer_relevancy         0.900
  f1                       0.833
  faithfulness             0.950
```

### Two families

**Lexical** — free, deterministic, no model. The right choice for a CI gate:

`exact_match` · `f1` · `rouge_l` · `answer_relevancy_lexical` · `context_precision_lexical` · `context_recall_lexical` · `answer_length`

**LLM-judged** — costs tokens, measures what overlap cannot:

`faithfulness` · `answer_relevancy` · `answer_correctness` · `context_relevancy`

```python
Windlass.evaluator("builtin", metrics=["faithfulness"], llm=Windlass.llm("gpt-4o-mini"))
```

### Faithfulness is the one to watch

It measures whether the answer is supported by the retrieved context — which is the direct measure of hallucination in a RAG system, and the metric that catches a retrieval regression before your users do.

### RAGAS and DeepEval

```bash
pip install "windlass[evaluation]"
```

```python
rag.evaluate(questions, evaluator=Windlass.evaluator("ragas", llm=judge))
rag.evaluate(questions, evaluator=Windlass.evaluator("deepeval", threshold=0.7))
```

RAGAS specialises in *reference-free* metrics, which makes it usable on production traffic rather than only on a curated set. DeepEval produces pass/fail verdicts with explanations, which fits CI naturally.

### Evaluating retrieval separately

Most RAG failures are retrieval failures wearing a generation costume. Test retrieval on its own — it is deterministic, so the test is not flaky:

```python
def test_retrieval_finds_the_refund_policy():
    hits = rag.search("What is the refund window?")
    assert any("30 days" in h.chunk.content for h in hits)
```

### In CI

```python
def test_quality_has_not_regressed():
    report = rag.evaluate(GOLDEN_SET, metrics=["faithfulness", "f1"])
    assert report.summary["faithfulness"] >= 0.85
    assert report.summary["f1"] >= 0.70
```

Twenty golden questions and a threshold turn "it feels worse since Tuesday" into a red build.

---

## Observability

Windlass instruments every meaningful operation as a **span** — a model call, a retrieval, a tool execution, an ingestion run. Instrumentation is always on and always cheap: with no tracer configured, spans resolve to `NullTracer`.

```python
rag = Windlass.rag().observe("console")     # print a trace tree
rag = Windlass.rag().observe("langfuse")    # ship to Langfuse
rag = Windlass.rag().observe("langsmith")   # ship to LangSmith
```

### Console — the debugging one

```
▸ rag.ask  (chain)
  ⌕ retrieve  (retriever)
    · 12ms
  ◆ generate  (llm)
    · 840ms  312 tok
  · 856ms
```

The fastest way to answer "what is this thing actually doing?" without signing up for anything.

```python
rag.observe("console", show_io=True)   # include inputs and outputs
```

### Hosted backends

```bash
pip install "windlass[observability]"
export LANGFUSE_PUBLIC_KEY=pk-... LANGFUSE_SECRET_KEY=sk-...
```

Nested spans, prompts, completions, token counts and latency, searchable across runs. That is the difference between "users say it got worse" and "the retriever started returning the wrong section on Tuesday".

Both exporters batch in the background, so tracing adds negligible request latency. Call `flush()` before a short-lived process exits or you lose the tail.

### Assertions in tests

```python
from windlass.providers.observability.console import MemoryTracer

tracer = MemoryTracer()
rag = Windlass.rag().observe(tracer)
rag.ask("...")

assert tracer.count("retriever") == 1
assert tracer.count("llm") == 1
assert tracer.errors() == []
assert tracer.total_usage()["total_tokens"] < 5000
```

That last assertion is a cost regression test.

### Your own spans

```python
tracer = rag.build().tracer

with tracer.span("business-rules", kind="chain") as span:
    span.set_input(order)
    result = apply_rules(order)
    span.set_output(result)
```

It nests inside the surrounding Windlass trace automatically.

### A tracer never breaks the application

Every export path swallows its own errors and logs a warning. Observability failing should never take down the thing it observes.

---

## Putting it together

```python
rag = (
    Windlass.rag()
    .llm("gpt-4o-mini", temperature=0)
    .preprocessor("clean")
    .preprocessor("pii", action="redact")       # PII never reaches the index
    .chunker("markdown")
    .retriever("hybrid")
    .guardrails(injection=True, on_violation="block")
    .observe("langfuse")
    .strict()                                   # refuse rather than invent
)

rag.ingest("./documents")

report = rag.evaluate(GOLDEN_SET, metrics=["faithfulness", "answer_relevancy"])
assert report.summary["faithfulness"] >= 0.85
```

Ten lines: guarded, observable, measured, and safe against the most common way RAG systems embarrass their owners.
