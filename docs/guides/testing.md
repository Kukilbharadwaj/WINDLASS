# Testing

Testing an AI application is hard for one specific reason: the model is non-deterministic, slow and costs money. The fix is to make every component replaceable — which is what the architecture already does.

The entire Windlass test suite (over 650 tests) runs offline in well under a minute with no API keys. Your suite can too.

That count includes every docstring example. `--doctest-modules` is on in `pyproject.toml`, so a documented example that does not run fails the build — which is the only way examples stay correct.

## The helpers

```python
from windlass.testing import (
    scripted_llm, call,
    fake_rag, fake_agent,
    isolated_registry, capture_spans,
    make_documents, make_chunks,
    assert_answer_uses_context, RecordingTool,
)
```

## A scripted model

```python
llm = scripted_llm(["first answer", "second answer"])

assert llm.complete("a").content == "first answer"
assert llm.complete("b").content == "second answer"

assert llm.last_prompt() == "b"     # it records what it was asked
assert len(llm.calls) == 2
```

Driving an agent through a specific tool-use path:

```python
llm = scripted_llm(
    ["", "The weather in Paris is 18°C."],
    tool_calls=[[call("get_weather", city="Paris")], []],
)
```

Turn one emits a tool call and no text; turn two, having seen the result, answers. That is the exact shape of a real agent turn.

Asserting on what the model actually received:

```python
def handler(messages, tools):
    assert any("Paris" in m.content for m in messages)
    assert tools and tools[0]["function"]["name"] == "get_weather"
    return "checked"

llm = scripted_llm([], handler=handler)
```

## A whole RAG pipeline, offline

```python
def test_retrieval_finds_the_refund_policy():
    rag = fake_rag(
        ["Refunds are available within 30 days of purchase."],
        answers=["Within 30 days."],
    )
    answer = rag.ask("What is the refund window?")

    assert answer.answer == "Within 30 days."
    assert "30 days" in answer.contexts[0].chunk.content
```

Or the assertion that matters, stated directly:

```python
assert_answer_uses_context(answer, "30 days")
```

!!! tip "Assert on retrieval, not on wording"
    `answer.answer` is the model's phrasing — that is the model's behaviour, not yours, and asserting on it produces brittle tests. `answer.contexts` is *your* retrieval, and it is deterministic.

## A whole agent, offline

```python
def test_the_agent_searches_before_answering():
    agent = fake_agent(
        ["", "I found three open tickets about billing."],
        tools=[search_tickets],
        tool_calls=[[call("search_tickets", query="billing")], []],
    )

    response = agent.run("What are people complaining about?")

    assert response.output.startswith("I found three")
    assert response.tool_calls[0].name == "search_tickets"
    assert response.tool_calls[0].arguments == {"query": "billing"}
```

## Recording tool calls

```python
def test_the_agent_passes_the_right_arguments():
    recorder = RecordingTool("lookup", result={"found": True})

    agent = fake_agent(
        ["", "Found it."],
        tools=[recorder.as_tool()],
        tool_calls=[[call("lookup", key="abc")], []],
    )
    agent.run("look up abc")

    recorder.assert_called_with(key="abc")
    assert recorder.call_count == 1
```

## Asserting on traces

```python
def test_the_cached_path_makes_no_model_call():
    with capture_spans() as spans:
        rag = Windlass.rag().observe("memory")
        rag.ingest_text("Some content.")
        rag.ask("what content?")

    assert spans.count("retriever") == 1
    assert spans.count("llm") == 1
    assert spans.errors() == []
    assert spans.total_usage()["total_tokens"] < 1000     # a cost regression test
```

## Isolating registrations

Registering a component is a global side effect. Without isolation, a test that registers one leaks into every test that follows:

```python
def test_my_custom_chunker():
    with isolated_registry() as registry:

        @register.chunker("test-only")
        class TestChunker(Chunker):
            def split_text(self, text):
                return [text]

        assert registry.has("chunker", "test-only")
        assert Windlass.rag().chunker("test-only").ingest_text("x") == 1

    assert not REGISTRY.has("chunker", "test-only")     # gone
```

## Fixtures

Settings are a process-wide singleton, so reset them:

```python
# conftest.py
import pytest
from windlass import reset_settings
from windlass.core.registry import REGISTRY


@pytest.fixture(autouse=True)
def clean_settings():
    reset_settings()
    yield
    reset_settings()


@pytest.fixture
def registry():
    snapshot = REGISTRY.snapshot()
    yield REGISTRY
    REGISTRY.restore(snapshot)


@pytest.fixture
def rag():
    from windlass.testing import fake_rag
    return fake_rag(["Paris is the capital of France."], answers=["Paris."])
```

## Building test data

```python
from windlass.testing import make_chunks, make_documents

documents = make_documents("first doc", "second doc", corpus="unit-test")
chunks = make_chunks("alpha", "beta", dimensions=128)     # already embedded

store.add(chunks)
```

## Async tests

Windlass sets `asyncio_mode = "auto"` in its own config; do the same and async tests need no decorator:

```toml
[tool.pytest.ini_options]
asyncio_mode = "auto"
```

```python
async def test_async_ingestion():
    rag = Windlass.rag()
    assert await rag.aingest_text("some content") >= 1
    assert (await rag.aask("what content?")).contexts
```

## Testing a custom component

Test the **contract**, then test that it composes:

```python
class TestMyRetriever:
    async def test_returns_scored_chunks(self):
        retriever = MyRetriever(client=fake_client)
        hits = await retriever.aretrieve_chunks("query", k=3)

        assert all(isinstance(h, ScoredChunk) for h in hits)
        assert len(hits) <= 3

    async def test_honours_the_result_limit(self):
        assert len(await MyRetriever(client=fake_client).aretrieve("q", k=2)) <= 2

    def test_composes_into_a_pipeline(self):
        rag = Windlass.rag().retriever(MyRetriever(client=fake_client))
        assert rag.ask("anything").contexts

    def test_composes_into_hybrid_retrieval(self):
        hybrid = Windlass.retriever(
            "hybrid", retrievers=[MyRetriever(client=fake_client), other]
        )
        assert hybrid.retrieve("query").hits
```

That last one catches the failure mode where a component works alone and breaks in composition.

## What to test, and what not to

### Do test

- **Retrieval.** Deterministic, yours, and where most quality problems live.
- **Tool schemas.** `tool.schema()` is what the model sees; assert on it.
- **Tool behaviour.** Tools are ordinary functions.
- **Pipeline wiring.** That the semantic chunker got the embedder, that filters are applied.
- **Error paths.** Guardrails firing, approvals pausing, budgets enforcing.
- **That nothing executed before approval.** The property that actually matters in HITL.

### Don't test

- **The model's exact wording.** That is the provider's behaviour, and it changes.
- **Third-party adapters against live APIs** in unit tests. Mark them `@pytest.mark.network` and run them separately.
- **Semantic quality with assertions.** Use `evaluate()` and thresholds instead.

## Quality gates

Behavioural tests answer "does it work". Evaluation answers "is it good". You want both:

```python
GOLDEN = [
    {"question": "What is the refund window?", "reference": "30 days"},
    {"question": "Who approves releases?", "reference": "The release manager"},
]


@pytest.mark.network
def test_quality_has_not_regressed():
    report = production_pipeline().evaluate(
        GOLDEN, metrics=["faithfulness", "answer_relevancy"]
    )
    print(report)
    assert report.summary["faithfulness"] >= 0.85
    assert report.summary["answer_relevancy"] >= 0.80
```

Cheap lexical metrics can run on every commit; the LLM-judged ones cost tokens, so run those nightly and on release branches.

## Markers

Windlass declares four:

```toml
markers = [
    "integration: end-to-end tests across multiple components",
    "network: requires network access or live API credentials",
    "optional: requires an optional dependency group",
    "slow: takes more than a second",
]
```

```bash
pytest -m "not network"        # the default developer loop
pytest -m integration
pytest --cov=myapp
```

## Testing MCP without a server

```python
from windlass.providers.mcp.fastmcp import StaticMCPClient

def test_the_agent_uses_the_mcp_tool():
    client = StaticMCPClient(tools={"read_file": lambda path: "file contents"})

    agent = fake_agent(
        ["", "The file says: file contents"],
        tool_calls=[[call("read_file", path="/tmp/x")], []],
    ).mcp(client)

    assert "file contents" in agent.run("read /tmp/x").output
```

Every MCP path — discovery, namespacing, proxying, binding — without a subprocess.

## Testing human-in-the-loop

```python
def test_nothing_happens_without_approval():
    executed = []

    @tool(requires_approval=True)
    def deploy(env: str) -> str:
        """Deploy to an environment."""
        executed.append(env)
        return f"deployed to {env}"

    agent = (
        Windlass.agent()
        .llm("fake", responses=["", "Deployed."],
             tool_calls=[[call("deploy", env="prod")], []])
        .tool(deploy)
        .checkpoint()
    )

    with pytest.raises(AgentInterrupt):
        agent.run("deploy to prod", thread_id="t1")

    assert executed == []                                   # the point

    assert agent.resume("t1", approved=True).output == "Deployed."
    assert executed == ["prod"]
```
