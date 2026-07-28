"""Regressions for wiring defects that only a real integration reproduces.

Each test here covers a defect that the rest of the suite could not see, for the
same reason in every case: the failure is in how components are *wired together*
at build time, not in any component's own logic. A unit test that constructs a
component directly, or a builder test that always names its components
explicitly, walks straight past all of them.
"""

from __future__ import annotations

import threading

import pytest

from windlass import Windlass, tool
from windlass.core.container import Container
from windlass.core.exceptions import InterruptedError_ as AgentInterrupt
from windlass.core.types import ToolCall
from windlass.providers.observability.console import MemoryTracer
from windlass.testing import capture_spans


class TestContainerBindingsAreHonoured:
    """A bound component must actually be used by the builders.

    The builders passed the *configured default name* whenever the user had not
    named a component, and ``Container.component`` only consults its bindings
    when the spec is ``None``. So every binding was silently ignored — including
    ``Windlass.container().bind_instance(...)``, which the public API documents as
    the place to do application-wide wiring.
    """

    def test_rag_uses_a_bound_llm(self) -> None:
        container = Container()
        container.bind_instance("llm", Windlass.llm("fake", responses=["from container"]))
        rag = Windlass.rag(container)
        rag.ingest_text("Some indexed content about bindings.")
        assert rag.ask("anything").answer == "from container"

    def test_agent_uses_a_bound_llm(self) -> None:
        container = Container()
        container.bind_instance("llm", Windlass.llm("fake", responses=["from container"]))
        assert Windlass.agent(container).run("hi").output == "from container"

    def test_bound_tracer_is_used_without_calling_observe(self) -> None:
        tracer = MemoryTracer()
        container = Container()
        container.bind_instance("tracer", tracer)
        rag = Windlass.rag(container).llm("fake", responses=["answer"])
        rag.ingest_text("Content that should produce spans when traced.")
        rag.ask("question")
        assert tracer.spans, "a bound tracer collected nothing"
        assert "retriever" in {span.kind for span in tracer.spans}

    def test_explicit_choice_still_beats_a_binding(self) -> None:
        container = Container()
        container.bind_instance("llm", Windlass.llm("fake", responses=["binding"]))
        rag = Windlass.rag(container).llm("fake", responses=["explicit"])
        rag.ingest_text("Precedence between an explicit call and a binding.")
        assert rag.ask("anything").answer == "explicit"

    def test_capture_spans_collects_without_observe(self) -> None:
        """The shipped test helper binds a tracer; it has to be consulted.

        The helper's own doctest asserted ``count() >= 0``, which is true of an
        empty tracer, so nothing caught this.
        """
        with capture_spans() as spans:
            Windlass.agent().llm("fake", responses=["hi"]).run("hello")
        assert spans.count() > 0, "capture_spans() captured nothing"


class TestLangGraphStateResolution:
    """The graph runtime must build under ``from __future__ import annotations``.

    LangGraph resolves a state schema with ``get_type_hints``, which evaluates
    the annotations against the *defining module's* globals. A ``State``
    ``TypedDict`` declared inside the builder method, with ``Annotated``
    imported into that method's local scope, raised ``NameError: name
    'Annotated' is not defined`` before a single node ever ran.
    """

    def test_graph_agent_runs(self) -> None:
        pytest.importorskip("langgraph")
        agent = Windlass.agent().llm("fake", responses=["graph answer"]).graph()
        assert agent.run("hello").output == "graph answer"

    def test_native_graph_is_editable_and_recompilable(self) -> None:
        pytest.importorskip("langgraph")
        agent = Windlass.agent().llm("fake", responses=["after edit"]).graph()
        agent.build()
        graph = agent.native_graph()
        assert hasattr(graph, "add_node")
        graph.add_node("critic", lambda state: state)
        agent.recompile()
        assert agent.run("x").output == "after edit"

    def test_graph_executes_tool_calls(self) -> None:
        pytest.importorskip("langgraph")

        @tool
        def double(x: int) -> int:
            """Double a number.

            Args:
                x: The number to double.
            """
            return x * 2

        agent = (
            Windlass.agent()
            .llm(
                "fake",
                responses=["", "done"],
                tool_calls=[[ToolCall(name="double", arguments={"x": 4})], []],
            )
            .tool(double)
            .graph()
        )
        response = agent.run("double four")
        assert [r.data for s in response.steps for r in s.tool_results] == [8]


class TestResumeKeepsTheAuditTrail:
    """An approved tool call must appear in ``steps``, not only in ``messages``.

    ``aresume`` delegates to ``arun``, which starts a fresh step list, so the one
    call a human explicitly authorised was missing from the structure an audit
    trail would read.
    """

    @staticmethod
    def _agent(ledger: list[tuple[float, str]]):
        @tool(requires_approval=True)
        def pay(amount_usd: float, recipient: str) -> str:
            """Pay someone.

            Args:
                amount_usd: Amount in dollars.
                recipient: Who to pay.
            """
            ledger.append((amount_usd, recipient))
            return f"paid {amount_usd} to {recipient}"

        return (
            Windlass.agent()
            .llm(
                "fake",
                responses=["", "Done."],
                tool_calls=[
                    [ToolCall(name="pay", arguments={"amount_usd": 100.0, "recipient": "acme"})],
                    [],
                ],
            )
            .tool(pay)
            .checkpoint()
        )

    def test_approved_call_is_in_steps(self) -> None:
        ledger: list[tuple[float, str]] = []
        agent = self._agent(ledger)
        with pytest.raises(AgentInterrupt):
            agent.run("pay acme", thread_id="a1")
        assert ledger == [], "the tool ran before approval"

        response = agent.resume("a1", approved=True)
        executed = [r.name for step in response.steps for r in step.tool_results]
        assert "pay" in executed, "the approved call is missing from steps"
        assert ledger == [(100.0, "acme")]

    def test_edited_arguments_are_reflected_in_steps_and_effects(self) -> None:
        ledger: list[tuple[float, str]] = []
        agent = self._agent(ledger)
        with pytest.raises(AgentInterrupt):
            agent.run("pay acme", thread_id="a2")

        pending = agent.pending_approvals("a2")
        agent.resume(
            "a2",
            approved=True,
            edited_arguments={pending[0].id: {"amount_usd": 10.0, "recipient": "acme"}},
        )
        assert ledger == [(10.0, "acme")], "the human's edit was not applied"

    def test_step_indices_stay_sequential(self) -> None:
        ledger: list[tuple[float, str]] = []
        agent = self._agent(ledger)
        with pytest.raises(AgentInterrupt):
            agent.run("pay acme", thread_id="a3")
        response = agent.resume("a3", approved=True)
        assert [s.index for s in response.steps] == list(range(len(response.steps)))


class TestTracerNeverBreaksTheApplication:
    """A tracer flush must not raise, including during interpreter shutdown."""

    def test_bounded_flush_survives_a_refused_thread(self, monkeypatch) -> None:
        """Python refuses new threads once shutdown starts — exactly when an
        atexit flush runs. That surfaced as a traceback from the tracer.
        """
        from windlass.providers.observability import platforms

        def refuse(self) -> None:
            raise RuntimeError("can't create new thread at interpreter shutdown")

        monkeypatch.setattr(threading.Thread, "start", refuse)
        calls: list[str] = []

        class _Log:
            def debug(self, *args, **kwargs) -> None:
                calls.append("debug")

            def warning(self, *args, **kwargs) -> None:
                calls.append("warning")

        platforms._bounded_flush(
            lambda: None, label="Test", timeout=1.0, log=_Log()
        )  # must not raise
        assert calls == ["debug"]


class TestLangfuseHostAliases:
    """``LANGFUSE_BASE_URL`` is read by the vendor SDK, so Windlass reads it too.

    Without the alias, a project on a non-default region configured that way
    silently disagreed: ``windlass config`` reported the default cloud host while
    the vendor client quietly used the right one.
    """

    def test_base_url_alias_is_honoured(self, monkeypatch) -> None:
        from windlass.core.config import WindlassSettings

        monkeypatch.delenv("LANGFUSE_HOST", raising=False)
        monkeypatch.setenv("LANGFUSE_BASE_URL", "https://us.cloud.langfuse.com")
        assert WindlassSettings().langfuse_host == "https://us.cloud.langfuse.com"

    def test_langfuse_host_still_wins(self, monkeypatch) -> None:
        from windlass.core.config import WindlassSettings

        monkeypatch.setenv("LANGFUSE_HOST", "https://self-hosted.example.com")
        monkeypatch.setenv("LANGFUSE_BASE_URL", "https://us.cloud.langfuse.com")
        assert WindlassSettings().langfuse_host == "https://self-hosted.example.com"
