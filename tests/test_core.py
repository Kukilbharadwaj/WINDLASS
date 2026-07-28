"""Unit tests for the core runtime: types, registry, container, config, concurrency."""

from __future__ import annotations

import asyncio

import pytest

from windlass.core.cache import MemoryCache, NullCache, cached_call, make_key
from windlass.core.concurrency import batched, gather_bounded, iter_sync, map_async, run_sync
from windlass.core.config import RetryConfig, configure, load_config_file, reset_settings, settings
from windlass.core.container import Container
from windlass.core.exceptions import (
    ComponentNotFoundError,
    ConfigurationError,
    DuplicateComponentError,
    MissingDependencyError,
    WindlassError,
)
from windlass.core.lazy import is_available, require
from windlass.core.registry import Registry
from windlass.core.retry import backoff_delays, is_retryable, retry_async
from windlass.core.text import (
    count_tokens,
    detect_language,
    normalize_whitespace,
    split_sentences,
    strip_html,
    text_hash,
    tokenize_words,
)
from windlass.core.types import (
    Chunk,
    Completion,
    Document,
    Message,
    Role,
    Usage,
    content_id,
    normalize_messages,
)
from windlass.core.vectors import (
    cosine_similarity,
    mmr,
    normalize,
    reciprocal_rank_fusion,
    top_k,
)


# ---------------------------------------------------------------------------
# types
# ---------------------------------------------------------------------------
class TestTypes:
    def test_document_id_is_content_derived_and_stable(self):
        a = Document(content="same text", source="s.txt")
        b = Document(content="same text", source="s.txt")
        c = Document(content="other text", source="s.txt")
        assert a.id == b.id != c.id

    def test_content_id_is_deterministic(self):
        assert content_id("x", "doc") == content_id("x", "doc")
        assert content_id("x") != content_id("y")

    def test_chunk_fills_in_offsets_and_id(self):
        chunk = Chunk(content="hello", document_id="d1", index=2, start_char=10)
        assert chunk.end_char == 15
        assert chunk.id.startswith("chk_")

    def test_usage_addition_aggregates_calls_and_cost(self):
        total = Usage(prompt_tokens=10, cost_usd=0.01) + Usage(completion_tokens=5, cost_usd=0.02)
        assert total.prompt_tokens == 10
        assert total.completion_tokens == 5
        assert total.total_tokens == 15
        assert total.calls == 2
        assert total.cost_usd == pytest.approx(0.03)

    def test_usage_cost_stays_none_when_neither_side_reports_it(self):
        assert (Usage(prompt_tokens=1) + Usage(prompt_tokens=2)).cost_usd is None

    def test_message_role_accepts_a_string(self):
        assert Message(role="user", content="hi").role is Role.USER

    @pytest.mark.parametrize(
        "prompt",
        [
            "hello",
            Message.user("hello"),
            [Message.user("hello")],
            [{"role": "user", "content": "hello"}],
        ],
    )
    def test_normalize_messages_accepts_every_prompt_shape(self, prompt):
        messages = normalize_messages(prompt)
        assert len(messages) == 1
        assert messages[0].content == "hello"

    def test_normalize_messages_rejects_nonsense(self):
        with pytest.raises(TypeError):
            normalize_messages([object()])

    def test_completion_converts_to_an_assistant_message(self):
        message = Completion(content="answer").to_message()
        assert message.role is Role.ASSISTANT
        assert message.content == "answer"

    def test_rag_answer_sources_are_unique_and_ordered(self):
        from windlass.core.types import RAGAnswer, ScoredChunk

        answer = RAGAnswer(
            contexts=[
                ScoredChunk(chunk=Chunk(content="a", metadata={"source": "x"})),
                ScoredChunk(chunk=Chunk(content="b", metadata={"source": "y"})),
                ScoredChunk(chunk=Chunk(content="c", metadata={"source": "x"})),
            ]
        )
        assert answer.sources == ["x", "y"]


# ---------------------------------------------------------------------------
# exceptions
# ---------------------------------------------------------------------------
class TestExceptions:
    def test_missing_dependency_names_the_install_command(self):
        error = MissingDependencyError("The thing", extra="rag")
        assert 'pip install "windlass[rag]"' in str(error)
        assert error.context["extra"] == "rag"

    def test_component_not_found_lists_alternatives(self):
        error = ComponentNotFoundError("chunker", "nope", ["recursive", "semantic"])
        assert "recursive, semantic" in str(error)

    def test_every_error_derives_from_windlass_error(self):
        assert issubclass(MissingDependencyError, WindlassError)
        assert issubclass(ComponentNotFoundError, WindlassError)

    def test_hint_is_rendered_after_the_message(self):
        assert "Hint: do this" in str(WindlassError("broke", hint="do this"))


# ---------------------------------------------------------------------------
# registry
# ---------------------------------------------------------------------------
class TestRegistry:
    def test_register_and_create(self):
        registry = Registry()

        class Widget:
            def __init__(self, size: int = 1) -> None:
                self.size = size

        registry.register("chunker", "widget", Widget)
        assert registry.has("chunker", "widget")
        assert registry.create("chunker", "widget", size=5).size == 5

    def test_a_constructed_registry_is_isolated_from_installed_plugins(self):
        """A hand-built registry must hold exactly what you put in it.

        Auto-discovering entry points here would mean that installing any
        third-party Windlass plugin silently changes the contents of every
        registry in the process — including the clean ones tests build — and
        would make the suite's result depend on what else is in the venv.
        """
        registry = Registry()
        assert registry.names("chunker") == []
        registry.register("chunker", "only-mine", lambda: None)
        assert registry.names("chunker") == ["only-mine"]

    def test_discovery_is_opt_in_and_the_global_registry_opts_in(self):
        from windlass.core.registry import REGISTRY

        assert Registry()._discover is False
        assert REGISTRY._discover is True

    def test_load_plugins_still_works_on_an_isolated_registry(self):
        """Discovery on demand stays available; it is only the automatic
        discovery on first lookup that an isolated registry declines."""
        assert isinstance(Registry().load_plugins(), list)

    def test_lookup_is_case_insensitive_and_alias_aware(self):
        registry = Registry()
        registry.register("llm", "thing", lambda: "made", aliases=("alias", "other"))
        assert registry.get("llm", "THING").name == "thing"
        assert registry.get("llm", "alias").name == "thing"

    def test_duplicate_registration_raises_without_override(self):
        registry = Registry()
        registry.register("llm", "dup", lambda: 1)
        with pytest.raises(DuplicateComponentError):
            registry.register("llm", "dup", lambda: 2)
        registry.register("llm", "dup", lambda: 3, override=True)

    def test_lazy_registration_defers_the_import(self):
        registry = Registry()
        registry.register_lazy(
            "chunker", "lazy", "windlass.providers.chunkers.recursive:RecursiveChunker"
        )
        spec = registry.get("chunker", "lazy")
        assert spec.target is None
        assert spec.resolve().__name__ == "RecursiveChunker"

    def test_manifest_and_decorator_registration_do_not_collide(self):
        """The lazy manifest and the module's own decorator describe one component."""
        from windlass.core.registry import REGISTRY

        spec = REGISTRY.get("chunker", "recursive")
        first = spec.resolve()  # imports the module, firing @register.chunker
        assert REGISTRY.get("chunker", "recursive").resolve() is first

    def test_unknown_name_reports_what_is_available(self):
        registry = Registry()
        registry.register("llm", "known", lambda: 1)
        with pytest.raises(ComponentNotFoundError, match="known"):
            registry.get("llm", "unknown")

    def test_unknown_kind_is_rejected(self):
        with pytest.raises(Exception, match="Unknown component kind"):
            Registry().register("not_a_kind", "x", lambda: 1)

    def test_custom_kinds_can_be_added(self):
        registry = Registry()
        registry.add_kind("reranker")
        registry.register("reranker", "cohere", lambda: "r")
        assert registry.names("reranker") == ["cohere"]

    def test_bad_constructor_arguments_produce_a_configuration_error(self):
        registry = Registry()
        registry.register("llm", "strict", lambda size: size)
        with pytest.raises(ConfigurationError, match="Could not construct"):
            registry.create("llm", "strict", wrong="arg")

    def test_register_decorator_namespace(self, registry):
        from windlass import register

        @register.chunker("decorated", description="test only")
        class Decorated:
            pass

        assert registry.get("chunker", "decorated").resolve() is Decorated

    def test_unknown_kind_on_decorator_namespace_raises(self):
        from windlass import register

        with pytest.raises(AttributeError, match="not a Windlass component kind"):
            register.reranker  # noqa: B018


# ---------------------------------------------------------------------------
# container
# ---------------------------------------------------------------------------
class TestContainer:
    def test_binding_and_resolution(self):
        container = Container()
        container.bind("greeting", lambda: "hello")
        assert container.resolve("greeting") == "hello"

    def test_singletons_are_produced_once(self):
        container = Container()
        counter = iter(range(10))
        container.bind("n", lambda: next(counter), singleton=True)
        assert container.resolve("n") == container.resolve("n") == 0

    def test_non_singletons_are_produced_every_time(self):
        container = Container()
        counter = iter(range(10))
        container.bind("n", lambda: next(counter), singleton=False)
        assert container.resolve("n") != container.resolve("n")

    def test_child_scopes_inherit_and_override(self):
        parent = Container()
        parent.bind("value", lambda: "parent")
        child = parent.scope()
        child.bind("value", lambda: "child")
        assert parent.resolve("value") == "parent"
        assert child.resolve("value") == "child"

    def test_missing_binding_raises_unless_a_default_is_given(self):
        container = Container()
        with pytest.raises(ConfigurationError, match="Nothing is bound"):
            container.resolve("absent")
        assert container.resolve("absent", "fallback") == "fallback"

    def test_component_accepts_name_instance_and_factory(self):
        from windlass.providers.llm.fake import FakeLLM

        container = Container()
        by_name = container.component("llm", "fake", responses=["a"])
        assert by_name.complete("x").content == "a"

        instance = FakeLLM(responses=["b"])
        assert container.component("llm", instance) is instance

        assert container.component("llm", lambda: instance) is instance

    def test_config_on_an_instance_is_rejected(self):
        from windlass.providers.llm.fake import FakeLLM

        with pytest.raises(ConfigurationError, match="already-constructed"):
            Container().component("llm", FakeLLM(), model="x")

    def test_factory_receiving_the_container(self):
        container = Container()
        container.bind("base", lambda: 10)
        container.bind("derived", lambda c: c.resolve("base") * 2)
        assert container.resolve("derived") == 20


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------
class TestConfig:
    def test_defaults_are_offline_capable(self):
        current = settings()
        assert current.default_llm == "fake"
        assert current.default_embedding == "hash"
        assert current.default_vectordb == "memory"

    def test_configure_merges_and_validates(self):
        updated = configure(temperature=0.0, retry={"attempts": 5})
        assert updated.temperature == 0.0
        assert updated.retry.attempts == 5

    def test_unknown_setting_is_rejected_with_the_valid_list(self):
        with pytest.raises(ConfigurationError, match="Unknown setting"):
            configure(not_a_setting=1)

    def test_out_of_range_value_is_rejected(self):
        with pytest.raises(ConfigurationError):
            configure(temperature=99.0)

    def test_reset_restores_defaults(self):
        configure(temperature=0.0)
        reset_settings()
        assert settings().temperature == 0.2

    def test_secrets_are_masked_for_display(self):
        configure(openai_api_key="sk-secret-value")
        masked = settings().masked()
        assert masked["openai_api_key"] == "***"
        assert settings().secret("openai_api_key") == "sk-secret-value"

    def test_env_var_is_read_without_the_windlass_prefix(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-from-env")
        reset_settings()
        assert settings().secret("anthropic_api_key") == "sk-ant-from-env"

    def test_json_config_file_unwraps_the_windlass_table(self, tmp_path):
        path = tmp_path / "windlass.json"
        path.write_text('{"windlass": {"default_llm": "echo", "top": 1}}', encoding="utf-8")
        assert load_config_file(path)["default_llm"] == "echo"

    def test_toml_config_file(self, tmp_path):
        path = tmp_path / "windlass.toml"
        path.write_text('default_llm = "echo"\ntemperature = 0.1\n', encoding="utf-8")
        loaded = load_config_file(path)
        assert loaded == {"default_llm": "echo", "temperature": 0.1}

    def test_unsupported_config_format(self, tmp_path):
        path = tmp_path / "windlass.ini"
        path.write_text("x=1", encoding="utf-8")
        with pytest.raises(ConfigurationError, match="Unsupported config format"):
            load_config_file(path)

    def test_require_secret_explains_how_to_set_it(self):
        from windlass.core.exceptions import AuthenticationError

        with pytest.raises(AuthenticationError, match="OPENAI_API_KEY"):
            settings().require_secret("openai_api_key", provider="openai", env_var="OPENAI_API_KEY")


# ---------------------------------------------------------------------------
# concurrency
# ---------------------------------------------------------------------------
class TestConcurrency:
    def test_run_sync_from_plain_code(self):
        async def work() -> int:
            return 42

        assert run_sync(work()) == 42

    def test_run_sync_from_inside_a_running_loop(self):
        """The Jupyter / FastAPI case: asyncio.run would raise here."""

        async def outer() -> int:
            def blocking() -> int:
                async def inner() -> int:
                    return 7

                return run_sync(inner())

            return await asyncio.to_thread(blocking)

        assert asyncio.run(outer()) == 7

    def test_run_sync_reuses_one_loop_across_calls(self):
        """Regression: a fresh loop per call kills pooled connections.

        asyncio.run closes its loop on exit. Any provider holding a long-lived
        httpx.AsyncClient pools keep-alive sockets on the loop that opened them,
        so a second blocking call raised `RuntimeError: Event loop is closed`
        from inside httpx. Mock transports never caught it because they open no
        sockets — loop identity is the property to assert instead.
        """

        async def current_loop_id() -> int:
            return id(asyncio.get_running_loop())

        first = run_sync(current_loop_id())
        second = run_sync(current_loop_id())
        assert first == second

    def test_a_loop_bound_resource_survives_between_blocking_calls(self):
        """The behaviour the loop identity above actually buys."""
        holder: dict[str, asyncio.Queue] = {}

        async def create() -> None:
            holder["queue"] = asyncio.Queue()
            await holder["queue"].put("written on the first call")

        async def consume() -> str:
            return await holder["queue"].get()

        run_sync(create())
        assert run_sync(consume()) == "written on the first call"

    def test_run_sync_refuses_to_deadlock_on_its_own_loop(self):
        """Blocking on the background loop from inside it can never return."""

        async def reenter() -> None:
            async def inner() -> int:
                return 1

            coro = inner()
            try:
                run_sync(coro)
            finally:
                coro.close()  # never awaited once run_sync rejects it

        with pytest.raises(RuntimeError, match="background event loop"):
            run_sync(reenter())

    def test_iter_sync_drains_the_whole_generator(self):
        """Regression: repeated asyncio.run finalises suspended async generators."""

        async def generate():
            for i in range(5):
                yield i

        assert list(iter_sync(generate())) == [0, 1, 2, 3, 4]

    async def test_gather_bounded_preserves_order_and_limits_concurrency(self):
        in_flight = 0
        peak = 0

        async def work(value: int) -> int:
            nonlocal in_flight, peak
            in_flight += 1
            peak = max(peak, in_flight)
            await asyncio.sleep(0.01)
            in_flight -= 1
            return value * 2

        results = await gather_bounded([work(i) for i in range(8)], limit=3)
        assert results == [0, 2, 4, 6, 8, 10, 12, 14]
        assert peak <= 3

    async def test_gather_bounded_can_collect_exceptions(self):
        async def boom() -> int:
            raise ValueError("nope")

        async def fine() -> int:
            return 1

        results = await gather_bounded([boom(), fine()], limit=2, return_exceptions=True)
        assert isinstance(results[0], ValueError)
        assert results[1] == 1

    async def test_map_async(self):
        async def square(x: int) -> int:
            return x * x

        assert await map_async(square, [1, 2, 3]) == [1, 4, 9]

    def test_batched(self):
        assert list(batched(range(5), 2)) == [[0, 1], [2, 3], [4]]
        assert list(batched([], 2)) == []
        with pytest.raises(ValueError):
            list(batched([1], 0))


# ---------------------------------------------------------------------------
# retry
# ---------------------------------------------------------------------------
class TestRetry:
    def test_transient_errors_are_retryable_and_auth_errors_are_not(self):
        from windlass.core.exceptions import AuthenticationError, RateLimitError

        assert is_retryable(RateLimitError("slow down"))
        assert is_retryable(ConnectionError())
        assert not is_retryable(AuthenticationError("bad key"))
        assert not is_retryable(ValueError("bad input"))

    def test_http_status_codes_are_classified(self):
        class Failure(Exception):
            status_code = 503

        class Rejected(Exception):
            status_code = 400

        assert is_retryable(Failure())
        assert not is_retryable(Rejected())

    def test_backoff_schedule_is_exponential_and_capped(self):
        delays = backoff_delays(RetryConfig(attempts=5, initial_delay=1, multiplier=2, max_delay=4))
        assert delays == [1.0, 2.0, 4.0, 4.0]

    async def test_retry_recovers_from_a_transient_failure(self):
        attempts = 0

        async def flaky() -> str:
            nonlocal attempts
            attempts += 1
            if attempts < 3:
                raise ConnectionError("reset")
            return "ok"

        result = await retry_async(flaky, config=RetryConfig(attempts=4, initial_delay=0, jitter=0))
        assert result == "ok"
        assert attempts == 3

    async def test_non_retryable_failures_fail_immediately(self):
        from windlass.core.exceptions import AuthenticationError

        attempts = 0

        async def denied() -> str:
            nonlocal attempts
            attempts += 1
            raise AuthenticationError("bad key")

        with pytest.raises(AuthenticationError):
            await retry_async(denied, config=RetryConfig(attempts=5, initial_delay=0))
        assert attempts == 1


# ---------------------------------------------------------------------------
# cache
# ---------------------------------------------------------------------------
class TestCache:
    def test_key_is_order_independent(self):
        assert make_key("a", {"x": 1, "y": 2}) == make_key("a", {"y": 2, "x": 1})

    def test_memory_cache_evicts_least_recently_used(self):
        cache = MemoryCache(max_size=2)
        cache.set("a", 1)
        cache.set("b", 2)
        cache.get("a")  # "a" becomes most recent
        cache.set("c", 3)  # evicts "b"
        assert cache.get("a") == 1
        assert cache.get("b") is None
        assert cache.get("c") == 3

    def test_ttl_expiry(self):
        cache = MemoryCache(ttl=-1)
        cache.set("k", "v")
        assert cache.get("k") is None

    def test_stats(self):
        cache = MemoryCache()
        cache.set("k", 1)
        cache.get("k")
        cache.get("missing")
        stats = cache.stats()
        assert stats["hits"] == 1 and stats["misses"] == 1
        assert stats["hit_rate"] == 0.5

    def test_null_cache_stores_nothing(self):
        cache = NullCache()
        cache.set("k", "v")
        assert cache.get("k") is None

    async def test_cached_call_computes_once(self):
        cache = MemoryCache()
        calls = 0

        async def compute() -> int:
            nonlocal calls
            calls += 1
            return 5

        assert await cached_call(cache, "k", compute) == 5
        assert await cached_call(cache, "k", compute) == 5
        assert calls == 1


# ---------------------------------------------------------------------------
# lazy imports
# ---------------------------------------------------------------------------
class TestLazy:
    def test_is_available_does_not_raise(self):
        assert is_available("json") is True
        assert is_available("definitely_not_a_real_module_xyz") is False

    def test_require_returns_the_module(self):
        assert require("json", extra="core", feature="JSON").dumps([1]) == "[1]"

    def test_require_translates_a_missing_module(self):
        with pytest.raises(MissingDependencyError) as caught:
            require("definitely_not_a_real_module_xyz", extra="rag", feature="The thing")
        assert "windlass[rag]" in str(caught.value)


# ---------------------------------------------------------------------------
# text helpers
# ---------------------------------------------------------------------------
class TestText:
    def test_normalize_whitespace_preserves_paragraphs(self):
        assert normalize_whitespace("a  \t b\n\n\n\nc") == "a b\n\nc"

    def test_sentence_splitting_survives_abbreviations_and_decimals(self):
        assert split_sentences("Dr. Smith paid $1.50. Then he left.") == [
            "Dr. Smith paid $1.50.",
            "Then he left.",
        ]

    def test_sentence_splitting_survives_initials(self):
        assert split_sentences("A. B. Jones arrived. He waited.") == [
            "A. B. Jones arrived.",
            "He waited.",
        ]

    def test_empty_text_yields_no_sentences(self):
        assert split_sentences("   ") == []

    def test_tokenize_keeps_hyphens_and_apostrophes(self):
        assert tokenize_words("State-of-the-art RAG, don't you think?") == [
            "state-of-the-art",
            "rag",
            "don't",
            "you",
            "think",
        ]

    def test_strip_html(self):
        assert strip_html("<p>Hello <b>world</b></p>") == "Hello world"
        assert "alert" not in strip_html("<script>alert(1)</script><p>safe</p>")

    def test_count_tokens_scales_with_length(self):
        assert count_tokens("") == 0
        assert count_tokens("hello world") >= 1
        assert count_tokens("word " * 100) > count_tokens("word")

    def test_text_hash_ignores_case_and_whitespace(self):
        assert text_hash("Hello  World") == text_hash("hello world")
        assert text_hash("a") != text_hash("b")

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("the quick brown fox is in the garden and there", "en"),
            ("这是一个测试文本内容", "zh"),
            ("это тестовый текст", "ru"),
            ("", "en"),
        ],
    )
    def test_language_detection(self, text, expected):
        assert detect_language(text) == expected


# ---------------------------------------------------------------------------
# vectors
# ---------------------------------------------------------------------------
class TestVectors:
    def test_cosine_similarity_bounds(self):
        assert cosine_similarity([1, 0], [1, 0]) == pytest.approx(1.0)
        assert cosine_similarity([1, 0], [0, 1]) == pytest.approx(0.0)
        assert cosine_similarity([1, 0], [-1, 0]) == pytest.approx(-1.0)

    def test_zero_vector_does_not_divide_by_zero(self):
        assert cosine_similarity([0, 0], [1, 1]) == 0.0

    def test_length_mismatch_is_rejected(self):
        with pytest.raises(ValueError, match="length mismatch"):
            cosine_similarity([1, 2], [1, 2, 3])

    def test_normalize_produces_unit_length(self):
        assert normalize([3.0, 4.0]) == pytest.approx([0.6, 0.8])
        assert normalize([0.0, 0.0]) == [0.0, 0.0]

    def test_top_k_ranks_by_similarity(self):
        vectors = [[1, 0], [0, 1], [0.9, 0.1]]
        ranked = top_k([1, 0], vectors, k=2)
        assert [i for i, _ in ranked] == [0, 2]

    def test_mmr_trades_relevance_for_diversity(self):
        vectors = [[1, 0], [1, 0], [0, 1]]
        assert mmr([1, 0], vectors, k=2, diversity=0.9) == [0, 2]

    def test_rrf_fuses_disagreeing_rankings_symmetrically(self):
        fused = reciprocal_rank_fusion([["a", "b"], ["b", "a"]])
        assert fused["a"] == pytest.approx(fused["b"])

    def test_rrf_weights_must_match(self):
        with pytest.raises(ValueError):
            reciprocal_rank_fusion([["a"], ["b"]], weights=[1.0])
