# Contributing to Windlass

Thank you for considering it. Windlass is built to be extended, and the most valuable contributions come from people who have just been confused by something.

---

## Setup

```bash
git clone https://github.com/windlass/windlass && cd windlass
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -e ".[dev]"
pre-commit install
```

Verify:

```bash
pytest                            # 300+ tests, offline, a few seconds
ruff check .
black --check .
mypy
windlass doctor
```

All four should be clean before you open a pull request. `pre-commit` runs the first three automatically.

---

## The architecture rule

There is one rule, and it is not negotiable:

> **No module in Windlass may import a concrete provider.**

Everything else follows from it — the tiny core install, the swappable providers, the offline test suite, the plugin system.

In practice:

```python
# Wrong — this pulls chromadb into the import graph for everyone
from windlass.providers.vectordb.chroma import ChromaVectorStore

# Right — resolved by name, imported only when actually used
store = container.component("vectordb", "chroma")
```

Optional dependencies are imported through `windlass.core.lazy.require`, which turns a missing package into an actionable `MissingDependencyError`:

```python
from windlass.core.lazy import require

faiss = require("faiss", extra="faiss", feature="The FAISS vector store")
```

A raw `ImportError` from an optional dependency reaching user code is a bug.

---

## Layout

| Path | Rule |
|---|---|
| `core/` | Framework machinery. Knows nothing about AI. Depends on nothing else in Windlass |
| `interfaces/` | The thirteen contracts. Depends only on `core` |
| `providers/` | Concrete implementations. Registered lazily in `providers/__init__.py` |
| `rag/`, `agent/`, `tools/` | Runtime. Depends on `interfaces`, never on `providers` |
| `api.py` | The public facade |
| `testing.py` | Doubles for people building on Windlass |

Dependencies point inwards. If you find yourself importing "upwards", the design has gone wrong.

---

## Adding a provider

Read `providers/llm/openai.py` first — it is the reference implementation.

Then:

1. **Implement the interface.** One or two async methods, usually.
2. **Register it lazily** in `providers/__init__.py` with a description, aliases and the extras group it needs.
3. **Add the extra** to `pyproject.toml`, and to the `_EXTRAS` map in `cli.py` so `windlass doctor` reports it.
4. **Translate errors** into the Windlass hierarchy, with a `hint` that says what to do.
5. **Return the SDK object** from `native()`.
6. **Write tests** — the pure logic (message translation, schema mapping) should be testable without the SDK installed.

Example of point 4, which is the one people skip:

```python
if name == "RateLimitError":
    return RateLimitError(
        f"Acme rate limit: {message}",
        provider="acme",
        retry_after=advertised_delay,
        original=exc,
    )
```

The retry policy reads `retry_after`, so getting this right is what makes backoff behave.

---

## Code style

**Python 3.12+**, `ruff` and `black` at 100 columns, `mypy` clean.

**Type hints everywhere.** Modern syntax: `str | None`, `list[str]`, `dict[str, Any]`.

**Google-style docstrings** on every public class and function, with `Args`, `Returns`, `Raises` and an `Example`. Add `Performance` or `Note` where a caller could reasonably be surprised.

````python
def chunk(self, documents: Sequence[Document]) -> list[Chunk]:
    """Split documents into chunks.

    Args:
        documents: The documents to split.

    Returns:
        Chunks carrying each document's metadata plus chunk-level annotations.

    Raises:
        IngestionError: When chunking produces nothing.

    Performance:
        CPU-bound strategies see little benefit from concurrency; strategies
        that await a model see a lot.

    Example:
        >>> chunker.chunk([Document(content="hello")])
        [Chunk(content='hello', ...)]
    """
````

**Comments explain *why*.** The code already says what.

```python
# Wrong
# Loop over the chunks
for chunk in chunks:

# Right
# Chunk ids are content hashes, so re-adding the same content upserts
# rather than duplicating — which is what makes ingestion idempotent.
```

**Async-first.** Implement the `a*` coroutine; the blocking form is derived by the base class. Never write the two separately — they drift.

---

## Testing

The suite runs **offline**, in seconds, with no API keys. Keep it that way.

```bash
pytest
pytest -m "not network"
pytest --cov=windlass --cov-report=term-missing
```

**Use the fakes.**

```python
from windlass.testing import fake_rag, fake_agent, call, capture_spans, isolated_registry
```

**Test behaviour, not wording.** Assert on retrieval, on tool calls, on error paths. Never on a model's exact phrasing.

**Isolate global state.** `isolated_registry()` for registrations; the `reset_settings` fixture for configuration.

**Mark what needs marking:**

```python
@pytest.mark.network       # needs credentials
@pytest.mark.optional      # needs an optional extra
@pytest.mark.integration   # spans several components
@pytest.mark.slow          # over a second
```

Anything touching a live API must be skippable.

---

## Documentation

Documentation is a deliverable, not an afterthought.

- The **API reference** is generated from docstrings, so writing a good docstring updates the site.
- **Prose docs** live in `docs/`. Build with `mkdocs serve`.
- **Examples** must run. If one needs an extra, it should say so and exit cleanly rather than crashing.

```bash
pip install -e ".[docs]"
mkdocs serve
```

The single most valuable documentation contribution is a fix to something that confused you. You are the last person who will ever see it with fresh eyes.

---

## Pull requests

**One concern per PR.** A provider, a bug fix, a doc improvement. Not three.

**Explain the why.** What problem, what alternatives, why this approach.

**Include tests.** A bug fix without a regression test invites the bug back.

**Update the changelog.** Add an entry under `## [Unreleased]`.

Before opening:

```bash
pytest && ruff check . && black --check . && mypy
```

### Commit messages

```
component: short imperative summary

Longer explanation of why, if the diff does not make it obvious.

Fixes #123
```

Prefixes: `core:` `llm:` `rag:` `agent:` `tools:` `mcp:` `docs:` `tests:` `ci:`

---

## Reporting bugs

Include:

```bash
windlass doctor
windlass config          # secrets are masked
python -c "import windlass, sys; print(windlass.__version__, sys.version)"
```

Plus the full traceback, and a minimal reproduction — ideally one that uses the `fake` provider so anyone can run it.

Windlass errors carry structured context worth including:

```python
except WindlassError as exc:
    print(exc.message, exc.hint, exc.context)
```

---

## What is most wanted

**Provider adapters.** More models, more vector stores, more loaders. The pattern is consistent and each one is a self-contained contribution.

**Documentation.** Especially from someone who has just been confused.

**Real-world examples.** A pattern that worked for you is worth more than a synthetic demo.

**Bug reports with reproductions.** A `fake`-provider reproduction is worth ten paragraphs of description.

### Please discuss first

Anything that changes an interface, adds a component kind, or adds a **core** dependency. The core install being tiny is a load-bearing property of the design — a new core dependency needs a strong argument.

---

## Releasing

For maintainers:

1. Update `CHANGELOG.md` — move `Unreleased` into a version heading.
2. Bump `src/windlass/_version.py`.
3. `pytest && ruff check . && black --check . && mypy`
4. `python -m build && twine check dist/*`
5. Tag `vX.Y.Z` and push. CI publishes to PyPI.

Semantic versioning. Pre-1.0, minor versions may break the public API; every break gets a changelog entry and a migration note.

---

## Code of conduct

Be decent. Assume good faith. Critique code, not people. Remember that the person asking an obvious question is the person your documentation failed.

---

## License

Contributions are licensed under Apache 2.0, matching the project.
