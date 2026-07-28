# 04 — Custom components

Write your own chunker, retriever, guardrail and LLM provider. Each one registers the same way the built-ins do, and composes the same way — because there is no privileged path.

## Run

```bash
pip install windlass
python examples/04_custom_components/main.py
```

## What it shows

1. **A chunker** — one method, `split_text`.
2. **A retriever** backed by something that is not a vector database. This is the most common real plugin.
3. **A guardrail** that reports detections and leaves the policy to `on_violation`.
4. **An LLM provider** — one method, `agenerate`.
5. All four **used by name**, exactly like built-ins.
6. The custom guardrail **composed** with a built-in one using `&`.
7. The custom retriever **fused** into hybrid retrieval alongside dense search.

## The point

After registration, your component is indistinguishable from a built-in:

```python
Windlass.rag().chunker("by-sentence").retriever("keyword-index").llm("shouty")
```

It appears in `Windlass.list()`, it takes configuration, it is traced, it composes into hybrid retrieval, and it works with everything else. Windlass resolves its own providers through the same registry — there is no separate mechanism for "official" components.

## How little code it takes

The chunker is the whole story:

```python
@register.chunker("by-sentence")
class SentenceChunker(Chunker):
    """Groups sentences into fixed-size chunks."""

    def split_text(self, text: str) -> list[str]:
        sentences = split_sentences(text)
        step = self.sentences_per_chunk
        return [" ".join(sentences[i:i + step]) for i in range(0, len(sentences), step)]
```

Document iteration, metadata propagation, offset tracking, id assignment, under-sized chunk merging and bounded concurrency are all provided by the base class. You write the part that is actually yours.

## What each kind requires

| Kind | Implement |
|---|---|
| `llm` | `agenerate`; optionally `astream_generate` |
| `embedding` | `aembed_texts` |
| `chunker` | `split_text` (or `asplit_text` if you need to await something) |
| `retriever` | `aretrieve_chunks`; optionally `aindex` |
| `vectordb` | `aadd`, `asearch`, `adelete`, `acount` |
| `guardrail` | `acheck` |
| `memory` | `aadd`, `aget` |
| `loader` | `aload_source` |
| `preprocessor` | `aprocess_one` |
| `evaluator` | `aevaluate_sample` |
| `tracer` | `start_span` |
| `tool` | `acall` — or just use `@tool` |
| `mcp` | `aconnect`, `alist_tools`, `acall_tool` |

Always implement the **async** method. The blocking API is derived from it, so the two cannot drift.

## Guardrails: report, don't decide

```python
return GuardrailResult(
    allowed=not found,
    content=redacted,        # always supply the sanitised form
    detections=[...],
    rule="competitor" if found else None,
)
```

Report *what you found* and let `on_violation` decide what happens. That keeps one detector usable in blocking, redacting and warning configurations — which matters, because you will want `warn` while calibrating and `block` afterwards.

## Shipping it to other people

The decorator only works once your module is imported. To make a component appear in other people's Windlass, publish an entry point:

```toml
[project.entry-points."windlass.retriever"]
company-search = "my_package.retrievers:CompanySearch"
```

```bash
pip install my-package
```

```python
Windlass.rag().retriever("company-search")     # it is simply there
```

The value is a dotted path, so your module is imported only when somebody actually uses the component.

## Next

The [plugin guide](../../docs/guides/plugins.md) covers packaging, entry points, testing against the interface contract, and a checklist for a production-quality plugin.
