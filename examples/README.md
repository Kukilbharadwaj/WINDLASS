# Examples

Every example here is runnable. The first four need **no API key and no extras** — they use the dependency-free providers, so you can see the whole framework work before installing anything heavy.

```bash
pip install windlass
python examples/01_rag_basics/main.py
```

## The examples

| | Example | Needs | What it shows |
|---|---|---|---|
| 01 | [RAG basics](01_rag_basics/) | nothing | Ingest, retrieve, generate. Inspecting what was retrieved |
| 02 | [Agent with tools](02_agent_tools/) | nothing | The reason/act loop, schema generation, parallel tools, error recovery |
| 03 | [Hybrid retrieval](03_hybrid_retrieval/) | nothing | Why dense-only search misses identifiers, and what fusion fixes |
| 04 | [Custom components](04_custom_components/) | nothing | Writing and registering your own chunker, retriever and guardrail |
| 05 | [Production RAG](05_production_rag/) | `[openai,rag]` | Guardrails, PII redaction, evaluation, persistence, tracing |
| 06 | [Multi-agent research](06_multi_agent/) | `[openai]` | Supervisor, specialists, broadcast, pipeline |
| 07 | [Human in the loop](07_human_in_the_loop/) | nothing | Approval gating, interrupts, resume, edited arguments |
| 08 | [MCP integration](08_mcp/) | nothing* | Binding MCP tools; `[mcp]` for real servers |

Each directory has a `README.md` explaining what it does, what to expect, and what to change.

## Running them all

```bash
for dir in examples/*/; do
    echo "=== $dir ==="
    python "$dir/main.py" || echo "SKIPPED (missing extra)"
done
```

Examples that need an extra print a clear message and exit cleanly rather than failing.

## Using them as a starting point

Examples 01–04 are teaching material — read them top to bottom.

Example 05 is the one to copy for a real project. It is a complete production pipeline: configuration, guardrails, PII handling, persistence, an evaluation gate and tracing, in about a hundred lines.
