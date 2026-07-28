# Examples

Every example in the [`examples/`](https://github.com/Kukilbharadwaj/WINDLASS/tree/main/examples) directory runs on a bare `pip install windlass` — **no API key and no extras**.

```bash
pip install windlass
python examples/01_rag_basics/main.py
```

| | Example | Better with | Shows |
|---|---|---|---|
| 01 | RAG basics | — | Ingest, retrieve, generate; inspecting what was retrieved; idempotent ingestion |
| 02 | Agent with tools | — | The reason/act loop, schema generation, parallel tools, error recovery |
| 03 | Hybrid retrieval | — | Why dense-only search misses identifiers, and what RRF fixes |
| 04 | Custom components | — | Writing and registering your own chunker, retriever, guardrail and LLM |
| 05 | Production RAG | `[openai,rag]` + a key | Guardrails, PII redaction, dedup, persistence, evaluation gate, tracing |
| 06 | Multi-agent research | `[openai]` + a key | Supervisor, specialists, broadcast, pipeline, RAG-as-a-tool |
| 07 | Human in the loop | — | Approval gating, interrupts, resume, rejection feedback, edited arguments |
| 08 | MCP integration | — | Binding MCP tools, namespacing, graceful degradation |

Examples 05 and 06 fall back to the offline providers when no key is configured, so they still run and still show the structure — they just script the model instead of asking a real one. The "better with" column is what they upgrade to, never what they require.

Each directory has a `README.md` explaining the output and what to change.

---

## Which one to read

**New to Windlass?** Start at 01 and work forward. They build on each other.

**Building something real?** Copy 05. It is a complete production pipeline — configuration, guardrails, PII handling, persistence, an evaluation gate and tracing — in about a hundred lines.

**Extending Windlass?** Read 04, then the [plugin guide](guides/plugins.md).

**Worried about agent safety?** Read 07. The property it demonstrates — nothing executes before a human approves — is the one worth testing in your own system.

---

## The teaching moments

Each example is built around one thing that is easy to get wrong.

### 01 — retrieval quality is answer quality

The example prints what was retrieved *before* it prints the answer, because that is the order you should debug in. If the right chunk is not in `answer.contexts`, no prompt change will help.

It also shows `strict()` paired with `min_score()`. Strict mode alone rarely fires: dense retrieval returns its nearest neighbours however far away they are, so the result set is almost never empty.

### 02 — tool failures belong to the model

```
--- A tool that fails ---
  output: That service is unavailable right now.
    step 0: flaky_service({}) -> [ERROR] ConnectionError: upstream returned 503
```

The exception did not propagate. It became a tool result the model could respond to. An exception that unwound the agent loop would remove the model's chance to recover, and recovering is the point of an agent.

### 03 — dense-only search has a blind spot

Three queries, three retrievers, side by side. `E1042` is invisible to embeddings; "my login keeps failing" is invisible to BM25. If your corpus contains identifiers, use hybrid retrieval.

### 04 — your component is not a second-class citizen

Four custom components, all used by name, one of them fused into hybrid retrieval alongside a built-in. Windlass resolves its own providers through the same registry.

### 05 — redact before you embed

The corpus contains an email address, a phone number and a duplicated document. Watch what is actually stored. Once personal data is embedded, deleting a row does not undo the copies in your backups.

### 06 — specialist descriptions are the routing logic

```python
"researcher": "Searches the web. Use for anything requiring external, current information."
```

The supervisor chooses purely on this text. Write it like a prompt.

### 07 — nothing runs before approval

```
wants to call:   issue_refund({'order_id': 'A-1234', 'amount_cents': 4900})
executed so far: []   <- nothing happened
```

That empty list is the assertion worth writing in your own tests.

### 08 — a dead server should not stop the agent

```
bound tools: ['filesystem_list_files', 'filesystem_read_file']
answer:      Still working with the tools I have.
```

MCP servers are subprocesses and network endpoints. They will be unavailable sometimes.

---

## Running them all

```bash
for dir in examples/0[1-8]_*/; do
    echo "=== $dir ==="
    python "$dir/main.py"
done
```

CI runs exactly this on every commit, against a bare `pip install -e .`, so an example that stops working fails the build.
