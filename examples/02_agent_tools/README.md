# 02 — Agents and tools

The reason/act loop, made visible. The model is scripted, so the tool-use path is deterministic and you can watch exactly what happens.

## Run

```bash
pip install windlass
python examples/02_agent_tools/main.py
```

## What it shows

1. **Schema generation** — what the model actually sees, derived from type hints and the docstring.
2. **Provider dialects** — the same tool rendered for OpenAI, Anthropic and Gemini.
3. **The loop** — turn one emits a tool call, turn two answers having seen the result.
4. **Parallel tools** — two calls in one turn, executed concurrently.
5. **Failure recovery** — a tool that raises becomes an error result the model can respond to.
6. **Memory** — follow-up questions resolving against earlier turns.
7. **Output guardrails** — PII redacted on the way out.
8. **Streaming**.
9. **Tools as plain functions** — still directly callable and directly testable.

## The important bit

Look at the failing-tool section:

```
--- A tool that fails ---
  output: That service is unavailable right now.
    step 0: flaky_service({}) -> [ERROR] ConnectionError: upstream returned 503
  steps: 2  tokens: ...
```

The `ConnectionError` did **not** propagate. It was captured, turned into a tool result, and handed back to the model — which is what allows the model to explain the problem or try something else.

An exception that unwound the agent loop would remove the model's chance to recover, and recovering is the entire point of an agent.

Direct invocation behaves the opposite way, because there is no model to recover:

```python
flaky_service.run()      # raises ToolExecutionError
```

## Why the schema matters

```json
{
  "name": "get_weather",
  "description": "Look up the current temperature in a city.",
  "parameters": {
    "type": "object",
    "properties": {
      "city": {"type": "string", "description": "City name, e.g. \"Paris\"."},
      "units": {"enum": ["celsius", "fahrenheit"], "type": "string",
                "description": "Temperature scale to report."}
    },
    "required": ["city"],
    "additionalProperties": false
  }
}
```

The model chooses tools from this text. `Literal` became an enum, the docstring's `Args:` entries became per-parameter descriptions, and `units` is optional because it has a default.

Treat tool descriptions as prompts, because that is what they are.

## Things to try

Point it at a real model and let it decide for itself:

```python
agent = Windlass.agent().llm("gpt-4o-mini").tool(get_weather, convert_currency)
print(agent.run("Is it warmer in Cairo than in London, and by how much?"))
```

Then watch the trace:

```python
agent.observe("console")
```

Force serial tool execution and compare the timings:

```python
agent.parallel_tools(False)
```
