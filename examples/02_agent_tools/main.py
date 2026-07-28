"""Agents and tools — the reason/act loop, offline.

Runs with no API key and no optional dependencies. The model is scripted, so the
tool-use path is deterministic and you can see exactly what the loop does.

    python examples/02_agent_tools/main.py
"""

from __future__ import annotations

import json
from typing import Literal

from windlass import ToolCall, Windlass, tool

# ---------------------------------------------------------------------------
# Tools. The schema comes from the type hints; the description from the
# docstring. There is no second source of truth to keep in sync.
# ---------------------------------------------------------------------------

WEATHER = {"Paris": 18, "London": 14, "Cairo": 33}


@tool
def get_weather(city: str, units: Literal["celsius", "fahrenheit"] = "celsius") -> dict:
    """Look up the current temperature in a city.

    Args:
        city: City name, e.g. "Paris".
        units: Temperature scale to report.
    """
    celsius = WEATHER.get(city)
    if celsius is None:
        raise KeyError(f"No weather station for {city!r}")
    value = celsius if units == "celsius" else round(celsius * 9 / 5 + 32)
    return {"city": city, "temperature": value, "units": units}


@tool
def convert_currency(amount: float, source: str, target: str) -> dict:
    """Convert an amount between two currencies.

    Args:
        amount: How much to convert.
        source: Three-letter source currency code.
        target: Three-letter target currency code.
    """
    rates = {("USD", "EUR"): 0.92, ("EUR", "USD"): 1.09}
    rate = rates.get((source, target))
    if rate is None:
        raise ValueError(f"No rate for {source}->{target}")
    return {"amount": round(amount * rate, 2), "currency": target, "rate": rate}


@tool(timeout=5.0)
def flaky_service() -> str:
    """Query a service that is currently down."""
    raise ConnectionError("upstream returned 503")


def main() -> None:
    # -------------------------------------------------------------------
    # 1. The generated schema — this is what the model actually sees.
    # -------------------------------------------------------------------
    print("--- Generated schema ---")
    print(json.dumps(get_weather.schema()["function"], indent=2))

    print("\n--- Same tool, other providers ---")
    print("anthropic:", list(get_weather.schema(style="anthropic")))
    print("gemini:   ", list(get_weather.schema(style="gemini")))

    # -------------------------------------------------------------------
    # 2. A single tool call, then an answer. Two model turns.
    # -------------------------------------------------------------------
    print("\n--- One tool call ---")
    agent = (
        Windlass.agent()
        .llm(
            "fake",
            responses=["", "It is 18°C in Paris."],
            tool_calls=[[ToolCall(name="get_weather", arguments={"city": "Paris"})], []],
        )
        .tool(get_weather)
    )
    response = agent.run("What is the weather in Paris?")
    _trace(response)

    # -------------------------------------------------------------------
    # 3. Two tools in one turn. Windlass runs them concurrently — models
    #    routinely ask for several independent lookups at once, and
    #    running those serially triples the wait for no reason.
    # -------------------------------------------------------------------
    print("\n--- Parallel tool calls ---")
    agent = (
        Windlass.agent()
        .llm(
            "fake",
            responses=["", "Paris is 18°C and London is 14°C."],
            tool_calls=[
                [
                    ToolCall(name="get_weather", arguments={"city": "Paris"}),
                    ToolCall(name="get_weather", arguments={"city": "London"}),
                ],
                [],
            ],
        )
        .tool(get_weather)
    )
    _trace(agent.run("Compare the weather in Paris and London"))

    # -------------------------------------------------------------------
    # 4. A failing tool. The exception is reported *to the model*, not
    #    raised — which is what lets it recover or explain.
    # -------------------------------------------------------------------
    print("\n--- A tool that fails ---")
    agent = (
        Windlass.agent()
        .llm(
            "fake",
            responses=["", "That service is unavailable right now."],
            tool_calls=[[ToolCall(name="flaky_service")], []],
        )
        .tool(flaky_service)
    )
    _trace(agent.run("Check the service"))

    # -------------------------------------------------------------------
    # 5. Memory makes follow-up questions work.
    # -------------------------------------------------------------------
    print("\n--- Memory across turns ---")
    agent = (
        Windlass.agent()
        .llm("fake", responses=["Nice to meet you, Ada.", "Your name is Ada."])
        .memory("buffer")
    )
    print(" ", agent.run("My name is Ada.", thread_id="chat").output)
    print(" ", agent.run("What is my name?", thread_id="chat").output)
    print(f"  transcript: {len(agent.build().memory.get(thread_id='chat'))} messages")

    # -------------------------------------------------------------------
    # 6. Guardrails on the way out.
    # -------------------------------------------------------------------
    print("\n--- Output guardrail ---")
    agent = (
        Windlass.agent()
        .llm("fake", responses=["You can reach me at ada@example.com"])
        .guardrails(on_violation="redact")
    )
    print(" ", agent.run("What is your email?").output)

    # -------------------------------------------------------------------
    # 7. Streaming.
    # -------------------------------------------------------------------
    print("\n--- Streaming ---")
    agent = Windlass.agent().llm("fake", responses=["This arrives token by token."])
    print("  ", end="")
    for event in agent.stream("go"):
        if event.type == "text":
            print(event.delta, end="", flush=True)
    print()

    # -------------------------------------------------------------------
    # 8. Tools are still ordinary functions.
    # -------------------------------------------------------------------
    print("\n--- Direct invocation ---")
    print("  ", convert_currency(100.0, "USD", "EUR"))
    print("  ", convert_currency.run(amount=100.0, source="USD", target="EUR").data)


def _trace(response) -> None:
    """Print an agent run as a readable trace."""
    print(f"  output: {response.output}")
    for step in response.steps:
        for call, result in zip(step.tool_calls, step.tool_results, strict=True):
            status = "ERROR" if result.is_error else "ok"
            print(
                f"    step {step.index}: {call.name}({call.arguments}) "
                f"-> [{status}] {result.content}"
            )
    print(f"  steps: {len(response.steps)}  tokens: {response.usage.total_tokens}")


if __name__ == "__main__":
    main()
