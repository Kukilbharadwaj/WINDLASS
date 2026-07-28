"""Multi-agent research — specialists coordinated by a supervisor.

Runs offline with scripted models so the delegation path is deterministic.
Set OPENAI_API_KEY to let real models decide the routing for themselves.

    python examples/06_multi_agent/main.py
"""

from __future__ import annotations

import os

from windlass import Windlass, ToolCall, tool

# ---------------------------------------------------------------------------
# A small knowledge base, exposed to one specialist as a RAG-backed tool.
# ---------------------------------------------------------------------------

INTERNAL_DOCS = [
    "Our data retention policy keeps customer records for 24 months after account closure.",
    "The EU AI Act classifies our recommendation engine as limited risk, "
    "requiring transparency notices.",
    "Model changes affecting user-facing output need sign-off from the AI governance board.",
]

LIVE = bool(os.getenv("OPENAI_API_KEY"))


def internal_search_tool(rag):
    """Wrap a RAG pipeline as a tool an agent can call.

    A RAG pipeline is just a tool — nothing special is needed to combine them.
    """

    @tool(name="search_internal_docs")
    def search_internal_docs(query: str) -> list[dict]:
        """Search internal policy documentation.

        Args:
            query: What to look for.
        """
        return [
            {"content": hit.chunk.content, "score": round(hit.score, 3)}
            for hit in rag.search(query, k=2)
        ]

    return search_internal_docs


@tool
def web_search(query: str) -> list[dict]:
    """Search the web for current information.

    Args:
        query: What to search for.
    """
    # Stand-in for a real search API.
    return [
        {
            "title": "EU AI Act — implementation timeline",
            "url": "https://example.org/ai-act",
            "snippet": "Obligations for limited-risk systems apply from August 2026.",
        }
    ]


def build_team():
    """Build three specialists and a supervisor to coordinate them."""
    # A RAG pipeline the analyst can query.
    rag = Windlass.rag().llm("fake", responses=["ok"]).retriever("hybrid").top_k(2)
    for text in INTERNAL_DOCS:
        rag.ingest_text(text, source="policies")

    researcher = (
        Windlass.agent()
        .llm(
            "gpt-4o-mini" if LIVE else "fake",
            **(
                {}
                if LIVE
                else {
                    "responses": [
                        "",
                        "Obligations for limited-risk systems apply from August 2026.",
                    ],
                    "tool_calls": [
                        [ToolCall(name="web_search", arguments={"query": "EU AI Act timeline"})],
                        [],
                    ],
                }
            ),
        )
        .tool(web_search)
        .system("You find source material. Report findings with the URL you took each from.")
        .name("researcher")
    )

    analyst = (
        Windlass.agent()
        .llm(
            "gpt-4o-mini" if LIVE else "fake",
            **(
                {}
                if LIVE
                else {
                    "responses": [
                        "",
                        "Our recommendation engine is limited risk and needs transparency notices.",
                    ],
                    "tool_calls": [
                        [
                            ToolCall(
                                name="search_internal_docs",
                                arguments={"query": "AI Act classification"},
                            )
                        ],
                        [],
                    ],
                }
            ),
        )
        .tool(internal_search_tool(rag))
        .system("You interpret findings against internal policy. Flag any disagreement.")
        .name("analyst")
    )

    writer = (
        Windlass.agent()
        .llm(
            "gpt-4o-mini" if LIVE else "fake",
            **(
                {}
                if LIVE
                else {
                    "responses": [
                        "Briefing: limited-risk classification, transparency "
                        "notices required from August 2026."
                    ]
                }
            ),
        )
        .system("You write briefings: two-sentence summary, then detail, then sources.")
        .name("writer")
    )

    supervisor_llm = Windlass.llm(
        "gpt-4o" if LIVE else "fake",
        **(
            {}
            if LIVE
            else {
                "responses": [
                    "",
                    "",
                    "The Act classifies us as limited risk; notices are required from August 2026.",
                ],
                "tool_calls": [
                    [ToolCall(name="researcher", arguments={"task": "EU AI Act timeline"})],
                    [
                        ToolCall(
                            name="analyst", arguments={"task": "how does this affect our engine"}
                        )
                    ],
                    [],
                ],
            }
        ),
    )

    boss = Windlass.supervisor(
        {"researcher": researcher, "analyst": analyst, "writer": writer},
        llm=supervisor_llm,
        descriptions={
            "researcher": "Searches the web. Use for anything requiring external, "
            "current information.",
            "analyst": "Searches internal policy documentation and interprets "
            "findings against it.",
            "writer": "Turns notes into a finished briefing. Use last.",
        },
        max_iterations=8,
    )
    return boss


def main() -> None:
    print(f"Provider: {'openai' if LIVE else 'fake (offline, scripted routing)'}\n")
    boss = build_team()

    # -------------------------------------------------------------------
    # 1. Let the supervisor route. Delegation is ordinary tool calling —
    #    each specialist appears to the supervisor as a tool.
    # -------------------------------------------------------------------
    print("=== Supervisor decides ===")
    response = boss.run("How does the EU AI Act affect our recommendation engine?")
    print(f"\nAnswer: {response.output}\n")
    print("Delegation trace:")
    for step in response.steps:
        for call, result in zip(step.tool_calls, step.tool_results, strict=True):
            task = call.arguments.get("task", "")
            print(f"  -> {call.name}({task!r})")
            print(f"       {result.content[:78]}")
    print(f"\nSteps: {len(response.steps)}   Tokens: {response.usage.total_tokens}")

    # -------------------------------------------------------------------
    # 2. Broadcast: everyone works on the same task, concurrently.
    #    Costs the slowest specialist, not the sum.
    # -------------------------------------------------------------------
    print("\n=== Broadcast ===")
    for name, result in boss.broadcast("Summarise the compliance risk in one line").items():
        print(f"  {name:<12} {result.output[:70]}")

    # -------------------------------------------------------------------
    # 3. Pipeline: a fixed sequence, each stage seeing the previous output.
    # -------------------------------------------------------------------
    print("\n=== Pipeline ===")
    final = boss.pipeline(
        "Brief us on the EU AI Act's effect on recommendations",
        ["researcher", "analyst", "writer"],
    )
    print(f"  final: {final.output}")
    print(f"  stages: {' -> '.join(step.node for step in final.steps)}")

    # -------------------------------------------------------------------
    # 4. What the supervisor actually sees.
    # -------------------------------------------------------------------
    print("\n=== The supervisor's tools ===")
    for description in boss.tools.describe():
        print(f"  {description['name']:<12} {description['description'][:64]}")
    print("\n  These descriptions *are* the routing logic — the supervisor")
    print("  chooses purely on this text. Write them like prompts.")


if __name__ == "__main__":
    main()
