"""Human in the loop — approval gating, interrupts, resume and edited arguments.

Runs with no API key and no optional dependencies.

    python examples/07_human_in_the_loop/main.py
"""

from __future__ import annotations

import contextlib
import tempfile
from pathlib import Path

from windlass import AgentInterrupt, ToolCall, Windlass, tool

# ---------------------------------------------------------------------------
# The audit log. In a real system these are irreversible side effects: money
# moved, an email sent, a row deleted.
# ---------------------------------------------------------------------------

EXECUTED: list[dict] = []


@tool
def lookup_order(order_id: str) -> dict:
    """Look up an order. Safe — read only, no approval needed.

    Args:
        order_id: The order to look up.
    """
    return {"order_id": order_id, "total_cents": 4900, "status": "delivered"}


@tool(requires_approval=True)
def issue_refund(order_id: str, amount_cents: int) -> dict:
    """Issue a refund to the customer's original payment method.

    Args:
        order_id: The order to refund.
        amount_cents: How much to refund, in cents.
    """
    EXECUTED.append({"order_id": order_id, "amount_cents": amount_cents})
    return {"refunded": amount_cents, "order_id": order_id}


def build_agent(state_path: Path):
    """An agent that wants to refund an order, gated on human approval."""
    return (
        Windlass.agent()
        .llm(
            "fake",
            responses=["", "I have issued the refund."],
            tool_calls=[
                [
                    ToolCall(
                        id="call-1",
                        name="issue_refund",
                        arguments={"order_id": "A-1234", "amount_cents": 4900},
                    )
                ],
                [],
            ],
        )
        .tool(lookup_order, issue_refund)
        # Approval interrupts need somewhere to store the paused run. SQLite
        # means the approval queue can be asynchronous and survive a restart.
        .checkpoint("sqlite", path=str(state_path))
    )


def main() -> None:
    state_path = Path(tempfile.mkdtemp()) / "state.db"

    # -------------------------------------------------------------------
    # 1. The run pauses before the gated tool. Nothing has executed.
    # -------------------------------------------------------------------
    print("=== 1. Approval requested ===")
    agent = build_agent(state_path)
    try:
        agent.run("Refund order A-1234 in full", thread_id="ticket-77")
        print("  (no interrupt — unexpected)")
    except AgentInterrupt as pause:
        print(f"  paused on thread: {pause.thread_id}")
        for call in pause.payload:
            print(f"  wants to call:   {call['name']}({call['arguments']})")
    print(f"  executed so far: {EXECUTED}   <- nothing happened\n")

    # -------------------------------------------------------------------
    # 2. Inspect the queue. In a real system this is a UI or a Slack message.
    # -------------------------------------------------------------------
    print("=== 2. The approval queue ===")
    for call in agent.pending_approvals("ticket-77"):
        print(f"  {call.id}  {call.name}  {call.arguments}")
    print()

    # -------------------------------------------------------------------
    # 3. Approve. The run resumes exactly where it stopped — it does not
    #    re-pay for the tokens already spent.
    # -------------------------------------------------------------------
    print("=== 3. Approved ===")
    response = agent.resume("ticket-77", approved=True)
    print(f"  output:   {response.output}")
    print(f"  executed: {EXECUTED}\n")

    # -------------------------------------------------------------------
    # 4. Rejection. The feedback reaches the model as the tool result, so it
    #    changes approach rather than repeating itself.
    # -------------------------------------------------------------------
    print("=== 4. Rejected, with feedback ===")
    EXECUTED.clear()
    agent = build_agent(state_path)
    with contextlib.suppress(AgentInterrupt):  # we expect the pause
        agent.run("Refund order A-1234 in full", thread_id="ticket-78")

    response = agent.resume(
        "ticket-78",
        approved=False,
        feedback="Outside the 30-day window. Offer store credit instead.",
    )
    print(f"  executed: {EXECUTED}   <- still nothing")
    for message in response.messages:
        if message.role.value == "tool":
            print(f"  the model was told: {message.content}")
    print()

    # -------------------------------------------------------------------
    # 5. Approve the intent, fix the parameters. This is the case that
    #    matters most in practice: the agent was nearly right.
    # -------------------------------------------------------------------
    print("=== 5. Approved with edited arguments ===")
    EXECUTED.clear()
    agent = build_agent(state_path)
    with contextlib.suppress(AgentInterrupt):
        agent.run("Refund order A-1234 in full", thread_id="ticket-79")

    agent.resume(
        "ticket-79",
        approved=True,
        edited_arguments={"call-1": {"order_id": "A-1234", "amount_cents": 2500}},
    )
    print(f"  executed: {EXECUTED}   <- a partial refund, as the human decided\n")

    # -------------------------------------------------------------------
    # 6. Durability. A different process can approve the run.
    # -------------------------------------------------------------------
    print("=== 6. Checkpoints survive the process ===")
    from windlass.agent.checkpoint import SQLiteCheckpointer

    saver = SQLiteCheckpointer(state_path)
    print(f"  threads on disk: {saver.threads()}")
    snapshot = saver.get("ticket-79")
    print(f"  messages in the newest snapshot: {len(snapshot['messages'])}")
    print(f"  history depth: {len(saver.history('ticket-79'))} checkpoints\n")

    # -------------------------------------------------------------------
    # 7. Ungated tools still run without asking.
    # -------------------------------------------------------------------
    print("=== 7. Read-only tools are not gated ===")
    EXECUTED.clear()
    reader = (
        Windlass.agent()
        .llm(
            "fake",
            responses=["", "Order A-1234 was delivered and cost $49.00."],
            tool_calls=[[ToolCall(name="lookup_order", arguments={"order_id": "A-1234"})], []],
        )
        .tool(lookup_order, issue_refund)
        .checkpoint("sqlite", path=str(state_path))
    )
    print(f"  {reader.run('What is the status of order A-1234?', thread_id='t-80').output}")
    print("  (no interrupt: lookup_order does not require approval)")


if __name__ == "__main__":
    main()
