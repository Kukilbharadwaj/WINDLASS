# 07 — Human in the loop

Approval gating for actions an agent should not take on its own.

## Run

```bash
pip install windlass
python examples/07_human_in_the_loop/main.py
```

## The problem

An agent that can issue refunds will eventually issue a wrong one. The mitigation is not a better prompt — it is a human in the path for anything irreversible.

```python
@tool(requires_approval=True)
def issue_refund(order_id: str, amount_cents: int) -> dict:
    """Issue a refund to the customer's original payment method."""
```

## What it shows

1. **The pause** — the run stops *before* the gated tool executes, and is checkpointed.
2. **The queue** — pending calls are inspectable, so a UI or Slack message can present them.
3. **Approval** — the run resumes where it stopped, without re-paying for tokens already spent.
4. **Rejection with feedback** — the reason reaches the model as the tool result, so it changes approach rather than repeating itself.
5. **Edited arguments** — approve the intent, fix the parameters. The most common real case.
6. **Durability** — with SQLite, the approval queue survives a process restart, so it can be genuinely asynchronous.
7. **Ungated tools** — read-only tools still run without asking.

## The assertion that matters

```
=== 1. Approval requested ===
  paused on thread: ticket-77
  wants to call:   issue_refund({'order_id': 'A-1234', 'amount_cents': 4900})
  executed so far: []   <- nothing happened
```

`EXECUTED` is empty. The tool was **not** called. That is the property worth testing:

```python
def test_nothing_happens_without_approval():
    with pytest.raises(AgentInterrupt):
        agent.run("refund it", thread_id="t1")
    assert EXECUTED == []
```

## The three resume paths

```python
agent.resume("ticket-77", approved=True)

agent.resume("ticket-78", approved=False,
             feedback="Outside the 30-day window. Offer store credit instead.")

agent.resume("ticket-79", approved=True,
             edited_arguments={"call-1": {"amount_cents": 2500}})
```

The third is the one you will use most. The agent identified the right action but got a parameter wrong, and a human is better placed to fix the parameter than to re-prompt.

## Why checkpointing is required

A paused run has to be stored somewhere before it can be resumed. Approval interrupts therefore need a checkpointer:

```python
.checkpoint()                              # in-process — fine for a demo
.checkpoint("sqlite", path="./state.db")   # durable — required in production
```

With SQLite the approval queue is just a table of thread ids, and approval can happen minutes or days later, from a different process:

```python
for thread in checkpointer.threads():
    if pending := agent.pending_approvals(thread):
        notify_reviewer(thread, pending)
```

## Gating everything

```python
agent.human_in_the_loop()
```

Pauses before *every* tool call, not just those whose tools declare it. Useful while you are still learning what an agent does with a new tool set.

## What to gate

Anything that:

- moves money
- sends a message to a person
- deletes or overwrites data
- changes access or permissions
- is expensive to undo

Read-only tools should stay ungated, or approval fatigue sets in and reviewers start clicking yes without reading — which is worse than no gate at all, because it looks like control.
