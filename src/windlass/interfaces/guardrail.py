"""The guardrail interface.

Guardrails inspect what goes into a model and what comes out. Windlass runs them
at two points — ``input`` before the prompt is sent, ``output`` before the
answer is returned — and lets each guardrail decide between three verdicts:

* **allow** — nothing objectionable found.
* **redact** — rewrite the content (mask a credit card, strip a name) and
  continue. This is the default for PII.
* **block** — refuse. Raises :class:`~windlass.core.exceptions.GuardrailViolation`
  unless the pipeline was told to return a refusal message instead.

Implementers override one coroutine, :meth:`Guardrail.acheck`.

Example:
    >>> from windlass.providers.guardrails.rules import RuleGuardrail
    >>> g = RuleGuardrail(pii=True, on_violation="redact")
    >>> g.check("mail me at a@b.com").content
    'mail me at [EMAIL]'
"""

from __future__ import annotations

import abc
from typing import Any

from windlass.core.concurrency import run_sync
from windlass.core.exceptions import GuardrailViolation
from windlass.core.types import GuardrailResult
from windlass.interfaces.base import Component

__all__ = ["Guardrail", "GuardrailChain"]

#: What to do when a rule fires.
VIOLATION_ACTIONS = ("block", "redact", "warn", "allow")


class Guardrail(Component):
    """Abstract input/output safety check.

    Args:
        on_violation: One of ``block``, ``redact``, ``warn`` or ``allow``.
            ``warn`` logs and lets the content through unchanged, which is the
            right setting while you are calibrating a new policy in production.
        stages: Which stages this guardrail runs at — any of ``input`` and
            ``output``.
        name: Component name for traces.
        **config: Policy-specific options.

    Attributes:
        on_violation: The configured action.
        stages: The stages this guardrail participates in.

    Raises:
        ValueError: If ``on_violation`` is not a recognised action.

    Example:
        Implementing a guardrail takes one method::

            class NoShouting(Guardrail):
                provider_name = "no_shouting"

                async def acheck(self, content, *, stage="input", context=None):
                    if content.isupper():
                        return GuardrailResult(
                            allowed=False, content=content,
                            rule="shouting", stage=stage,
                        )
                    return GuardrailResult(allowed=True, content=content, stage=stage)
    """

    kind = "guardrail"
    provider_name: str = "guardrail"

    def __init__(
        self,
        *,
        on_violation: str = "block",
        stages: tuple[str, ...] = ("input", "output"),
        name: str | None = None,
        **config: Any,
    ) -> None:
        if on_violation not in VIOLATION_ACTIONS:
            raise ValueError(
                f"on_violation must be one of {VIOLATION_ACTIONS}, got {on_violation!r}"
            )
        super().__init__(
            name=name or self.provider_name,
            on_violation=on_violation,
            stages=stages,
            **config,
        )
        self.on_violation = on_violation
        self.stages = tuple(stages)

    # -- provider hooks ---------------------------------------------------
    @abc.abstractmethod
    async def acheck(
        self,
        content: str,
        *,
        stage: str = "input",
        context: dict[str, Any] | None = None,
    ) -> GuardrailResult:
        """Inspect content and return a verdict.

        Implementations should report *what they found* and leave the policy
        decision to :attr:`on_violation` — that keeps one detector usable in
        both blocking and redacting configurations.

        Args:
            content: The text to inspect.
            stage: ``"input"`` or ``"output"``.
            context: Extra signals — retrieved chunks, the user id, the tool
                about to be called.

        Returns:
            The verdict, with ``content`` set to the (possibly rewritten) text.
        """

    # -- public API -------------------------------------------------------
    async def avalidate(
        self,
        content: str,
        *,
        stage: str = "input",
        context: dict[str, Any] | None = None,
    ) -> str:
        """Check content and apply the configured policy.

        Args:
            content: The text to check.
            stage: ``"input"`` or ``"output"``.
            context: Extra signals for the check.

        Returns:
            The content to use downstream — unchanged, or redacted.

        Raises:
            GuardrailViolation: When a rule fires and ``on_violation='block'``.

        Example:
            >>> import asyncio
            >>> from windlass.providers.guardrails.rules import RuleGuardrail
            >>> g = RuleGuardrail(banned_words=["secret"], on_violation="redact")
            >>> asyncio.run(g.avalidate("the secret plan"))
            'the [REDACTED] plan'
        """
        if stage not in self.stages:
            return content

        result = await self.acheck(content, stage=stage, context=context)
        if result.allowed and not result.detections:
            return content

        if self.on_violation == "allow":
            return content
        if self.on_violation == "warn":
            self._log.warning("Guardrail %s flagged %s content: %s", self.name, stage, result.rule)
            return content
        if self.on_violation == "redact":
            return result.content or content

        raise GuardrailViolation(
            f"Guardrail {self.name!r} blocked the {stage}: {result.rule or 'policy violation'}",
            stage=stage,
            rule=result.rule,
            detections=result.detections,
        )

    def validate(
        self,
        content: str,
        *,
        stage: str = "input",
        context: dict[str, Any] | None = None,
    ) -> str:
        """Blocking :meth:`avalidate`."""
        return run_sync(self.avalidate(content, stage=stage, context=context))

    def check(
        self,
        content: str,
        *,
        stage: str = "input",
        context: dict[str, Any] | None = None,
    ) -> GuardrailResult:
        """Blocking :meth:`acheck` — returns the verdict without enforcing it."""
        return run_sync(self.acheck(content, stage=stage, context=context))

    def __and__(self, other: Guardrail) -> GuardrailChain:
        """Compose guardrails with ``&``.

        Args:
            other: Guardrail to run after this one.

        Returns:
            A :class:`GuardrailChain`.
        """
        left = self.guards if isinstance(self, GuardrailChain) else [self]
        right = other.guards if isinstance(other, GuardrailChain) else [other]
        return GuardrailChain([*left, *right])

    def __repr__(self) -> str:
        return f"{type(self).__name__}(on_violation={self.on_violation!r})"


class GuardrailChain(Guardrail):
    """Runs several guardrails in order, threading redactions through.

    Each guardrail sees whatever the previous one produced, so a PII redactor
    followed by an injection detector inspects the already-masked text.

    Args:
        guards: The guardrails to run.
        name: Component name for traces.

    Attributes:
        guards: The configured guardrails.

    Example:
        >>> from windlass.providers.guardrails.rules import RuleGuardrail
        >>> chain = RuleGuardrail(pii=True, on_violation="redact") & RuleGuardrail(
        ...     banned_words=["nope"], on_violation="redact"
        ... )
        >>> chain.validate("a@b.com says nope")
        '[EMAIL] says [REDACTED]'
    """

    provider_name = "chain"

    def __init__(self, guards: list[Guardrail] | None = None, *, name: str | None = None) -> None:
        super().__init__(name=name or "chain", on_violation="block")
        self.guards: list[Guardrail] = list(guards or [])
        self.stages = ("input", "output")

    def add(self, guard: Guardrail) -> GuardrailChain:
        """Append a guardrail and return ``self``."""
        self.guards.append(guard)
        return self

    async def acheck(
        self,
        content: str,
        *,
        stage: str = "input",
        context: dict[str, Any] | None = None,
    ) -> GuardrailResult:
        """Aggregate every child's detections without enforcing them."""
        current = content
        detections: list[dict[str, Any]] = []
        rule: str | None = None
        allowed = True
        for guard in self.guards:
            if stage not in guard.stages:
                continue
            result = await guard.acheck(current, stage=stage, context=context)
            detections.extend(result.detections)
            if result.detections and guard.on_violation == "redact":
                current = result.content or current
            if not result.allowed and guard.on_violation == "block":
                allowed = False
                rule = rule or result.rule
        return GuardrailResult(
            allowed=allowed, content=current, detections=detections, rule=rule, stage=stage
        )

    async def avalidate(
        self,
        content: str,
        *,
        stage: str = "input",
        context: dict[str, Any] | None = None,
    ) -> str:
        """Run every guardrail's own policy in sequence."""
        current = content
        for guard in self.guards:
            current = await guard.avalidate(current, stage=stage, context=context)
        return current

    def describe(self) -> dict[str, Any]:
        """Return a summary including each guardrail."""
        return {**super().describe(), "guards": [g.describe() for g in self.guards]}

    def __len__(self) -> int:
        return len(self.guards)

    def __repr__(self) -> str:
        return f"GuardrailChain({' & '.join(g.name for g in self.guards)})"
