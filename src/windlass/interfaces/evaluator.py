"""The evaluation interface.

Evaluation answers "is this system actually any good?", and it is the part
teams skip until something breaks in production. Windlass makes it a first-class
component so a regression suite is a few lines rather than a project.

The unit of work is a :class:`EvalSample` — a question, the system's answer, the
contexts it retrieved, and optionally a reference answer. Metrics score samples;
evaluators run metrics over a dataset and aggregate.

Implementers override one coroutine, :meth:`Evaluator.aevaluate_sample`.

Example:
    >>> from windlass.providers.evaluation.builtin import BuiltinEvaluator
    >>> from windlass.interfaces.evaluator import EvalSample
    >>> ev = BuiltinEvaluator(metrics=["exact_match"])
    >>> sample = EvalSample(question="2+2?", answer="4", reference="4")
    >>> ev.evaluate([sample]).summary["exact_match"]
    1.0
"""

from __future__ import annotations

import abc
from collections.abc import Sequence
from typing import Any

from pydantic import Field

from windlass.core.concurrency import gather_bounded, run_sync
from windlass.core.config import settings
from windlass.core.types import EvaluationReport, EvaluationResult, RAGAnswer, WindlassModel
from windlass.interfaces.base import Component

__all__ = ["EvalSample", "Evaluator"]


class EvalSample(WindlassModel):
    """One evaluated interaction.

    Attributes:
        id: Sample identifier, used to correlate results.
        question: The user's input.
        answer: What the system produced.
        contexts: Retrieved texts that were placed in the prompt. Required by
            faithfulness and context-precision metrics.
        reference: The ground-truth answer, when you have one.
        reference_contexts: The ideal contexts, for retrieval recall metrics.
        metadata: Anything else worth slicing results by (tenant, model, ...).

    Example:
        >>> EvalSample(question="q", answer="a").id != ""
        True
    """

    id: str = Field(default_factory=lambda: __import__("uuid").uuid4().hex[:12])
    question: str = ""
    answer: str = ""
    contexts: list[str] = Field(default_factory=list)
    reference: str = ""
    reference_contexts: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_answer(cls, answer: RAGAnswer, *, reference: str = "") -> EvalSample:
        """Build a sample straight from a RAG answer.

        This is the ergonomic path: run your pipeline, feed the answers to the
        evaluator, done.

        Args:
            answer: What :meth:`~windlass.rag.pipeline.RAGPipeline.ask` returned.
            reference: Ground truth, when available.

        Returns:
            A populated sample.

        Example:
            >>> from windlass.core.types import RAGAnswer
            >>> EvalSample.from_answer(RAGAnswer(answer="a", question="q")).question
            'q'
        """
        return cls(
            question=answer.question,
            answer=answer.answer,
            contexts=[hit.chunk.content for hit in answer.contexts],
            reference=reference,
            metadata=dict(answer.metadata),
        )


class Evaluator(Component):
    """Abstract evaluation backend.

    Args:
        metrics: Metric names to compute. Which names are valid depends on the
            backend; ask it with :meth:`available_metrics`.
        threshold: Score at or above which a result counts as passing.
        llm: Judge model for metrics that need one. Many quality metrics are
            themselves LLM calls.
        concurrency: Maximum simultaneous sample evaluations.
        name: Component name for traces.
        **config: Backend-specific options.

    Attributes:
        metrics: The configured metric names.
        threshold: The configured pass threshold.

    Example:
        Implementing an evaluator takes one method::

            class LengthEvaluator(Evaluator):
                provider_name = "length"

                async def aevaluate_sample(self, sample):
                    score = min(1.0, len(sample.answer) / 100)
                    return [EvaluationResult(metric="length", score=score)]
    """

    kind = "evaluator"
    provider_name: str = "evaluator"

    def __init__(
        self,
        *,
        metrics: Sequence[str] | None = None,
        threshold: float = 0.5,
        llm: Any = None,
        concurrency: int | None = None,
        name: str | None = None,
        **config: Any,
    ) -> None:
        super().__init__(
            name=name or self.provider_name,
            metrics=list(metrics or self.default_metrics()),
            threshold=threshold,
            **config,
        )
        self.metrics: list[str] = list(metrics or self.default_metrics())
        self.threshold = threshold
        self.llm = llm
        self.concurrency = concurrency or settings().max_concurrency

    # -- provider hooks ---------------------------------------------------
    @classmethod
    def default_metrics(cls) -> tuple[str, ...]:
        """Return the metrics used when the caller does not choose any."""
        return ()

    @classmethod
    def available_metrics(cls) -> tuple[str, ...]:
        """Return every metric name this backend understands."""
        return cls.default_metrics()

    @abc.abstractmethod
    async def aevaluate_sample(self, sample: EvalSample) -> list[EvaluationResult]:
        """Score one sample against every configured metric.

        Args:
            sample: The interaction to score.

        Returns:
            One :class:`~windlass.core.types.EvaluationResult` per metric.

        Raises:
            EvaluationError: When a metric cannot be computed.
        """

    # -- public API -------------------------------------------------------
    async def aevaluate(
        self, samples: Sequence[EvalSample | RAGAnswer | dict[str, Any]]
    ) -> EvaluationReport:
        """Evaluate a dataset and aggregate the results.

        Args:
            samples: Samples, RAG answers, or dicts that coerce into samples.

        Returns:
            An aggregated :class:`~windlass.core.types.EvaluationReport`.

        Raises:
            EvaluationError: When every sample fails to evaluate.

        Performance:
            Samples run concurrently up to ``concurrency``. LLM-judged metrics
            dominate the cost, so keep an eye on that budget when evaluating
            thousands of rows.

        Example:
            >>> import asyncio
            >>> from windlass.providers.evaluation.builtin import BuiltinEvaluator
            >>> ev = BuiltinEvaluator(metrics=["answer_relevancy_lexical"])
            >>> report = asyncio.run(ev.aevaluate([
            ...     EvalSample(question="what is rag", answer="rag is retrieval")
            ... ]))
            >>> report.samples
            1
        """
        prepared = [self._coerce(s) for s in samples]
        if not prepared:
            return EvaluationReport(samples=0)

        outcomes = await gather_bounded(
            [self.aevaluate_sample(s) for s in prepared],
            limit=self.concurrency,
            return_exceptions=True,
        )

        results: list[EvaluationResult] = []
        failures = 0
        for sample, outcome in zip(prepared, outcomes, strict=True):
            if isinstance(outcome, BaseException):
                failures += 1
                self._log.warning("Evaluation failed for sample %s: %s", sample.id, outcome)
                continue
            for result in outcome:
                if not result.sample_id:
                    result.sample_id = sample.id
                if result.threshold is None:
                    result.threshold = self.threshold
                result.passed = result.score >= result.threshold
                results.append(result)

        if failures and not results:
            from windlass.core.exceptions import EvaluationError

            raise EvaluationError(
                f"All {failures} samples failed to evaluate.",
                hint="Check that the judge LLM is configured and that samples "
                "carry the fields your metrics need (contexts, reference).",
            )

        return EvaluationReport(results=results, samples=len(prepared))

    def evaluate(
        self, samples: Sequence[EvalSample | RAGAnswer | dict[str, Any]]
    ) -> EvaluationReport:
        """Blocking :meth:`aevaluate`."""
        return run_sync(self.aevaluate(samples))

    # -- helpers ----------------------------------------------------------
    @staticmethod
    def _coerce(sample: EvalSample | RAGAnswer | dict[str, Any]) -> EvalSample:
        """Turn any accepted sample shape into an :class:`EvalSample`."""
        if isinstance(sample, EvalSample):
            return sample
        if isinstance(sample, RAGAnswer):
            return EvalSample.from_answer(sample)
        return EvalSample.model_validate(sample)

    def _require_llm(self) -> Any:
        """Return the judge model, or explain how to configure one."""
        if self.llm is None:
            from windlass.core.exceptions import EvaluationError

            raise EvaluationError(
                f"The {self.name!r} evaluator needs a judge model for these metrics.",
                hint="Pass one: Windlass.evaluator('builtin', llm=Windlass.llm('openai')).",
                context={"metrics": self.metrics},
            )
        return self.llm

    def __repr__(self) -> str:
        return f"{type(self).__name__}(metrics={self.metrics})"
