"""RAGAS and DeepEval adapters.

Both are established evaluation frameworks with metric implementations that have
been validated against human judgement. Windlass wraps them so you can run their
metrics on Windlass samples without learning their data models.

Install with::

    pip install "windlass[evaluation]"

Example:
    >>> from windlass import Windlass                                       # doctest: +SKIP
    >>> ev = Windlass.evaluator("ragas", metrics=["faithfulness"],         # doctest: +SKIP
    ...                        llm=Windlass.llm("openai"))                 # doctest: +SKIP
    >>> ev.evaluate(answers)                                              # doctest: +SKIP
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, ClassVar

from windlass.core.concurrency import to_thread
from windlass.core.exceptions import EvaluationError
from windlass.core.lazy import require
from windlass.core.registry import register
from windlass.core.types import EvaluationReport, EvaluationResult
from windlass.interfaces.evaluator import EvalSample, Evaluator

__all__ = ["DeepEvalEvaluator", "RagasEvaluator"]


@register.evaluator(
    "ragas",
    description="RAGAS reference-free RAG metrics.",
)
class RagasEvaluator(Evaluator):
    """Evaluation via the RAGAS framework.

    RAGAS specialises in *reference-free* RAG evaluation — it scores faithfulness
    and context quality without needing ground-truth answers, which is what makes
    it usable on production traffic rather than only on a curated test set.

    Args:
        metrics: RAGAS metric names — ``faithfulness``, ``answer_relevancy``,
            ``context_precision``, ``context_recall``, ``answer_correctness``,
            ``answer_similarity``.
        threshold: Score at or above which a result passes.
        llm: Judge model. RAGAS needs a LangChain-compatible model; a Windlass
            LLM whose ``native()`` is LangChain-shaped is unwrapped
            automatically.
        embeddings: Embedding model for similarity-based metrics.
        **config: Forwarded to :class:`~windlass.interfaces.evaluator.Evaluator`.

    Raises:
        MissingDependencyError: When ``ragas`` is not installed.
        EvaluationError: When a metric name is unknown or the run fails.

    Note:
        RAGAS evaluates a whole dataset at once, so this adapter overrides
        :meth:`aevaluate` rather than scoring sample by sample.
    """

    provider_name = "ragas"

    def __init__(
        self,
        *,
        metrics: Sequence[str] | None = None,
        threshold: float = 0.5,
        llm: Any = None,
        embeddings: Any = None,
        **config: Any,
    ) -> None:
        super().__init__(metrics=metrics, threshold=threshold, llm=llm, **config)
        self._ragas = require("ragas", extra="evaluation", feature="The RAGAS evaluator")
        self.embeddings = embeddings

    @classmethod
    def default_metrics(cls) -> tuple[str, ...]:
        """Return RAGAS's core reference-free metrics."""
        return ("faithfulness", "answer_relevancy", "context_precision")

    @classmethod
    def available_metrics(cls) -> tuple[str, ...]:
        """Return every RAGAS metric this adapter can resolve."""
        return (
            "faithfulness",
            "answer_relevancy",
            "context_precision",
            "context_recall",
            "answer_correctness",
            "answer_similarity",
        )

    def native(self) -> Any:
        """Return the imported ``ragas`` module (Level 3 access)."""
        return self._ragas

    async def aevaluate_sample(self, sample: EvalSample) -> list[EvaluationResult]:
        """Evaluate a single sample by running the dataset path with one row."""
        report = await self.aevaluate([sample])
        return report.results

    async def aevaluate(self, samples: Sequence[EvalSample | Any]) -> EvaluationReport:
        """Run RAGAS over the whole dataset.

        Args:
            samples: Samples, RAG answers, or dicts.

        Returns:
            The aggregated report.

        Raises:
            EvaluationError: When RAGAS fails or produces no scores.
        """
        prepared = [self._coerce(s) for s in samples]
        if not prepared:
            return EvaluationReport(samples=0)

        metrics = self._resolve_metrics()
        dataset = {
            "question": [s.question for s in prepared],
            "answer": [s.answer for s in prepared],
            "contexts": [s.contexts or [""] for s in prepared],
            "ground_truth": [s.reference for s in prepared],
        }

        def _run() -> Any:
            datasets = require("datasets", extra="evaluation", feature="The RAGAS evaluator")
            hf_dataset = datasets.Dataset.from_dict(dataset)
            kwargs: dict[str, Any] = {"dataset": hf_dataset, "metrics": metrics}
            judge = _langchain_model(self.llm)
            if judge is not None:
                kwargs["llm"] = judge
            if self.embeddings is not None:
                kwargs["embeddings"] = _langchain_model(self.embeddings) or self.embeddings
            return self._ragas.evaluate(**kwargs)

        try:
            outcome = await to_thread(_run)
        except Exception as exc:
            raise EvaluationError(
                f"RAGAS evaluation failed: {exc}",
                hint="RAGAS needs a LangChain-compatible judge model; pass "
                "llm=... explicitly if the default could not be unwrapped.",
            ) from exc

        return self._to_report(outcome, prepared)

    def _resolve_metrics(self) -> list[Any]:
        """Turn metric names into RAGAS metric objects."""
        from importlib import import_module

        module = import_module("ragas.metrics")
        resolved: list[Any] = []
        for name in self.metrics:
            metric = getattr(module, name, None)
            if metric is None:
                raise EvaluationError(
                    f"RAGAS has no metric named {name!r}.",
                    hint=f"Available: {', '.join(self.available_metrics())}",
                )
            resolved.append(metric)
        return resolved

    def _to_report(self, outcome: Any, samples: list[EvalSample]) -> EvaluationReport:
        """Translate a RAGAS result object into a Windlass report."""
        results: list[EvaluationResult] = []
        try:
            frame = outcome.to_pandas()
            for position, row in enumerate(frame.to_dict("records")):
                sample_id = samples[position].id if position < len(samples) else ""
                for name in self.metrics:
                    value = row.get(name)
                    if value is None:
                        continue
                    score = float(value)
                    results.append(
                        EvaluationResult(
                            metric=name,
                            score=score,
                            passed=score >= self.threshold,
                            threshold=self.threshold,
                            sample_id=sample_id,
                        )
                    )
        except Exception:
            for name in self.metrics:
                value = None
                try:
                    value = outcome[name]
                except Exception:
                    value = getattr(outcome, name, None)
                if value is None:
                    continue
                score = float(value)
                results.append(
                    EvaluationResult(
                        metric=name,
                        score=score,
                        passed=score >= self.threshold,
                        threshold=self.threshold,
                    )
                )

        if not results:
            raise EvaluationError(
                "RAGAS returned no scores.",
                hint="Check that your samples carry the fields the chosen metrics "
                "need — context_recall and answer_correctness both require a reference.",
            )
        return EvaluationReport(results=results, samples=len(samples))


@register.evaluator(
    "deepeval",
    description="DeepEval metrics, including its pytest-style assertions.",
)
class DeepEvalEvaluator(Evaluator):
    """Evaluation via the DeepEval framework.

    DeepEval's angle is unit-test-style evaluation: metrics carry thresholds and
    produce pass/fail verdicts with explanations, which fits naturally into CI.

    Args:
        metrics: DeepEval metric names — ``answer_relevancy``, ``faithfulness``,
            ``contextual_precision``, ``contextual_recall``,
            ``contextual_relevancy``, ``hallucination``, ``bias``, ``toxicity``.
        threshold: Threshold passed to each metric.
        llm: Judge model. DeepEval accepts a model name string or its own
            ``DeepEvalBaseLLM``; a Windlass LLM contributes its model name.
        **config: Forwarded to :class:`~windlass.interfaces.evaluator.Evaluator`.

    Raises:
        MissingDependencyError: When ``deepeval`` is not installed.
        EvaluationError: When a metric name is unknown.
    """

    provider_name = "deepeval"

    _METRIC_CLASSES: ClassVar[dict[str, str]] = {
        "answer_relevancy": "AnswerRelevancyMetric",
        "faithfulness": "FaithfulnessMetric",
        "contextual_precision": "ContextualPrecisionMetric",
        "contextual_recall": "ContextualRecallMetric",
        "contextual_relevancy": "ContextualRelevancyMetric",
        "hallucination": "HallucinationMetric",
        "bias": "BiasMetric",
        "toxicity": "ToxicityMetric",
    }

    def __init__(
        self,
        *,
        metrics: Sequence[str] | None = None,
        threshold: float = 0.5,
        llm: Any = None,
        **config: Any,
    ) -> None:
        super().__init__(metrics=metrics, threshold=threshold, llm=llm, **config)
        self._deepeval = require("deepeval", extra="evaluation", feature="The DeepEval evaluator")
        unknown = set(self.metrics) - set(self._METRIC_CLASSES)
        if unknown:
            raise EvaluationError(
                f"DeepEval has no metric named {', '.join(sorted(unknown))}.",
                hint=f"Available: {', '.join(self.available_metrics())}",
            )

    @classmethod
    def default_metrics(cls) -> tuple[str, ...]:
        """Return DeepEval's core RAG metrics."""
        return ("answer_relevancy", "faithfulness")

    @classmethod
    def available_metrics(cls) -> tuple[str, ...]:
        """Return every DeepEval metric this adapter can construct."""
        return tuple(cls._METRIC_CLASSES)

    def native(self) -> Any:
        """Return the imported ``deepeval`` module (Level 3 access)."""
        return self._deepeval

    async def aevaluate_sample(self, sample: EvalSample) -> list[EvaluationResult]:
        """Score one sample with every configured DeepEval metric.

        Args:
            sample: The interaction to score.

        Returns:
            One result per metric, carrying DeepEval's own explanation.

        Raises:
            EvaluationError: When DeepEval fails to construct or run a metric.
        """
        from importlib import import_module

        def _run() -> list[EvaluationResult]:
            test_case_module = import_module("deepeval.test_case")
            metrics_module = import_module("deepeval.metrics")

            case = test_case_module.LLMTestCase(
                input=sample.question,
                actual_output=sample.answer,
                expected_output=sample.reference or None,
                retrieval_context=sample.contexts or None,
                context=sample.contexts or None,
            )

            produced: list[EvaluationResult] = []
            for name in self.metrics:
                metric_class = getattr(metrics_module, self._METRIC_CLASSES[name])
                kwargs: dict[str, Any] = {"threshold": self.threshold}
                model_name = _model_name(self.llm)
                if model_name:
                    kwargs["model"] = model_name
                metric = metric_class(**kwargs)
                metric.measure(case)
                produced.append(
                    EvaluationResult(
                        metric=name,
                        score=float(metric.score or 0.0),
                        passed=bool(metric.is_successful()),
                        threshold=self.threshold,
                        reason=str(getattr(metric, "reason", ""))[:500],
                        sample_id=sample.id,
                    )
                )
            return produced

        try:
            return await to_thread(_run)
        except Exception as exc:
            raise EvaluationError(f"DeepEval evaluation failed: {exc}") from exc


def _langchain_model(candidate: Any) -> Any:
    """Return a LangChain-compatible model from a Windlass LLM, if possible.

    Args:
        candidate: A Windlass LLM, a LangChain model, or ``None``.

    Returns:
        Something RAGAS can use, or ``None`` to let it use its own default.
    """
    if candidate is None:
        return None
    native = getattr(candidate, "native", None)
    resolved = native() if callable(native) else candidate
    return resolved if hasattr(resolved, "invoke") or hasattr(resolved, "generate") else None


def _model_name(candidate: Any) -> str | None:
    """Return the model identifier from a Windlass LLM, if it has one."""
    name = getattr(candidate, "model", None)
    return str(name) if name else None
