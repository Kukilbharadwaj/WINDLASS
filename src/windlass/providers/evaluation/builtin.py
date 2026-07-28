"""Built-in evaluation metrics.

Two families, deliberately separated:

**Lexical metrics** need no model and no dependencies. They are fast, free and
deterministic, which makes them the right choice for CI regression gates:
``exact_match``, ``f1``, ``rouge_l``, ``answer_relevancy_lexical``,
``context_precision_lexical``, ``context_recall_lexical``.

**LLM-judged metrics** need a judge model and cost tokens, but measure things
lexical overlap cannot: ``faithfulness`` (is the answer supported by the
retrieved context?), ``answer_relevancy``, ``answer_correctness``,
``context_relevancy``.

Faithfulness is the one to watch. It is the direct measure of hallucination in a
RAG system, and it is the metric that catches a retrieval regression before your
users do.

Example:
    >>> from windlass.interfaces.evaluator import EvalSample
    >>> ev = BuiltinEvaluator(metrics=["exact_match", "f1"])
    >>> report = ev.evaluate([EvalSample(question="q", answer="the cat", reference="the cat")])
    >>> report.summary["exact_match"], report.summary["f1"]
    (1.0, 1.0)
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

from windlass.core.concurrency import gather_bounded
from windlass.core.registry import register
from windlass.core.text import tokenize_words
from windlass.core.types import EvaluationResult
from windlass.interfaces.evaluator import EvalSample, Evaluator

__all__ = ["JUDGED_METRICS", "LEXICAL_METRICS", "BuiltinEvaluator"]

#: Metrics computable without a model.
LEXICAL_METRICS = (
    "exact_match",
    "f1",
    "rouge_l",
    "answer_relevancy_lexical",
    "context_precision_lexical",
    "context_recall_lexical",
    "answer_length",
)

#: Metrics requiring a judge model.
JUDGED_METRICS = (
    "faithfulness",
    "answer_relevancy",
    "answer_correctness",
    "context_relevancy",
)

_FAITHFULNESS_PROMPT = """\
You are grading whether an answer is fully supported by the context it was given.

<context>
{context}
</context>

<answer>
{answer}
</answer>

List each factual claim the answer makes, then decide whether the context
supports it. Finish with a line of exactly this form:

SCORE: <supported claims> / <total claims>
"""

_RELEVANCY_PROMPT = """\
Rate how well the answer addresses the question, from 0 to 10.

Question: {question}
Answer: {answer}

10 = answers the question completely and directly.
5  = partially answers, or answers something adjacent.
0  = does not address the question at all.

Reply with a line of exactly this form:

SCORE: <number>
"""

_CORRECTNESS_PROMPT = """\
Compare an answer against the reference answer and rate its factual accuracy
from 0 to 10.

Question:  {question}
Reference: {reference}
Answer:    {answer}

10 = factually equivalent to the reference (wording may differ).
5  = partly correct, or correct but missing key information.
0  = contradicts the reference.

Reply with a line of exactly this form:

SCORE: <number>
"""

_CONTEXT_RELEVANCY_PROMPT = """\
Decide how much of the retrieved context is actually relevant to the question.

Question: {question}

<context>
{context}
</context>

Count the context sentences that help answer the question. Finish with a line of
exactly this form:

SCORE: <relevant sentences> / <total sentences>
"""

_SCORE_RE = re.compile(r"SCORE:\s*([0-9]*\.?[0-9]+)\s*(?:/\s*([0-9]*\.?[0-9]+))?", re.IGNORECASE)


@register.evaluator(
    "builtin",
    aliases=("default", "windlass"),
    description="Lexical and LLM-judged RAG metrics with no required dependencies.",
)
class BuiltinEvaluator(Evaluator):
    """Windlass's own evaluation metrics.

    Args:
        metrics: Metric names from :data:`LEXICAL_METRICS` and
            :data:`JUDGED_METRICS`. Defaults to the lexical set, so evaluation
            works with no model configured.
        threshold: Score at or above which a result passes.
        llm: Judge model. Required only for :data:`JUDGED_METRICS`.
        concurrency: Maximum simultaneous sample evaluations.
        **config: Forwarded to :class:`~windlass.interfaces.evaluator.Evaluator`.

    Raises:
        ValueError: For an unknown metric name.
        EvaluationError: When a judged metric is requested with no judge model.

    Performance:
        Lexical metrics are pure Python and effectively free. Each judged metric
        costs one model call per sample, so a 4-metric run over 500 samples is
        2,000 calls — batch it and use a small judge model.
    """

    provider_name = "builtin"

    def __init__(
        self,
        *,
        metrics: Sequence[str] | None = None,
        threshold: float = 0.5,
        llm: Any = None,
        concurrency: int | None = None,
        **config: Any,
    ) -> None:
        chosen = list(metrics or self.default_metrics())
        unknown = set(chosen) - set(LEXICAL_METRICS) - set(JUDGED_METRICS)
        if unknown:
            raise ValueError(
                f"Unknown metric(s): {', '.join(sorted(unknown))}. "
                f"Available: {', '.join(sorted(self.available_metrics()))}"
            )
        super().__init__(
            metrics=chosen, threshold=threshold, llm=llm, concurrency=concurrency, **config
        )
        if any(m in JUDGED_METRICS for m in chosen) and llm is None:
            self._require_llm()

    @classmethod
    def default_metrics(cls) -> tuple[str, ...]:
        """Return the default metric set: lexical only, so no model is needed."""
        return ("exact_match", "f1", "rouge_l", "answer_relevancy_lexical")

    @classmethod
    def available_metrics(cls) -> tuple[str, ...]:
        """Return every metric this evaluator supports."""
        return LEXICAL_METRICS + JUDGED_METRICS

    async def aevaluate_sample(self, sample: EvalSample) -> list[EvaluationResult]:
        """Score one sample against every configured metric.

        Args:
            sample: The interaction to score.

        Returns:
            One result per metric. A metric whose inputs are missing (a
            reference-based metric with no reference) is skipped rather than
            scored zero, so averages stay honest.
        """
        lexical = [m for m in self.metrics if m in LEXICAL_METRICS]
        judged = [m for m in self.metrics if m in JUDGED_METRICS]

        results = [r for r in (self._lexical(name, sample) for name in lexical) if r]
        if judged:
            produced = await gather_bounded(
                [self._judged(name, sample) for name in judged],
                limit=len(judged),
                return_exceptions=True,
            )
            for name, outcome in zip(judged, produced, strict=True):
                if isinstance(outcome, BaseException):
                    self._log.warning("Metric %s failed for %s: %s", name, sample.id, outcome)
                    continue
                if outcome:
                    results.append(outcome)
        return results

    # -- lexical ----------------------------------------------------------
    def _lexical(self, name: str, sample: EvalSample) -> EvaluationResult | None:
        """Compute one dependency-free metric."""
        answer, reference = sample.answer, sample.reference
        context = " ".join(sample.contexts)

        match name:
            case "exact_match":
                if not reference:
                    return None
                score = float(_normalise(answer) == _normalise(reference))
                reason = "exact string match" if score else "answers differ"
            case "f1":
                if not reference:
                    return None
                score = _token_f1(answer, reference)
                reason = f"token overlap F1 {score:.2f}"
            case "rouge_l":
                if not reference:
                    return None
                score = _rouge_l(answer, reference)
                reason = f"longest common subsequence F1 {score:.2f}"
            case "answer_relevancy_lexical":
                score = _token_recall(sample.question, answer)
                reason = f"{score:.0%} of question terms appear in the answer"
            case "context_precision_lexical":
                if not sample.contexts:
                    return None
                score = _context_precision(sample.question, sample.contexts)
                reason = f"{score:.0%} of retrieved chunks share terms with the question"
            case "context_recall_lexical":
                if not context:
                    return None
                score = _token_recall(answer, context)
                reason = f"{score:.0%} of answer terms appear in the context"
            case "answer_length":
                words = len(answer.split())
                score = min(1.0, words / 50) if words else 0.0
                reason = f"{words} words"
            case _:  # pragma: no cover - guarded in __init__
                return None

        return EvaluationResult(
            metric=name, score=score, reason=reason, sample_id=sample.id, threshold=self.threshold
        )

    # -- judged -----------------------------------------------------------
    async def _judged(self, name: str, sample: EvalSample) -> EvaluationResult | None:
        """Compute one LLM-judged metric."""
        llm = self._require_llm()
        context = "\n\n".join(sample.contexts)

        match name:
            case "faithfulness":
                if not context or not sample.answer:
                    return None
                prompt = _FAITHFULNESS_PROMPT.format(context=context, answer=sample.answer)
            case "answer_relevancy":
                prompt = _RELEVANCY_PROMPT.format(question=sample.question, answer=sample.answer)
            case "answer_correctness":
                if not sample.reference:
                    return None
                prompt = _CORRECTNESS_PROMPT.format(
                    question=sample.question, reference=sample.reference, answer=sample.answer
                )
            case "context_relevancy":
                if not context:
                    return None
                prompt = _CONTEXT_RELEVANCY_PROMPT.format(question=sample.question, context=context)
            case _:  # pragma: no cover
                return None

        completion = await llm.acomplete(prompt, temperature=0.0, max_tokens=600)
        score = _parse_score(completion.content)
        return EvaluationResult(
            metric=name,
            score=score,
            reason=completion.content.strip()[:500],
            sample_id=sample.id,
            threshold=self.threshold,
        )


# --------------------------------------------------------------------------
# scoring helpers
# --------------------------------------------------------------------------
def _normalise(text: str) -> str:
    """Lowercase, strip punctuation and collapse whitespace for comparison."""
    return " ".join(re.sub(r"[^\w\s]", " ", text.lower()).split())


def _token_f1(prediction: str, reference: str) -> float:
    """Return the token-overlap F1 of two strings, in ``[0, 1]``.

    Example:
        >>> round(_token_f1("the cat sat", "the cat"), 3)
        0.8
    """
    predicted = tokenize_words(prediction)
    expected = tokenize_words(reference)
    if not predicted or not expected:
        return 0.0
    common = set(predicted) & set(expected)
    if not common:
        return 0.0
    precision = len(common) / len(set(predicted))
    recall = len(common) / len(set(expected))
    return 2 * precision * recall / (precision + recall)


def _token_recall(needle: str, haystack: str) -> float:
    """Return the fraction of ``needle``'s meaningful tokens found in ``haystack``.

    Example:
        >>> _token_recall("cat dog", "the cat sleeps")
        0.5
    """
    from windlass.providers.retrievers.bm25 import STOPWORDS

    wanted = {t for t in tokenize_words(needle) if t not in STOPWORDS}
    if not wanted:
        return 1.0
    present = set(tokenize_words(haystack))
    return len(wanted & present) / len(wanted)


def _rouge_l(prediction: str, reference: str) -> float:
    """Return the ROUGE-L F1 (longest common subsequence) of two strings.

    Unlike token overlap, ROUGE-L rewards preserved *order*, which makes it a
    better proxy for whether an answer reads like the reference.

    Example:
        >>> _rouge_l("a b c", "a b c")
        1.0
    """
    predicted = tokenize_words(prediction)
    expected = tokenize_words(reference)
    if not predicted or not expected:
        return 0.0

    rows, cols = len(predicted), len(expected)
    table = [[0] * (cols + 1) for _ in range(rows + 1)]
    for i in range(1, rows + 1):
        for j in range(1, cols + 1):
            if predicted[i - 1] == expected[j - 1]:
                table[i][j] = table[i - 1][j - 1] + 1
            else:
                table[i][j] = max(table[i - 1][j], table[i][j - 1])

    lcs = table[rows][cols]
    if not lcs:
        return 0.0
    precision = lcs / rows
    recall = lcs / cols
    return 2 * precision * recall / (precision + recall)


def _context_precision(question: str, contexts: list[str]) -> float:
    """Return the fraction of retrieved chunks sharing terms with the question.

    A cheap proxy for "how much of what we retrieved was worth retrieving" —
    low precision means the top-k is being filled with noise.

    Example:
        >>> _context_precision("cats", ["cats purr", "engines roar"])
        0.5
    """
    from windlass.providers.retrievers.bm25 import STOPWORDS

    terms = {t for t in tokenize_words(question) if t not in STOPWORDS}
    if not terms or not contexts:
        return 0.0
    relevant = sum(1 for c in contexts if terms & set(tokenize_words(c)))
    return relevant / len(contexts)


def _parse_score(text: str) -> float:
    """Extract a normalised score from a judge model's reply.

    Handles both ``SCORE: 8`` (a 0-10 rating) and ``SCORE: 3 / 4`` (a ratio),
    and falls back to any leading number when the model ignores the format.

    Args:
        text: The judge's reply.

    Returns:
        A score in ``[0, 1]``.

    Example:
        >>> _parse_score("SCORE: 8")
        0.8
        >>> _parse_score("SCORE: 3 / 4")
        0.75
        >>> _parse_score("no score here")
        0.0
    """
    match = _SCORE_RE.search(text)
    if match:
        numerator = float(match.group(1))
        denominator = match.group(2)
        if denominator:
            total = float(denominator)
            return max(0.0, min(1.0, numerator / total)) if total else 0.0
        return max(0.0, min(1.0, numerator / 10.0))

    fallback = re.search(r"\b([0-9]|10)(?:\.\d+)?\b", text)
    if fallback:
        return max(0.0, min(1.0, float(fallback.group(0)) / 10.0))
    # A judge that produced no parseable verdict gets no credit — silently
    # scoring 0.5 would hide a broken prompt behind a plausible-looking average.
    return 0.0
