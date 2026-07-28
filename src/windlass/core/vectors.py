"""Vector math with an optional NumPy fast path.

RAG needs a handful of linear-algebra primitives. Requiring NumPy in the core
install just for a dot product would be wrong, so every function here has a pure
Python implementation and transparently uses NumPy when it is available — which
it will be as soon as anyone installs ``windlass[rag]``.

The NumPy path is roughly two orders of magnitude faster on realistic corpora,
so the fallback is a correctness guarantee, not a performance strategy.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

from windlass.core.lazy import is_available

__all__ = [
    "HAS_NUMPY",
    "cosine_similarity",
    "dot",
    "euclidean_distance",
    "mmr",
    "normalize",
    "reciprocal_rank_fusion",
    "top_k",
]

#: Whether NumPy is importable. Checked once at import time via ``find_spec``.
HAS_NUMPY = is_available("numpy")

_np: Any = None


def _numpy() -> Any:
    """Return the NumPy module, importing it on first use."""
    global _np
    if _np is None:
        import numpy

        _np = numpy
    return _np


def dot(a: Sequence[float], b: Sequence[float]) -> float:
    """Return the dot product of two equal-length vectors.

    Args:
        a: First vector.
        b: Second vector.

    Returns:
        The dot product.

    Raises:
        ValueError: If the vectors have different lengths.

    Example:
        >>> dot([1, 2, 3], [4, 5, 6])
        32.0
    """
    if len(a) != len(b):
        raise ValueError(f"Vector length mismatch: {len(a)} != {len(b)}")
    if HAS_NUMPY:
        return float(_numpy().dot(a, b))
    return float(sum(x * y for x, y in zip(a, b, strict=True)))


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """Return the cosine similarity of two vectors, in ``[-1, 1]``.

    Zero-length vectors yield ``0.0`` rather than dividing by zero — an
    all-zeros embedding is a degenerate input, not a crash.

    Args:
        a: First vector.
        b: Second vector.

    Returns:
        Cosine similarity. ``1.0`` means identical direction.

    Raises:
        ValueError: If the vectors have different lengths.

    Example:
        >>> round(cosine_similarity([1, 0], [1, 0]), 6)
        1.0
        >>> round(cosine_similarity([1, 0], [0, 1]), 6)
        0.0
    """
    if len(a) != len(b):
        raise ValueError(f"Vector length mismatch: {len(a)} != {len(b)}")
    if HAS_NUMPY:
        np = _numpy()
        va, vb = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
        denom = float(np.linalg.norm(va) * np.linalg.norm(vb))
        return float(np.dot(va, vb) / denom) if denom else 0.0
    num = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b, strict=True):
        num += x * y
        na += x * x
        nb += y * y
    denom = math.sqrt(na) * math.sqrt(nb)
    return num / denom if denom else 0.0


def euclidean_distance(a: Sequence[float], b: Sequence[float]) -> float:
    """Return the L2 distance between two vectors.

    Args:
        a: First vector.
        b: Second vector.

    Returns:
        The Euclidean distance.

    Raises:
        ValueError: If the vectors have different lengths.

    Example:
        >>> euclidean_distance([0, 0], [3, 4])
        5.0
    """
    if len(a) != len(b):
        raise ValueError(f"Vector length mismatch: {len(a)} != {len(b)}")
    if HAS_NUMPY:
        np = _numpy()
        return float(np.linalg.norm(np.asarray(a) - np.asarray(b)))
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b, strict=True)))


def normalize(vector: Sequence[float]) -> list[float]:
    """Return ``vector`` scaled to unit length.

    Normalised vectors turn cosine similarity into a plain dot product, which is
    why FAISS indexes store them this way.

    Args:
        vector: The vector to scale.

    Returns:
        A new list with L2 norm 1, or a copy of the input when it is all zeros.

    Example:
        >>> normalize([3.0, 4.0])
        [0.6, 0.8]
    """
    if HAS_NUMPY:
        np = _numpy()
        arr = np.asarray(vector, dtype=np.float64)
        norm = float(np.linalg.norm(arr))
        return (arr / norm).tolist() if norm else list(map(float, vector))
    norm = math.sqrt(sum(x * x for x in vector))
    return [x / norm for x in vector] if norm else list(map(float, vector))


def top_k(
    query: Sequence[float],
    vectors: Sequence[Sequence[float]],
    k: int = 5,
    *,
    metric: str = "cosine",
) -> list[tuple[int, float]]:
    """Return the ``k`` best matches for ``query`` among ``vectors``.

    Args:
        query: The query vector.
        vectors: Candidate vectors, indexed positionally.
        k: How many results to return.
        metric: ``"cosine"``, ``"dot"`` or ``"euclidean"``. Euclidean scores are
            negated so that, for every metric, higher is better.

    Returns:
        ``(index, score)`` pairs sorted by descending score.

    Raises:
        ValueError: For an unknown metric.

    Example:
        >>> top_k([1, 0], [[1, 0], [0, 1], [0.7, 0.7]], k=2)[0][0]
        0
    """
    if not vectors or k <= 0:
        return []

    if metric == "cosine":
        score = cosine_similarity
    elif metric == "dot":
        score = dot
    elif metric in {"euclidean", "l2"}:

        def score(a: Sequence[float], b: Sequence[float]) -> float:
            return -euclidean_distance(a, b)

    else:
        raise ValueError(f"Unknown metric {metric!r}; use cosine, dot or euclidean.")

    if HAS_NUMPY and metric in {"cosine", "dot"}:
        np = _numpy()
        mat = np.asarray(vectors, dtype=np.float64)
        vec = np.asarray(query, dtype=np.float64)
        scores = mat @ vec
        if metric == "cosine":
            norms = np.linalg.norm(mat, axis=1) * np.linalg.norm(vec)
            scores = np.divide(scores, norms, out=np.zeros_like(scores), where=norms != 0)
        count = min(k, len(scores))
        idx = np.argpartition(-scores, count - 1)[:count]
        idx = idx[np.argsort(-scores[idx])]
        return [(int(i), float(scores[i])) for i in idx]

    scored = [(i, score(query, v)) for i, v in enumerate(vectors)]
    scored.sort(key=lambda pair: pair[1], reverse=True)
    return scored[:k]


def mmr(
    query: Sequence[float],
    vectors: Sequence[Sequence[float]],
    k: int = 5,
    *,
    diversity: float = 0.3,
) -> list[int]:
    """Select ``k`` indices with Maximal Marginal Relevance.

    Plain top-k retrieval often returns five near-identical chunks. MMR trades a
    little relevance for coverage: each pick maximises
    ``(1 - diversity) * sim(query, doc) - diversity * max sim(doc, picked)``.

    Args:
        query: The query vector.
        vectors: Candidate vectors.
        k: How many to select.
        diversity: ``0.0`` is pure relevance (identical to top-k), ``1.0`` is
            pure novelty. ``0.3`` is a good default for RAG.

    Returns:
        Selected indices, in selection order.

    Example:
        >>> mmr([1, 0], [[1, 0], [1, 0], [0, 1]], k=2, diversity=0.9)
        [0, 2]
    """
    if not vectors or k <= 0:
        return []
    k = min(k, len(vectors))
    relevance = [cosine_similarity(query, v) for v in vectors]
    selected: list[int] = [max(range(len(vectors)), key=relevance.__getitem__)]

    while len(selected) < k:
        best_idx, best_score = -1, -math.inf
        for i, vec in enumerate(vectors):
            if i in selected:
                continue
            redundancy = max(cosine_similarity(vec, vectors[j]) for j in selected)
            score = (1 - diversity) * relevance[i] - diversity * redundancy
            if score > best_score:
                best_idx, best_score = i, score
        if best_idx < 0:  # pragma: no cover - only when all candidates are taken
            break
        selected.append(best_idx)
    return selected


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[str]],
    *,
    k: int = 60,
    weights: Sequence[float] | None = None,
) -> dict[str, float]:
    """Fuse several ranked id lists into one score per id.

    RRF is what makes hybrid search work without score normalisation: BM25
    scores and cosine similarities live on incompatible scales, but their
    *ranks* are directly comparable. Each list contributes ``weight / (k + rank)``.

    Args:
        rankings: One sequence of ids per retriever, best first.
        k: Smoothing constant. The literature's default of 60 damps the
            influence of the very top positions.
        weights: Per-ranking weight. Defaults to equal weighting.

    Returns:
        A mapping of id to fused score, highest first.

    Raises:
        ValueError: If ``weights`` has a different length than ``rankings``.

    Example:
        >>> fused = reciprocal_rank_fusion([["a", "b"], ["b", "a"]])
        >>> round(fused["a"], 6) == round(fused["b"], 6)
        True
    """
    if weights is None:
        weights = [1.0] * len(rankings)
    if len(weights) != len(rankings):
        raise ValueError(f"Expected {len(rankings)} weights, got {len(weights)}.")

    scores: dict[str, float] = {}
    for ranking, weight in zip(rankings, weights, strict=True):
        for rank, item_id in enumerate(ranking, start=1):
            scores[item_id] = scores.get(item_id, 0.0) + weight / (k + rank)
    return dict(sorted(scores.items(), key=lambda kv: kv[1], reverse=True))
