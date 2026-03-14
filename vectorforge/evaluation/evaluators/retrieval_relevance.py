"""RetrievalRelevanceEvaluator — measures chunk relevance using LLM-as-judge."""

from __future__ import annotations

import logging
import statistics
from typing import Any

from vectorforge.evaluation.base import BaseEvaluator
from vectorforge.evaluation.evaluators._judge import judge
from vectorforge.evaluation.types import EvaluationResult, EvaluationSample
from vectorforge.llm.base import BaseLLMProvider

logger = logging.getLogger(__name__)

_RELEVANCE_PROMPT = """Given the following query and text chunk, rate the relevance
of the chunk to answering the query.

Query: {query}
Chunk: {chunk_text}

Rate relevance on a scale from 0.0 to 1.0:
- 1.0: Directly answers the query
- 0.7-0.9: Contains highly relevant information
- 0.4-0.6: Partially relevant
- 0.1-0.3: Tangentially related
- 0.0: Completely irrelevant

Respond with JSON: {{"score": <float>, "reasoning": "<explanation>"}}"""


class RetrievalRelevanceEvaluator(BaseEvaluator):
    """Evaluates whether retrieved chunks are relevant to the query.

    Uses an LLM judge to score each chunk individually, then aggregates.

    Args:
        llm: The LLM provider for judge calls.
        model: Model override for the judge.
    """

    def __init__(self, llm: BaseLLMProvider, model: str = "") -> None:
        self._llm = llm
        self._model = model

    @property
    def name(self) -> str:
        return "retrieval_relevance"

    @property
    def category(self) -> str:
        return "retrieval"

    @property
    def description(self) -> str:
        return "Measures whether retrieved chunks are relevant to the query"

    async def evaluate(self, sample: EvaluationSample) -> EvaluationResult:
        """Evaluate relevance of each retrieved chunk.

        Args:
            sample: The query-answer pair with retrieved chunks.

        Returns:
            EvaluationResult with per-chunk relevance scores.
        """
        if not sample.chunks:
            return EvaluationResult(
                query_log_id=sample.query_log_id,
                evaluator_name=self.name,
                score=0.0,
                details={"reason": "no_chunks"},
                reasoning="No chunks retrieved.",
            )

        per_chunk: list[dict[str, Any]] = []
        scores: list[float] = []

        for chunk in sample.chunks:
            prompt = _RELEVANCE_PROMPT.format(
                query=sample.query, chunk_text=chunk.text
            )
            try:
                result = await judge(self._llm, prompt, model=self._model)
                score = float(result.get("score", 0.0))
                score = max(0.0, min(1.0, score))
                reasoning = str(result.get("reasoning", ""))
            except Exception:
                logger.warning(
                    "Failed to judge chunk %s, defaulting to 0.0", chunk.chunk_id
                )
                score = 0.0
                reasoning = "Judge call failed"

            scores.append(score)
            per_chunk.append({
                "chunk_id": str(chunk.chunk_id),
                "score": score,
                "reasoning": reasoning,
            })

        overall = statistics.mean(scores) if scores else 0.0
        relevant_count = sum(1 for s in scores if s >= 0.5)
        precision_at_k = relevant_count / len(scores) if scores else 0.0

        return EvaluationResult(
            query_log_id=sample.query_log_id,
            evaluator_name=self.name,
            score=round(overall, 4),
            details={
                "per_chunk_scores": per_chunk,
                "precision_at_k": round(precision_at_k, 4),
                "relevant_count": relevant_count,
                "total_chunks": len(scores),
            },
            reasoning=(
                f"Average relevance: {overall:.2f}, "
                f"Precision@{len(scores)}: {precision_at_k:.2f}"
            ),
        )
