"""ChunkCoverageEvaluator — measures whether chunks cover key query aspects."""

from __future__ import annotations

import logging

from vectorforge.evaluation.base import BaseEvaluator
from vectorforge.evaluation.evaluators._judge import judge
from vectorforge.evaluation.types import EvaluationResult, EvaluationSample
from vectorforge.llm.base import BaseLLMProvider

logger = logging.getLogger(__name__)

_ASPECT_PROMPT = """Given this query, identify the key information aspects needed
to provide a complete answer. Return as a JSON list of strings.

Query: {query}

Respond with JSON: {{"aspects": ["aspect1", "aspect2", ...]}}"""

_COVERAGE_PROMPT = """Does any of the following retrieved text chunks contain
information about this aspect?

Aspect: {aspect}

Chunks:
{chunks_text}

Respond with JSON: {{"covered": true, "chunk_index": <int|null>, "reasoning": "<explanation>"}}"""


class ChunkCoverageEvaluator(BaseEvaluator):
    """Evaluates whether retrieved chunks cover the key aspects of a query.

    Uses an LLM judge to extract aspects from the query, then checks
    each aspect against the retrieved chunks.

    Args:
        llm: The LLM provider for judge calls.
        model: Model override for the judge.
    """

    def __init__(self, llm: BaseLLMProvider, model: str = "") -> None:
        self._llm = llm
        self._model = model

    @property
    def name(self) -> str:
        return "chunk_coverage"

    @property
    def category(self) -> str:
        return "retrieval"

    @property
    def description(self) -> str:
        return "Measures whether retrieved chunks cover the key aspects needed to answer the query"

    async def evaluate(self, sample: EvaluationSample) -> EvaluationResult:
        """Evaluate aspect coverage of the retrieved chunks.

        Args:
            sample: The query-answer pair with retrieved chunks.

        Returns:
            EvaluationResult with coverage ratio and aspect details.
        """
        if not sample.chunks:
            return EvaluationResult(
                query_log_id=sample.query_log_id,
                evaluator_name=self.name,
                score=0.0,
                details={"reason": "no_chunks"},
                reasoning="No chunks retrieved.",
            )

        # Step 1: Extract aspects
        aspects = await self._extract_aspects(sample.query)
        if not aspects:
            return EvaluationResult(
                query_log_id=sample.query_log_id,
                evaluator_name=self.name,
                score=1.0,
                details={"reason": "no_aspects_identified"},
                reasoning="No key aspects identified for query.",
            )

        # Step 2: Check coverage
        chunks_text = "\n".join(
            f"[{i}] {c.text}" for i, c in enumerate(sample.chunks)
        )
        covered: list[str] = []
        uncovered: list[str] = []

        for aspect in aspects:
            is_covered = await self._check_aspect(aspect, chunks_text)
            if is_covered:
                covered.append(aspect)
            else:
                uncovered.append(aspect)

        coverage_score = len(covered) / len(aspects)

        return EvaluationResult(
            query_log_id=sample.query_log_id,
            evaluator_name=self.name,
            score=round(coverage_score, 4),
            details={
                "aspects": aspects,
                "covered_aspects": covered,
                "uncovered_aspects": uncovered,
                "coverage_ratio": round(coverage_score, 4),
            },
            reasoning=f"Coverage: {len(covered)}/{len(aspects)} aspects covered",
        )

    async def _extract_aspects(self, query: str) -> list[str]:
        """Extract key information aspects from the query.

        Args:
            query: The user query.

        Returns:
            List of aspect strings.
        """
        prompt = _ASPECT_PROMPT.format(query=query)
        try:
            result = await judge(self._llm, prompt, model=self._model)
            aspects = result.get("aspects", [])
            return [str(a) for a in aspects if a]
        except Exception:
            logger.warning("Failed to extract aspects for query")
            return []

    async def _check_aspect(self, aspect: str, chunks_text: str) -> bool:
        """Check if an aspect is covered by the chunks.

        Args:
            aspect: The aspect to check.
            chunks_text: Formatted chunk text.

        Returns:
            True if the aspect is covered.
        """
        prompt = _COVERAGE_PROMPT.format(aspect=aspect, chunks_text=chunks_text)
        try:
            result = await judge(self._llm, prompt, model=self._model)
            return bool(result.get("covered", False))
        except Exception:
            logger.warning("Failed to check coverage for aspect: %s", aspect)
            return False
