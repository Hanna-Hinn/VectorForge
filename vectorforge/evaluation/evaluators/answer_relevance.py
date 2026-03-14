"""AnswerRelevanceEvaluator — measures whether the answer addresses the query."""

from __future__ import annotations

import logging

from vectorforge.evaluation.base import BaseEvaluator
from vectorforge.evaluation.evaluators._judge import judge
from vectorforge.evaluation.types import EvaluationResult, EvaluationSample
from vectorforge.llm.base import BaseLLMProvider

logger = logging.getLogger(__name__)

_DIRECT_RELEVANCE_PROMPT = """Rate how well the following answer addresses the given query.

Query: {query}
Answer: {answer}

Rate on a scale from 0.0 to 1.0:
- 1.0: Directly and completely answers the query
- 0.7-0.9: Answers the query well with minor gaps
- 0.4-0.6: Partially addresses the query
- 0.1-0.3: Tangentially related but doesn't answer
- 0.0: Completely off-topic

Respond with JSON: {{"score": <float>, "reasoning": "<explanation>"}}"""


class AnswerRelevanceEvaluator(BaseEvaluator):
    """Evaluates whether the generated answer addresses the user's query.

    Uses a direct LLM relevance judgment to score the answer-query
    alignment.

    Args:
        llm: The LLM provider for judge calls.
        model: Model override for the judge.
    """

    def __init__(self, llm: BaseLLMProvider, model: str = "") -> None:
        self._llm = llm
        self._model = model

    @property
    def name(self) -> str:
        return "answer_relevance"

    @property
    def category(self) -> str:
        return "generation"

    @property
    def description(self) -> str:
        return "Measures whether the generated answer actually addresses the user's question"

    async def evaluate(self, sample: EvaluationSample) -> EvaluationResult:
        """Evaluate answer relevance to the query.

        Args:
            sample: The query-answer pair.

        Returns:
            EvaluationResult with relevance score.
        """
        if not sample.answer:
            return EvaluationResult(
                query_log_id=sample.query_log_id,
                evaluator_name=self.name,
                score=0.0,
                details={"reason": "empty_answer"},
                reasoning="No answer to evaluate.",
            )

        direct_score, reasoning = await self._judge_relevance(
            sample.query, sample.answer
        )

        return EvaluationResult(
            query_log_id=sample.query_log_id,
            evaluator_name=self.name,
            score=round(direct_score, 4),
            details={
                "direct_relevance_score": round(direct_score, 4),
            },
            reasoning=reasoning,
        )

    async def _judge_relevance(
        self, query: str, answer: str
    ) -> tuple[float, str]:
        """Get a direct relevance score from the judge.

        Args:
            query: The user query.
            answer: The generated answer.

        Returns:
            Tuple of (score, reasoning).
        """
        prompt = _DIRECT_RELEVANCE_PROMPT.format(query=query, answer=answer)
        try:
            result = await judge(self._llm, prompt, model=self._model)
            score = float(result.get("score", 0.0))
            score = max(0.0, min(1.0, score))
            reasoning = str(result.get("reasoning", ""))
            return score, f"Answer relevance: {score:.2f} — {reasoning}"
        except Exception:
            logger.warning("Failed to judge answer relevance")
            return 0.0, "Judge call failed"
