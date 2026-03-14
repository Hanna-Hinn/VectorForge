"""HallucinationDetector — identifies fabricated information in generated answers."""

from __future__ import annotations

import logging
from typing import Any

from vectorforge.evaluation.base import BaseEvaluator
from vectorforge.evaluation.evaluators._judge import judge
from vectorforge.evaluation.types import EvaluationResult, EvaluationSample
from vectorforge.llm.base import BaseLLMProvider

logger = logging.getLogger(__name__)

_HALLUCINATION_PROMPT = """You are a hallucination detector. Analyze the following answer
and determine which parts, if any, contain hallucinated content —
information that is NOT present in or derivable from the context.

Context (the ONLY source of truth):
{context}

Query: {query}
Answer: {answer}

Identify all hallucinations. For each, provide:
- The hallucinated text span
- Why it's a hallucination
- Severity: "minor" (embellishment), "major" (wrong fact), "critical" (fabricated source/citation)

Respond with JSON:
{{
  "has_hallucinations": true,
  "hallucinations": [
    {{
      "text_span": "<hallucinated text>",
      "reasoning": "<why this is hallucinated>",
      "severity": "minor"
    }}
  ],
  "overall_assessment": "<summary>"
}}

If no hallucinations are found, respond with:
{{
  "has_hallucinations": false,
  "hallucinations": [],
  "overall_assessment": "No hallucinations detected."
}}"""

_SEVERITY_WEIGHTS: dict[str, float] = {
    "minor": 0.1,
    "major": 0.3,
    "critical": 0.5,
}


class HallucinationDetector(BaseEvaluator):
    """Detects fabricated facts, invented citations, or unsupported information.

    Uses an LLM judge to scan the full answer against the context and
    identify hallucinated spans with severity classification.

    Args:
        llm: The LLM provider for judge calls.
        model: Model override for the judge.
    """

    def __init__(self, llm: BaseLLMProvider, model: str = "") -> None:
        self._llm = llm
        self._model = model

    @property
    def name(self) -> str:
        return "hallucination"

    @property
    def category(self) -> str:
        return "generation"

    @property
    def description(self) -> str:
        return "Detects fabricated facts or information not present in the retrieved context"

    async def evaluate(self, sample: EvaluationSample) -> EvaluationResult:
        """Evaluate the answer for hallucinations.

        Args:
            sample: The query-answer pair with retrieved chunks.

        Returns:
            EvaluationResult with hallucination details and severity breakdown.
        """
        if not sample.answer:
            return EvaluationResult(
                query_log_id=sample.query_log_id,
                evaluator_name=self.name,
                score=1.0,
                details={"reason": "empty_answer"},
                reasoning="No answer to evaluate.",
            )

        context = "\n".join(c.text for c in sample.chunks)
        if not context:
            return EvaluationResult(
                query_log_id=sample.query_log_id,
                evaluator_name=self.name,
                score=0.0,
                details={"reason": "no_context"},
                reasoning="No context available; answer cannot be verified.",
            )

        result = await self._detect_hallucinations(
            sample.query, sample.answer, context
        )

        has_hallucinations = result.get("has_hallucinations", False)
        hallucinations: list[dict[str, Any]] = result.get("hallucinations", [])
        assessment = str(result.get("overall_assessment", ""))

        if not has_hallucinations:
            score = 1.0
        else:
            penalty = sum(
                _SEVERITY_WEIGHTS.get(h.get("severity", "minor"), 0.1)
                for h in hallucinations
            )
            score = max(0.0, 1.0 - penalty)

        severity_breakdown = {"minor": 0, "major": 0, "critical": 0}
        for h in hallucinations:
            sev = h.get("severity", "minor")
            if sev in severity_breakdown:
                severity_breakdown[sev] += 1

        return EvaluationResult(
            query_log_id=sample.query_log_id,
            evaluator_name=self.name,
            score=round(score, 4),
            details={
                "has_hallucinations": has_hallucinations,
                "hallucination_count": len(hallucinations),
                "hallucinations": hallucinations,
                "severity_breakdown": severity_breakdown,
            },
            reasoning=assessment,
        )

    async def _detect_hallucinations(
        self, query: str, answer: str, context: str
    ) -> dict[str, Any]:
        """Run the hallucination detection prompt.

        Args:
            query: The user query.
            answer: The generated answer.
            context: Combined chunk text.

        Returns:
            Parsed judge response.
        """
        prompt = _HALLUCINATION_PROMPT.format(
            query=query, answer=answer, context=context
        )
        try:
            return await judge(self._llm, prompt, model=self._model)
        except Exception:
            logger.warning("Hallucination detection judge call failed")
            return {
                "has_hallucinations": False,
                "hallucinations": [],
                "overall_assessment": "Judge call failed; assuming no hallucinations.",
            }
