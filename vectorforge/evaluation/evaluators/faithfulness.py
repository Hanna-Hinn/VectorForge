"""FaithfulnessEvaluator — measures whether answers are grounded in retrieved context."""

from __future__ import annotations

import logging
from typing import Any

from vectorforge.evaluation.base import BaseEvaluator
from vectorforge.evaluation.evaluators._judge import judge
from vectorforge.evaluation.types import EvaluationResult, EvaluationSample
from vectorforge.llm.base import BaseLLMProvider

logger = logging.getLogger(__name__)

_EXTRACT_CLAIMS_PROMPT = """Extract all factual claims from the following answer.
Each claim should be a single, verifiable statement.

Answer: {answer}

Respond with JSON: {{"claims": ["claim1", "claim2", ...]}}"""

_VERIFY_CLAIM_PROMPT = """Given the following context, determine if the claim is
supported by the information in the context.

Context:
{context}

Claim: {claim}

Respond with JSON:
{{
  "verdict": "supported" | "unsupported" | "ambiguous",
  "reasoning": "<explanation>",
  "supporting_text": "<quote from context if supported, null if not>"
}}"""


class FaithfulnessEvaluator(BaseEvaluator):
    """Evaluates whether the generated answer is grounded in retrieved context.

    Extracts claims from the answer, then verifies each claim against
    the combined chunk context using an LLM judge.

    Args:
        llm: The LLM provider for judge calls.
        model: Model override for the judge.
    """

    def __init__(self, llm: BaseLLMProvider, model: str = "") -> None:
        self._llm = llm
        self._model = model

    @property
    def name(self) -> str:
        return "faithfulness"

    @property
    def category(self) -> str:
        return "generation"

    @property
    def description(self) -> str:
        return "Measures whether the generated answer is grounded in the retrieved context"

    async def evaluate(self, sample: EvaluationSample) -> EvaluationResult:
        """Evaluate faithfulness of the generated answer.

        Args:
            sample: The query-answer pair with retrieved chunks.

        Returns:
            EvaluationResult with per-claim verdicts.
        """
        if not sample.answer:
            return EvaluationResult(
                query_log_id=sample.query_log_id,
                evaluator_name=self.name,
                score=1.0,
                details={"reason": "empty_answer"},
                reasoning="No answer to evaluate.",
            )

        # Step 1: Extract claims
        claims = await self._extract_claims(sample.answer)
        if not claims:
            return EvaluationResult(
                query_log_id=sample.query_log_id,
                evaluator_name=self.name,
                score=1.0,
                details={"reason": "no_claims"},
                reasoning="No factual claims found in answer.",
            )

        # Step 2: Verify claims
        context = "\n".join(c.text for c in sample.chunks)
        verdicts: list[dict[str, Any]] = []
        supported = 0
        unsupported = 0
        ambiguous = 0

        for claim in claims:
            verdict = await self._verify_claim(claim, context)
            verdicts.append(verdict)
            v = verdict.get("verdict", "unsupported")
            if v == "supported":
                supported += 1
            elif v == "ambiguous":
                ambiguous += 1
            else:
                unsupported += 1

        total = len(claims)
        score = supported / total if total > 0 else 1.0

        return EvaluationResult(
            query_log_id=sample.query_log_id,
            evaluator_name=self.name,
            score=round(score, 4),
            details={
                "total_claims": total,
                "supported_claims": supported,
                "unsupported_claims": unsupported,
                "ambiguous_claims": ambiguous,
                "per_claim_verdicts": verdicts,
            },
            reasoning=f"Faithfulness: {supported}/{total} claims supported by context",
        )

    async def _extract_claims(self, answer: str) -> list[str]:
        """Extract factual claims from the answer.

        Args:
            answer: The generated answer text.

        Returns:
            List of claim strings.
        """
        prompt = _EXTRACT_CLAIMS_PROMPT.format(answer=answer)
        try:
            result = await judge(self._llm, prompt, model=self._model)
            claims = result.get("claims", [])
            return [str(c) for c in claims if c]
        except Exception:
            logger.warning("Failed to extract claims from answer")
            return []

    async def _verify_claim(self, claim: str, context: str) -> dict[str, Any]:
        """Verify a single claim against the context.

        Args:
            claim: The claim to verify.
            context: Combined chunk text.

        Returns:
            Dict with verdict, reasoning, and supporting_text.
        """
        prompt = _VERIFY_CLAIM_PROMPT.format(claim=claim, context=context)
        try:
            return await judge(self._llm, prompt, model=self._model)
        except Exception:
            logger.warning("Failed to verify claim: %s", claim[:50])
            return {"verdict": "unsupported", "reasoning": "Judge call failed"}
