"""Abstract base evaluator for RAG quality evaluation."""

from __future__ import annotations

from abc import ABC, abstractmethod

from vectorforge.evaluation.types import EvaluationResult, EvaluationSample


class BaseEvaluator(ABC):
    """Abstract base class for all RAG evaluators.

    Each evaluator assesses one dimension of quality (e.g. faithfulness,
    relevance, hallucination).  Subclasses implement ``evaluate`` for a
    single sample and optionally override ``evaluate_batch`` for custom
    batching logic.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier for this evaluator."""

    @property
    @abstractmethod
    def category(self) -> str:
        """Category: ``retrieval`` or ``generation``."""

    @property
    @abstractmethod
    def description(self) -> str:
        """Human-readable description of what this evaluator measures."""

    @abstractmethod
    async def evaluate(
        self,
        sample: EvaluationSample,
    ) -> EvaluationResult:
        """Evaluate a single query-answer pair.

        Args:
            sample: The evaluation sample to assess.

        Returns:
            An EvaluationResult with score, details, and optional reasoning.
        """

    async def evaluate_batch(
        self,
        samples: list[EvaluationSample],
    ) -> list[EvaluationResult]:
        """Evaluate a batch of samples.

        Default implementation calls ``evaluate`` sequentially.
        Override for batched LLM calls or other optimisations.

        Args:
            samples: List of evaluation samples.

        Returns:
            List of EvaluationResults, one per sample.
        """
        results: list[EvaluationResult] = []
        for sample in samples:
            result = await self.evaluate(sample)
            results.append(result)
        return results
