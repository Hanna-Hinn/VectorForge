"""Evaluator registry — manages available evaluator implementations."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from vectorforge.evaluation.base import BaseEvaluator
from vectorforge.exceptions import ConfigurationError, DuplicateError

if TYPE_CHECKING:
    from vectorforge.llm.base import BaseLLMProvider

logger = logging.getLogger(__name__)


class EvaluatorRegistry:
    """Registry of evaluator instances with category-based lookup.

    Follows the same registry pattern as
    :class:`~vectorforge.embedding.registry.EmbeddingProviderRegistry`.
    """

    def __init__(self) -> None:
        self._evaluators: dict[str, BaseEvaluator] = {}

    def register(self, evaluator: BaseEvaluator) -> None:
        """Register an evaluator instance.

        Args:
            evaluator: The evaluator to register.

        Raises:
            DuplicateError: If an evaluator with this name already exists.
        """
        name = evaluator.name
        if name in self._evaluators:
            msg = f"Evaluator '{name}' is already registered"
            raise DuplicateError(msg)
        self._evaluators[name] = evaluator
        logger.info("Registered evaluator: %s", name)

    def get(self, name: str) -> BaseEvaluator:
        """Get an evaluator by name.

        Args:
            name: The evaluator name.

        Returns:
            The registered BaseEvaluator.

        Raises:
            ConfigurationError: If no evaluator with that name is registered.
        """
        if name not in self._evaluators:
            msg = f"Evaluator '{name}' not registered"
            raise ConfigurationError(msg)
        return self._evaluators[name]

    def list_available(self) -> list[str]:
        """List all registered evaluator names.

        Returns:
            Sorted list of evaluator name strings.
        """
        return sorted(self._evaluators.keys())

    def get_by_category(self, category: str) -> list[BaseEvaluator]:
        """Get all evaluators belonging to a category.

        Args:
            category: The category to filter by (``retrieval`` or ``generation``).

        Returns:
            List of matching evaluator instances.
        """
        return [
            evaluator
            for evaluator in self._evaluators.values()
            if evaluator.category == category
        ]

    def register_defaults(
        self,
        llm: BaseLLMProvider | None = None,
        judge_model: str = "",
    ) -> None:
        """Register all built-in evaluator implementations.

        LLM-based evaluators are only registered when an ``llm`` provider
        is supplied.  The ``EmbeddingDriftDetector`` is always registered
        since it does not require an LLM.

        Args:
            llm: Optional LLM provider for judge-based evaluators.
            judge_model: Model override for the judge LLM.
        """
        from vectorforge.evaluation.evaluators.embedding_drift import (
            EmbeddingDriftDetector,
        )

        evaluators: list[BaseEvaluator] = [EmbeddingDriftDetector()]

        if llm is not None:
            from vectorforge.evaluation.evaluators.answer_relevance import (
                AnswerRelevanceEvaluator,
            )
            from vectorforge.evaluation.evaluators.chunk_coverage import (
                ChunkCoverageEvaluator,
            )
            from vectorforge.evaluation.evaluators.faithfulness import (
                FaithfulnessEvaluator,
            )
            from vectorforge.evaluation.evaluators.hallucination import (
                HallucinationDetector,
            )
            from vectorforge.evaluation.evaluators.retrieval_relevance import (
                RetrievalRelevanceEvaluator,
            )

            evaluators.extend([
                RetrievalRelevanceEvaluator(llm, model=judge_model),
                ChunkCoverageEvaluator(llm, model=judge_model),
                FaithfulnessEvaluator(llm, model=judge_model),
                AnswerRelevanceEvaluator(llm, model=judge_model),
                HallucinationDetector(llm, model=judge_model),
            ])

        for evaluator in evaluators:
            if evaluator.name not in self._evaluators:
                self.register(evaluator)
