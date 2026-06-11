"""LLM-as-judge evaluation for generated Arabic responses."""

import logging
from pathlib import Path

from llm.client import OpenRouterClient, OpenRouterError

logger = logging.getLogger(__name__)

_PROMPT_PATH = Path("llm/prompts/judge_prompt.txt")
_RUBRIC_FIELDS = ("dialect_match", "correctness", "tone", "helpfulness")


class LLMJudge:
    """Wraps an LLM call that scores a generated response against a rubric."""

    def __init__(self, llm_client: OpenRouterClient | None = None) -> None:
        """Initialize the judge.

        Args:
            llm_client: Optional `OpenRouterClient` instance. If `None`, a
                new client is created from application settings.
        """
        self.llm_client = llm_client if llm_client is not None else OpenRouterClient()
        self._template = _PROMPT_PATH.read_text(encoding="utf-8")

    def judge(self, message: str, response: str, expected_dialect: str) -> dict[str, float | str]:
        """Score a generated response on dialect match, correctness, tone, and helpfulness.

        Args:
            message: The original customer message.
            response: The generated Arabic response to evaluate.
            expected_dialect: The dialect expected for this case.

        Returns:
            dict[str, float | str]: A dictionary with integer scores
                (1-5) for each rubric field plus a `comment`. All rubric
                fields default to `0` and `comment` to an empty string if
                the LLM call fails or returns invalid JSON.
        """
        prompt = self._template.format(
            message=message,
            expected_dialect=expected_dialect,
            response=response,
        )

        try:
            llm_response = self.llm_client.chat_completion(
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.0,
            )
            result = self.llm_client.extract_json(llm_response)
        except OpenRouterError as exc:
            logger.warning("LLM judge call failed: %s", exc)
            result = {}

        scores: dict[str, float | str] = {}
        for field in _RUBRIC_FIELDS:
            try:
                scores[field] = max(1, min(5, int(result.get(field, 0))))
            except (TypeError, ValueError):
                scores[field] = 0
        scores["comment"] = str(result.get("comment", ""))
        return scores
