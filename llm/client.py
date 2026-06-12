"""HTTP client for the OpenRouter chat-completions API."""

import json
import logging
import time
from typing import Any

import httpx

from config import settings

logger = logging.getLogger(__name__)

# Approximate USD price per 1K tokens (prompt, completion) for cost estimation.
_PRICE_TABLE: dict[str, tuple[float, float]] = {
    "qwen/qwen-2.5-72b-instruct": (0.00035, 0.0004),
    "qwen/qwen3-14b": (0.00006, 0.00024),
    "qwen/qwen3-14b:free": (0.0, 0.0),
    "deepseek/deepseek-chat": (0.00014, 0.00028),
    "meta-llama/llama-3.1-70b-instruct": (0.0004, 0.0004),
}
_DEFAULT_PRICE: tuple[float, float] = (0.0005, 0.0005)


class OpenRouterError(RuntimeError):
    """Raised when the OpenRouter API call fails after all retries."""


class OpenRouterClient:
    """Thin client around the OpenRouter chat-completions endpoint."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        timeout: int | None = None,
        max_retries: int | None = None,
    ) -> None:
        """Initialize the client.

        Args:
            api_key: OpenRouter API key. Defaults to `settings.openrouter_api_key`.
            model: Default model identifier. Defaults to `settings.openrouter_model`.
            base_url: API base URL. Defaults to `settings.openrouter_base_url`.
            timeout: Request timeout in seconds. Defaults to `settings.request_timeout`.
            max_retries: Maximum retry attempts. Defaults to `settings.max_retries`.
        """
        self.api_key = api_key or settings.openrouter_api_key
        self.model = model or settings.openrouter_model
        self.base_url = (base_url or settings.openrouter_base_url).rstrip("/")
        self.timeout = timeout if timeout is not None else settings.request_timeout
        self.max_retries = max_retries if max_retries is not None else settings.max_retries

        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_cost_usd = 0.0

        self._client = httpx.Client(timeout=self.timeout)

    def close(self) -> None:
        """Close the underlying HTTP connection pool."""
        self._client.close()

    def __enter__(self) -> "OpenRouterClient":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def _headers(self) -> dict[str, str]:
        """Build the HTTP headers required by OpenRouter."""
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json; charset=utf-8",
            "HTTP-Referer": "https://github.com/arabic-customer-support-agent",
            "X-Title": "Arabic Customer Support Agent",
        }

    def _estimate_cost(self, model: str, prompt_tokens: int, completion_tokens: int) -> float:
        """Estimate the USD cost of a completion based on a static price table.

        Args:
            model: The model identifier used for the request.
            prompt_tokens: Number of prompt tokens consumed.
            completion_tokens: Number of completion tokens generated.

        Returns:
            float: Estimated cost in USD.
        """
        prompt_price, completion_price = _PRICE_TABLE.get(model, _DEFAULT_PRICE)
        return (prompt_tokens / 1000.0) * prompt_price + (completion_tokens / 1000.0) * completion_price

    def chat_completion(
        self,
        messages: list[dict[str, Any]],
        model: str | None = None,
        temperature: float = 0.2,
        response_format: dict[str, Any] | None = None,
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        """Call the OpenRouter chat-completions endpoint with retries.

        Args:
            messages: List of chat messages (`role`/`content` dicts).
            model: Override the default model for this call.
            temperature: Sampling temperature.
            response_format: Optional structured-output specification, e.g.
                `{"type": "json_object"}`.
            tools: Optional list of tool/function definitions for tool calling.
            max_tokens: Optional maximum number of tokens to generate.

        Returns:
            dict[str, Any]: The full decoded JSON response from OpenRouter.

        Raises:
            OpenRouterError: If the request fails after exhausting retries
                or the API returns a non-2xx status code.
        """
        if not self.api_key:
            raise OpenRouterError("OPENROUTER_API_KEY is not configured.")

        payload: dict[str, Any] = {
            "model": model or self.model,
            "messages": messages,
            "temperature": temperature,
        }
        if response_format is not None:
            payload["response_format"] = response_format
        if tools is not None:
            payload["tools"] = tools
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                response = self._client.post(
                    f"{self.base_url}/chat/completions",
                    headers=self._headers(),
                    content=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                )
                response.raise_for_status()
                data: dict[str, Any] = response.json()

                usage = data.get("usage", {})
                prompt_tokens = int(usage.get("prompt_tokens", 0))
                completion_tokens = int(usage.get("completion_tokens", 0))
                self.total_prompt_tokens += prompt_tokens
                self.total_completion_tokens += completion_tokens
                self.total_cost_usd += self._estimate_cost(
                    payload["model"], prompt_tokens, completion_tokens
                )
                return data
            except (httpx.HTTPError, json.JSONDecodeError) as exc:
                last_error = exc
                logger.warning("OpenRouter request failed (attempt %s/%s): %s", attempt, self.max_retries, exc)
                if attempt < self.max_retries:
                    time.sleep(min(2 ** attempt, 8))

        raise OpenRouterError(f"OpenRouter request failed after {self.max_retries} attempts: {last_error}")

    def extract_text(self, response: dict[str, Any]) -> str:
        """Extract the assistant message content from a chat-completion response.

        Args:
            response: The raw response dict from `chat_completion`.

        Returns:
            str: The text content of the first choice, or an empty string if absent.
        """
        try:
            return str(response["choices"][0]["message"]["content"] or "")
        except (KeyError, IndexError, TypeError):
            return ""

    def extract_json(self, response: dict[str, Any]) -> dict[str, Any]:
        """Extract and parse JSON content from a chat-completion response.

        Args:
            response: The raw response dict from `chat_completion`.

        Returns:
            dict[str, Any]: The parsed JSON object, or an empty dict if parsing fails.
        """
        text = self.extract_text(response).strip()
        if text.startswith("```"):
            text = text.strip("`")
            if text.startswith("json"):
                text = text[4:]
            text = text.strip()

        try:
            parsed = json.loads(text)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            pass

        # The model sometimes prefixes/suffixes the JSON object with stray
        # text (e.g. a leftover sentence fragment), so fall back to the
        # outermost {...} substring.
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                parsed = json.loads(text[start : end + 1])
                return parsed if isinstance(parsed, dict) else {}
            except json.JSONDecodeError:
                pass

        logger.warning("Failed to parse JSON from LLM response: %s", text[:200])
        return {}
