from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from openai import OpenAI
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0


def _is_insufficient_quota_error(exc: BaseException) -> bool:
    text = str(exc).lower()
    if "insufficient_quota" in text:
        return True
    code = getattr(exc, "code", None)
    return str(code).lower() == "insufficient_quota"


def _should_retry(exc: BaseException) -> bool:
    # Quota exhaustion won't recover with retry; fail fast.
    return not _is_insufficient_quota_error(exc)


class OpenAIClient:
    """OpenAI-compatible client wrapper.

    Thin wrapper so you can swap gateways/providers without rewriting the app.
    """

    def __init__(
        self,
        api_key: str | None,
        base_url: str | None = None,
        organization: str | None = None,
        timeout_s: float = 30.0,
        max_retries: int = 3,
    ) -> None:
        normalized_api_key = api_key or None
        normalized_base_url = (base_url or "").strip() or "https://api.openai.com/v1"
        normalized_org = organization or None
        self._client = OpenAI(
            api_key=normalized_api_key,
            base_url=normalized_base_url,
            organization=normalized_org,
            timeout=timeout_s,
        )
        self._max_retries = max_retries

    @retry(
        wait=wait_exponential(min=0.5, max=8),
        stop=stop_after_attempt(3),
        retry=retry_if_exception(_should_retry),
        reraise=True,
    )
    def embed(self, model: str, inputs: list[str]) -> tuple[list[list[float]], Usage]:
        resp = self._client.embeddings.create(model=model, input=inputs)
        vectors = [d.embedding for d in resp.data]
        usage = Usage(total_tokens=getattr(resp.usage, "total_tokens", 0))
        return vectors, usage

    @retry(
        wait=wait_exponential(min=0.5, max=8),
        stop=stop_after_attempt(3),
        retry=retry_if_exception(_should_retry),
        reraise=True,
    )
    def chat(
        self,
        model: str,
        messages: list[dict],
        temperature: float,
        max_output_tokens: int,
        response_format: dict | None = None,
    ) -> tuple[str, Usage]:
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_output_tokens,
        }
        if response_format is not None:
            kwargs["response_format"] = response_format
        resp = self._client.chat.completions.create(**kwargs)
        text = resp.choices[0].message.content or ""
        u = resp.usage
        return text, Usage(
            input_tokens=getattr(u, "prompt_tokens", 0),
            output_tokens=getattr(u, "completion_tokens", 0),
            total_tokens=getattr(u, "total_tokens", 0),
        )
