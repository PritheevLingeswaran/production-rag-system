from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0


class OpenAIClient:
    """OpenAI-compatible client wrapper.

    Thin wrapper so you can swap gateways/providers without rewriting the app.
    """

    def __init__(
        self,
        api_key: Optional[str],
        base_url: Optional[str] = None,
        organization: Optional[str] = None,
        timeout_s: float = 30.0,
        max_retries: int = 3,
    ) -> None:
        self._client = OpenAI(api_key=api_key, base_url=base_url, organization=organization, timeout=timeout_s)
        self._max_retries = max_retries

    @retry(wait=wait_exponential(min=0.5, max=8), stop=stop_after_attempt(3), reraise=True)
    def embed(self, model: str, inputs: List[str]) -> Tuple[List[List[float]], Usage]:
        resp = self._client.embeddings.create(model=model, input=inputs)
        vectors = [d.embedding for d in resp.data]
        usage = Usage(total_tokens=getattr(resp.usage, "total_tokens", 0))
        return vectors, usage

    @retry(wait=wait_exponential(min=0.5, max=8), stop=stop_after_attempt(3), reraise=True)
    def chat(
        self,
        model: str,
        messages: List[dict],
        temperature: float,
        max_output_tokens: int,
        response_format: Optional[dict] = None,
    ) -> Tuple[str, Usage]:
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
