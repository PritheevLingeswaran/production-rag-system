from __future__ import annotations

from generation.prompts import load_prompt
from utils.logging import get_logger
from utils.openai_client import OpenAIClient
from utils.settings import Settings

log = get_logger(__name__)


def rewrite_query(settings: Settings, client: OpenAIClient, question: str) -> str:
    prompt = load_prompt("prompts/query_rewrite.txt")
    messages = [
        {"role": "system", "content": "You rewrite user questions into retrieval queries."},
        {"role": "user", "content": f"{prompt}\n\nUSER_QUESTION: {question}\nREWRITTEN_QUERY:"},
    ]
    text, _ = client.chat(
        model=settings.retrieval.query_rewrite.model,
        messages=messages,
        temperature=0.0,
        max_output_tokens=64,
    )
    rewritten = text.strip().strip('"')
    return rewritten if rewritten else question
