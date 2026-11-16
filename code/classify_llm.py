"""
LLM-based classifier for harassment and doxing content.

Connects to a vLLM OpenAI-compatible endpoint and returns structured
JSON classifications for each post. Uses guided JSON decoding so the
model is constrained to the output schema at the token level.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any

from openai import OpenAI
from pydantic import BaseModel, ValidationError, field_validator

logger = logging.getLogger(__name__)

PROMPTS_DIR = Path(__file__).parent / "prompts"
SYSTEM_PROMPT = (PROMPTS_DIR / "system_prompt.txt").read_text()

LABEL_VALUES = ("negative", "positive")
CONFIDENCE_VALUES = ("low", "medium", "high")

# JSON schema passed to vLLM guided_json for constrained decoding.
RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "label": {"type": "string", "enum": list(LABEL_VALUES)},
        "confidence": {"type": "string", "enum": list(CONFIDENCE_VALUES)},
        "reasoning": {"type": "string", "maxLength": 500},
    },
    "required": ["label", "confidence", "reasoning"],
    "additionalProperties": False,
}


class ClassificationResult(BaseModel):
    label: str
    confidence: str
    reasoning: str

    @field_validator("label")
    @classmethod
    def validate_label(cls, v: str) -> str:
        if v not in LABEL_VALUES:
            raise ValueError(f"label must be one of {LABEL_VALUES}, got {v!r}")
        return v

    @field_validator("confidence")
    @classmethod
    def validate_confidence(cls, v: str) -> str:
        if v not in CONFIDENCE_VALUES:
            raise ValueError(f"confidence must be one of {CONFIDENCE_VALUES}, got {v!r}")
        return v


def _build_user_message(post: dict[str, Any]) -> str:
    thread_title = post.get("thread_title", "")
    author = post.get("author", "unknown")
    content = post.get("content", "")
    previous_posts = post.get("previous_posts") or []

    parts = []
    if thread_title:
        parts.append(f"[Thread: {thread_title}]")
    parts.append(f"[Author: {author}]")
    if previous_posts:
        parts.append("Previous posts in the thread:")
        for prev in previous_posts:
            prev_author = prev.get("author", "unknown")
            prev_content = prev.get("content", "")
            parts.append(f"[Previous Author: {prev_author}] {prev_content}")
        parts.append("Current post:")
    parts.append(content)
    return "\n".join(parts)


def _parse_response(raw: str) -> ClassificationResult:
    """Parse and validate the model's JSON response."""
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"Model returned non-JSON: {raw[:200]!r}") from e
    try:
        return ClassificationResult(**data)
    except (ValidationError, TypeError) as e:
        raise ValueError(f"Schema validation failed: {e}") from e


class LLMClassifier:
    """
    Classifies posts by sending them to a vLLM endpoint.

    Parameters
    ----------
    base_url:
        OpenAI-compatible API root (e.g. http://localhost:8000/v1).
    model:
        Model name as registered in vLLM (e.g. Qwen/Qwen3-30B-A3B).
    max_retries:
        Number of times to retry a failed request before giving up.
    retry_delay:
        Seconds to wait between retries.
    thinking_budget:
        Token budget for Qwen3 thinking mode (0 disables thinking).
    """

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        max_retries: int = 3,
        retry_delay: float = 2.0,
        thinking_budget: int = 0,
    ) -> None:
        self.base_url = base_url or os.environ.get(
            "VLLM_BASE_URL", "http://localhost:8000/v1"
        )
        self.model = model or os.environ.get(
            "VLLM_MODEL", "Qwen/Qwen3.5-27B"
        )
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.thinking_budget = thinking_budget

        self._client = OpenAI(
            base_url=self.base_url,
            api_key=os.environ.get("VLLM_API_KEY", "EMPTY"),
        )

    def classify_post(self, post: dict[str, Any]) -> ClassificationResult:
        """
        Classify a single post.

        Raises ValueError if the model returns unparseable output after
        all retries are exhausted.
        """
        user_message = _build_user_message(post)

        extra_body: dict[str, Any] = {
            "guided_json": RESPONSE_SCHEMA,
            "chat_template_kwargs": {
                "enable_thinking": self.thinking_budget > 0,
            },
        }

        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                response = self._client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_message},
                    ],
                    temperature=0.0,
                    max_tokens=300,
                    extra_body=extra_body,
                )
                raw = response.choices[0].message.content or ""
                return _parse_response(raw)
            except (ValueError, Exception) as e:
                last_error = e
                if attempt < self.max_retries:
                    logger.warning(
                        "classify attempt %d/%d failed for message_id=%s: %s",
                        attempt,
                        self.max_retries,
                        post.get("message_id"),
                        e,
                    )
                    time.sleep(self.retry_delay)

        raise ValueError(
            f"All {self.max_retries} attempts failed for message_id={post.get('message_id')}: {last_error}"
        )

    def classify_batch(
        self, posts: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """
        Classify a list of posts, returning enriched records.

        Each returned record is the original post dict merged with the
        classification fields: label, confidence, reasoning, llm_error.
        Posts that fail all retries get label="error" for downstream triage.
        """
        results = []
        for post in posts:
            # previous_posts is only used to build the prompt; do not retain it
            # in the returned output schema.
            record = {k: v for k, v in post.items() if k != "previous_posts"}
            try:
                result = self.classify_post(post)
                record["label"] = result.label
                record["confidence"] = result.confidence
                record["reasoning"] = result.reasoning
                record["llm_error"] = None
            except ValueError as e:
                logger.error("Failed to classify message_id=%s: %s", post.get("message_id"), e)
                record["label"] = "error"
                record["confidence"] = "low"
                record["reasoning"] = ""
                record["llm_error"] = str(e)
            results.append(record)
        return results
