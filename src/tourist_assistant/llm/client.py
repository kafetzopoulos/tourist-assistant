from __future__ import annotations

import json
import logging
from functools import lru_cache
from typing import Any

from openai import AzureOpenAI, BadRequestError         
from pydantic import BaseModel, ValidationError

from tourist_assistant.config import settings

logger = logging.getLogger(__name__)


class LLMError(RuntimeError):
    pass


@lru_cache(maxsize=1)
def _client() -> AzureOpenAI:           
    return AzureOpenAI(
        api_key=settings.azure_openai_api_key,
        api_version=settings.azure_openai_api_version,
        azure_endpoint=settings.azure_openai_endpoint,
    )


def chat(
    messages: list[dict[str, str]],
    *,
    temperature: float = 0.1,
    max_tokens: int = 800,
) -> str:
    """Plain chat completion. Used for final natural-language phrasing."""
    try:
        resp = _client().chat.completions.create(
            model=settings.azure_openai_deployment,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    except Exception as exc:
        _raise_llm_error(exc)
    return resp.choices[0].message.content or ""


def chat_structured(
    messages: list[dict[str, str]],
    schema: type[BaseModel],
    *,
    temperature: float = 0.1,
    max_tokens: int = 500,
) -> BaseModel:
    """
    Structured output via JSON mode (Azure OpenAI).

    The caller passes a Pydantic model; we inject the JSON schema into the
    system message, request a JSON object, and validate the response. If the
    model returns invalid JSON, we retry once with a stricter reminder.
    """
    schema_json = json.dumps(schema.model_json_schema(), indent=2)
    augmented = list(messages)
    augmented[0] = {
        "role": "system",
        "content": (
            augmented[0]["content"]
            + "\n\nRespond ONLY with a JSON object that conforms to this schema:\n"
            + schema_json
        ),
    }

    for attempt in range(2):
        try:
            resp = _client().chat.completions.create(
                model=settings.azure_openai_deployment,
                messages=augmented,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format={"type": "json_object"},
            )
            raw = resp.choices[0].message.content or "{}"
            return schema.model_validate_json(raw)
        except (ValidationError, json.JSONDecodeError) as exc:
            logger.warning("Structured output invalid (attempt %d): %s", attempt + 1, exc)
            if attempt == 1:
                raise LLMError(f"LLM returned invalid structured output: {exc}") from exc
        except Exception as exc:
            _raise_llm_error(exc)

    raise LLMError("Unreachable")

def _raise_llm_error(exc: Exception) -> None:
    msg = str(exc).lower()
    if isinstance(exc, BadRequestError) and (
        "content management policy" in msg
        or "content_filter" in msg
        or "responsibleaipolicyviolation" in msg
    ):
        raise LLMError("Input blocked by Azure content safety policy.") from exc
    raise LLMError(f"LLM call failed: {exc}") from exc