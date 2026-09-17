# tourist_assistant/security/prompt_guard.py
from __future__ import annotations

import logging
from typing import Literal

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from tourist_assistant.config import settings

logger = logging.getLogger(__name__)

_tokenizer: AutoTokenizer | None = None
_model: AutoModelForSequenceClassification | None = None


def _load_model() -> tuple[AutoTokenizer, AutoModelForSequenceClassification]:
    global _tokenizer, _model
    if _tokenizer is None or _model is None:
        logger.info("Loading prompt guard model: %s", settings.prompt_guard_model)
        _tokenizer = AutoTokenizer.from_pretrained(settings.prompt_guard_model)
        _model = AutoModelForSequenceClassification.from_pretrained(settings.prompt_guard_model)
        _model.eval()
        logger.info("Prompt guard model loaded. Labels: %s", _model.config.id2label)
    return _tokenizer, _model


def check_prompt(text: str) -> tuple[Literal["benign", "injection", "jailbreak"], float]:
    tokenizer, model = _load_model()
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=settings.prompt_guard_max_tokens,
    )
    with torch.no_grad():
        logits = model(**inputs).logits

    probabilities = torch.softmax(logits, dim=-1)[0]
    predicted_id = logits.argmax().item()
    label = model.config.id2label[predicted_id].lower()
    confidence = probabilities[predicted_id].item()

    if label not in {"benign", "injection", "jailbreak"}:
        label = "jailbreak" if confidence > settings.prompt_guard_threshold else "benign"

    return label, confidence


def is_malicious(
    text: str, threshold: float | None = None
) -> tuple[bool, str, float]:
    effective_threshold = threshold if threshold is not None else settings.prompt_guard_threshold
    label, confidence = check_prompt(text)
    malicious = label != "benign" and confidence >= effective_threshold
    return malicious, label, confidence


def load_prompt_guard() -> None:
    """Eagerly load the prompt guard model."""
    _load_model()
    logger.info("Prompt guard ready.")