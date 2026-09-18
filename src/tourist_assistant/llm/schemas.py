from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

from tourist_assistant.models import Category, Pace, TransportMode


class RouterOutput(BaseModel):
    intent: Literal[
        "weather_query",
        "factual_qa",
        "recommendation",
        "plan_request",
        "modify_plan",
        "out_of_scope",
        "chitchat",
        "conversation_meta",
    ]
    time_window_hours: float | None = None
    start_date: str | None = None  
    destination: str | None = None
    preference_keywords: list[str] = Field(default_factory=list)
    reason: str | None = None

    @field_validator("preference_keywords", mode="before")
    @classmethod
    def _coerce_none_to_list(cls, v: Any) -> Any:
        return [] if v is None else v


class PreferenceOutput(BaseModel):
    interests: list[Category] = Field(default_factory=list)
    dislikes: list[Category] = Field(default_factory=list)
    pace: Pace | None = None
    transport_mode: TransportMode | None = None
    has_children: bool | None = None
    has_car: bool | None = None

    @field_validator("interests", "dislikes", mode="before")
    @classmethod
    def _coerce_none_to_list(cls, v: Any) -> Any:
        return [] if v is None else v

class ModifyOutput(BaseModel):
    action: Literal[
        "remove",
        "replace",
        "add",
        "change_time",
        "change_pace",
        "change_transport",
        "none",
    ]
    target: str | None = None
    value: str | None = None
    reason: str
    