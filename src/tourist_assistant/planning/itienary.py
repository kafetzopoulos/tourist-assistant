from __future__ import annotations

import logging
from datetime import datetime, timedelta

from tourist_assistant.config import settings
from tourist_assistant.models import (
    Activity,
    Attraction,
    Itinerary,
    UserPreferences,
    WeatherInfo,
)
from tourist_assistant.planning.feasibility import (
    build_itinerary,
    check_itinerary,
    plan_day,
    summarize,
)
from tourist_assistant.rag.retriever import get_retriever
from tourist_assistant.tools.weather import get_hourly_weather

logger = logging.getLogger(__name__)


class ItineraryError(RuntimeError):
    """Raised when an itinerary operation cannot be completed."""


class ItineraryService:
    """
    Domain service for creating and mutating itineraries.

    Owns the interaction between the retriever (candidate selection), the
    weather tool (hourly forecast), and the pure feasibility engine.
    Contains no LLM calls and no session state — those belong to the
    ConversationManager.
    """

    def __init__(self) -> None:
        self.retriever = get_retriever()

    # ---------- creation ----------

    def create(
        self,
        query: str,
        start_time: datetime,
        end_time: datetime,
        prefs: UserPreferences,
        origin: tuple[float, float] | None = None,
        candidate_limit: int = 8,
    ) -> Itinerary:
        """
        Build a new feasible itinerary from a natural-language-ish query.

        Candidate selection is done by RAG; scheduling is done by the
        deterministic planner. Weather is fetched for the target date and
        passed into the planner so outdoor activities can be reordered.
        """
        candidates = self._select_candidates(query, prefs, limit=candidate_limit)
        if not candidates:
            raise ItineraryError("No attractions matched the request.")

        weather = self._weather_for(start_time)
        itinerary = plan_day(
            candidates,
            start_time=start_time,
            end_time=end_time,
            prefs=prefs,
            origin=origin,
            weather_by_hour=weather,
        )
        logger.info("Created itinerary: %s", summarize(itinerary))
        return itinerary

    # ---------- mutation ----------

    def remove(self, itinerary: Itinerary, target: str) -> Itinerary:
        """Remove activities whose name contains `target` (case-insensitive)."""
        before = len(itinerary.activities)
        itinerary.activities = [
            a for a in itinerary.activities if target.lower() not in a.name.lower()
        ]
        removed = before - len(itinerary.activities)
        if removed == 0:
            itinerary.feasibility_notes.append(
                f"No activity matched '{target}' — nothing removed."
            )
        else:
            itinerary.feasibility_notes.append(
                f"Removed {removed} activity matching '{target}'."
            )
            itinerary = self._resequence(itinerary)
        return itinerary

    def replace(self, itinerary: Itinerary, target: str, prefs: UserPreferences) -> Itinerary:
        """Remove `target` and slot in the best alternative that still fits."""
        replacement = self._find_replacement(target, prefs)
        if replacement is None:
            itinerary.feasibility_notes.append(
                f"No replacement found for '{target}'."
            )
            return itinerary

        kept = [a for a in itinerary.activities if target.lower() not in a.name.lower()]
        rebuilt = self.rebuild(
            itinerary,
            attractions=self._resolve_attractions(kept) + [replacement],
            prefs=prefs,
        )
        rebuilt.feasibility_notes.append(
            f"Replaced '{target}' with '{replacement.name}'."
        )
        return rebuilt

    def add(self, itinerary: Itinerary, target: str, prefs: UserPreferences) -> Itinerary:
        """Insert a new attraction and re-flow the schedule."""
        attraction = self._find_replacement(target, prefs)
        if attraction is None:
            itinerary.feasibility_notes.append(
                f"Couldn't find '{target}' in the knowledge base."
            )
            return itinerary

        existing = self._resolve_attractions(itinerary.activities)
        rebuilt = self.rebuild(itinerary, attractions=existing + [attraction], prefs=prefs)
        if attraction.id not in {a.attraction_id for a in rebuilt.activities}:
            rebuilt.feasibility_notes.append(
                f"'{attraction.name}' didn't fit in the current window."
            )
        else:
            rebuilt.feasibility_notes.append(f"Added '{attraction.name}'.")
        return rebuilt

    def replan(self, itinerary: Itinerary, prefs: UserPreferences) -> Itinerary:
        """Rebuild the same set of attractions with (possibly) new preferences."""
        attractions = self._resolve_attractions(itinerary.activities)
        if not attractions:
            raise ItineraryError("Current itinerary has no resolvable attractions.")
        return self.rebuild(itinerary, attractions=attractions, prefs=prefs)

    def rebuild(
        self,
        template: Itinerary,
        attractions: list[Attraction],
        prefs: UserPreferences,
    ) -> Itinerary:
        """
        Re-run the feasibility engine over a fixed set of attractions while
        preserving the original time window. Used by replace/add/replan.
        """
        weather = self._weather_for(template.start_time)
        return plan_day(
            attractions,
            start_time=template.start_time,
            end_time=template.end_time,
            prefs=prefs,
            weather_by_hour=weather,
        )

    # ---------- validation ----------

    def validate(self, itinerary: Itinerary, prefs: UserPreferences) -> list[str]:
        """Expose the feasibility checker for externally-proposed plans."""
        return check_itinerary(itinerary, prefs)

    # ---------- rendering ----------

    def to_structured_payload(self, itinerary: Itinerary) -> dict:
        """
        Render the itinerary as a compact dict for the LLM to phrase.

        This is the only representation that crosses into the LLM. Keeping it
        minimal and explicit makes prompt injection via names harder and gives
        the evaluation harness a stable thing to assert against.
        """
        return {
            "window": {
                "start": itinerary.start_time.strftime("%H:%M"),
                "end": itinerary.end_time.strftime("%H:%M"),
            },
            "feasible": itinerary.feasible,
            "activities": [
                {
                    "name": a.name,
                    "start": a.start_time.strftime("%H:%M"),
                    "end": a.end_time.strftime("%H:%M"),
                    "travel_min": a.travel_time_before_min,
                    "indoor": a.indoor,
                }
                for a in itinerary.activities
            ],
            "notes": itinerary.feasibility_notes,
        }

    def to_text(self, itinerary: Itinerary) -> str:
        """Plain-text rendering for logs and the eval harness."""
        lines = []
        for a in itinerary.activities:
            lines.append(
                f"{a.start_time:%H:%M}-{a.end_time:%H:%M} {a.name} "
                f"(travel {a.travel_time_before_min} min)"
            )
        return "\n".join(lines) or "(empty)"

    # ---------- internals ----------

    def _select_candidates(
        self, query: str, prefs: UserPreferences, limit: int
    ) -> list[Attraction]:
        enriched = self._enrich_query(query, prefs)
        ranked = self.retriever.search_attractions(enriched, top_k=limit)
        candidates = [a for a, _ in ranked]
        # Fall back to everything if the RAG signal is weak, so we don't
        # return an empty plan for vague queries.
        return candidates or self.retriever.all_attractions()

    def _find_replacement(
        self, target: str, prefs: UserPreferences
    ) -> Attraction | None:
        # Exact-ish name match first.
        for a in self.retriever.all_attractions():
            if target.lower() in a.name.lower():
                return a
        # Semantic fallback, filtered against dislikes.
        for a, _ in self.retriever.search_attractions(target, top_k=5):
            if any(c in prefs.dislikes for c in a.categories):
                continue
            return a
        return None

    def _resolve_attractions(self, activities: list[Activity]) -> list[Attraction]:
        out: list[Attraction] = []
        for act in activities:
            att = self.retriever.get_attraction(act.attraction_id)
            if att is not None:
                out.append(att)
        return out

    def _resequence(self, itinerary: Itinerary) -> Itinerary:
        """
        Re-flow start/end times after a removal so there are no gaps or overlaps.

        Keeps the same start time and total window; travel times are preserved
        per activity because they were computed for the original ordering. The
        rebuild path is preferred when travel times matter, but resequencing is
        cheap and useful for removing the last activity.
        """
        if not itinerary.activities:
            return itinerary

        attractions = self._resolve_attractions(itinerary.activities)
        if len(attractions) != len(itinerary.activities):
            # Can't re-flow safely; leave as-is with a note.
            itinerary.feasibility_notes.append(
                "Could not resequence — some activities are no longer resolvable."
            )
            return itinerary

        rebuilt = self.rebuild(
            itinerary,
            attractions=attractions,
            prefs=UserPreferences(),  # pace/transport already baked into the plan
        )
        # Preserve the removal note from the original.
        rebuilt.feasibility_notes = (
            itinerary.feasibility_notes + rebuilt.feasibility_notes
        )
        return rebuilt

    def _enrich_query(self, query: str, prefs: UserPreferences) -> str:
        parts = [query]
        if prefs.interests:
            parts.append("interests: " + ", ".join(c.value for c in prefs.interests))
        if prefs.dislikes:
            parts.append("avoid: " + ", ".join(c.value for c in prefs.dislikes))
        if prefs.has_children:
            parts.append("family friendly")
        return " | ".join(parts)

    def _weather_for(self, dt: datetime) -> dict[int, WeatherInfo]:
        try:
            return get_hourly_weather(dt.date())
        except Exception:  # noqa: BLE001
            logger.warning("Weather fetch failed; continuing without forecast.")
            return {}


def get_itinerary_service() -> ItineraryService:
    return ItineraryService()