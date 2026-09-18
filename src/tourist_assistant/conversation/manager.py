from __future__ import annotations

import logging
from datetime import datetime, timedelta
from datetime import date as _date

from tourist_assistant.config import settings
from tourist_assistant.conversation.state import get_session_store
from tourist_assistant.llm import prompts
from tourist_assistant.llm.client import LLMError, chat, chat_structured
from tourist_assistant.llm.schemas import (
    ModifyOutput,
    PreferenceOutput,
    RouterOutput,
)
from tourist_assistant.models import (
    Activity,
    ConversationState,
    Itinerary,
    UserPreferences,
)
# from tourist_assistant.planning import plan_day, summarize
from tourist_assistant.planning.feasibility import build_itinerary, plan_day, summarize
from tourist_assistant.rag.retriever import get_retriever
from tourist_assistant.tools.weather import get_current_weather, get_hourly_weather
from tourist_assistant.security.prompt_guard import is_malicious

SUPPORTED_DESTINATIONS = {
    "alexandroupolis",
    "alexandroupoli",
    "αλεξανδρούπολη",
}

logger = logging.getLogger(__name__)

class ConversationManager:
    def __init__(self) -> None:
        self.store = get_session_store()
        self.retriever = get_retriever()

    # ---------- public API ----------

    def handle(self, message: str, session_id: str | None = None) -> dict:

        malicious, guard_label, guard_conf = is_malicious(message)

        if malicious: 
            logger.warning(
                "Prompt guard blocked input. label=%s confidence=%.4f",
                guard_label, guard_conf,
            )
            reply = (
                "I'm sorry, but I can't process that request. "
                "I'm here to help with travel in Alexandroupolis — "
                "feel free to ask about attractions, itineraries, or the weather."
            )
            state = self.store.get_or_create(session_id)
            state.history.append({"role": "user", "content": message})
            state.history.append({"role": "assistant", "content": reply})
            self.store.save(state)
            return {"session_id": state.session_id, "reply": reply}
        
        state = self.store.get_or_create(session_id)
        state.history.append({"role": "user", "content": message})

        # router = self._route(state, message)
        # logger.info("intent=%s reason=%s", router.intent, router.reason)

        try:
            router = self._route(state, message)
            logger.info("intent=%s reason=%s", router.intent, router.reason)

            if not self._is_supported_destination(router.destination):
                reply = self._handle_out_of_scope()
            elif router.intent == "out_of_scope":
                reply = self._handle_out_of_scope()
            elif router.intent == "chitchat":
                reply = self._handle_chitchat(state, message)
            elif router.intent == "conversation_meta":
                reply = self._handle_conversation_meta(state, message)
            elif router.intent == "weather_query":
                reply = self._handle_weather(state)
            elif router.intent == "factual_qa":
                reply = self._handle_factual(state, message)
            elif router.intent == "recommendation":
                reply = self._handle_recommendation(state, message)
            elif router.intent == "plan_request":
                reply = self._handle_plan_request(state, message, router)
            elif router.intent == "modify_plan":
                reply = self._handle_modify(state, message)
            else:
                reply = "I'm not sure how to help with that. Try asking about places to visit or a plan for a few hours."
        except LLMError as exc:
            logger.warning("LLM failure handled gracefully: %s", exc)
            if state.history and state.history[-1].get("role") == "user":
                state.history.pop()
            self.store.save(state)

            return {
                "session_id": state.session_id,
                "reply": (
                    "I'm not sure how to help with that. "
                    "Try asking about places to visit or a plan for a few hours."
                ),
            }

        state.history.append({"role": "assistant", "content": reply})
        # Trim history to keep context bounded.
        state.history = state.history[-settings.history_turns_to_llm * 2 :]
        self.store.save(state)
        return {"session_id": state.session_id, "reply": reply}

    # ---------- routing ----------

    def _route(self, state: ConversationState, message: str) -> RouterOutput:
        history_txt = "\n".join(
            f"{m['role']}: {m['content']}"
            for m in state.history[-settings.history_turns_to_llm :]
        )
        now = datetime.now()
        messages = [
            {"role": "system", "content": prompts.ROUTER_SYSTEM},
            {
                "role": "user",
                "content": prompts.ROUTER_USER_TEMPLATE.format(
                    history=history_txt,
                    message=message,
                    today=now.strftime("%Y-%m-%d"),
                    weekday=now.strftime("%A"),
                ),
            },
        ]
        return chat_structured(messages, RouterOutput)

    # ---------- handlers ----------

    def _handle_out_of_scope(self) -> str:
        return (
            "I can only help with travel in Alexandroupolis using trusted sources and "
            "live data. I can't plan trips to other cities or invent places outside "
            "this area. I'm happy to suggest real Alexandroupolis attractions, plan "
            "a few hours here, or check the local weather."
        )

    def _handle_chitchat(self, state: ConversationState, message: str) -> str:
        """Reply to greetings, thanks, small talk."""
        messages = [
            {"role": "system", "content": prompts.ANSWER_SYSTEM},
            {
                "role": "user",
                "content": (
                    f"The user said: {message!r}\n\n"
                    "Reply warmly and briefly (1-2 sentences). If it's a greeting, greet "
                    "back and offer 2-3 things you can help with: attractions to visit, "
                    "a short itinerary, or the weather. Do not mention that you are a "
                    "routing classifier or that anything is out of scope."
                ),
            },
        ]
        return chat(messages)

    def _handle_conversation_meta(
        self, state: ConversationState, message: str
    ) -> str:
        """Answer questions about the conversation itself."""
        prior = [m for m in state.history if m["role"] in ("user", "assistant")]
        # Drop the trailing user message (the current one).
        if prior and prior[-1]["role"] == "user":
            prior = prior[:-1]

        if not prior:
            return (
                "We haven't talked about anything else yet — this is your first message."
            )

        history_txt = "\n".join(f"{m['role']}: {m['content']}" for m in prior)
        messages = [
            {"role": "system", "content": prompts.ANSWER_SYSTEM},
            {
                "role": "user",
                "content": (
                    "Here is the conversation so far (oldest first):\n"
                    f"{history_txt}\n\n"
                    f"User asks: {message}\n\n"
                    "Answer the user's meta-question about the conversation using ONLY "
                    "the transcript above. If the answer isn't in the transcript, say so."
                ),
            },
        ]
        return chat(messages)

    def _handle_weather(self, state: ConversationState) -> str:
        current = get_current_weather()
        hourly = get_hourly_weather()
        # Summarize the next 6 hours deterministically, then let the LLM phrase it.
        now_hour = datetime.now().hour
        upcoming = [
            hourly[h] for h in range(now_hour, min(now_hour + settings.weather_forecast_hours, 24)) if h in hourly
        ]
        summary_lines = [
            f"- {w.timestamp:%H:%M}: {w.condition}, {w.temperature_c:.0f}°C, "
            f"rain {w.precipitation_probability}%"
            for w in upcoming
        ]
        context = (
            f"Current: {current.condition}, {current.temperature_c:.0f}°C, "
            f"rain probability {current.precipitation_probability}%.\n"
            "Next hours:\n" + "\n".join(summary_lines)
        )
        messages = [
            {"role": "system", "content": prompts.ANSWER_SYSTEM},
            {
                "role": "user",
                "content": (
                    "LIVE DATA:\n"
                    f"{context}\n\n"
                    "User asked about the weather. Summarize concisely and, if rain is "
                    "likely soon, suggest indoor alternatives."
                ),
            },
        ]
        return chat(messages)

    def _handle_factual(self, state: ConversationState, message: str) -> str:
        hits = self.retriever.search(message, top_k=settings.rag_top_k_factual)

        logger.info("RAG factual query=%r top_k=%d", message, settings.rag_top_k_factual)
        for i, h in enumerate(hits, 1):
            logger.info("  %d. %s (score=%.3f) chunk=%r",
                        i, h.attraction_name, h.score, h.text[:120])

        blocks = []
        for h in hits:
            att = self.retriever.get_attraction(h.attraction_id)
            if att is None:
                continue
            hours = "; ".join(f"{k}: {v}" for k, v in att.opening_hours.items())
            blocks.append(
                f"[{att.name}]\n"
                f"Description: {att.description}\n"
                f"Categories: {', '.join(c.value for c in att.categories)}\n"
                f"Indoor: {'yes' if att.indoor else 'no'}\n"
                f"Typical duration: {att.typical_duration_min} min\n"
                f"Opening hours: {hours}\n"
                f"Coordinates: {att.lat}, {att.lon}\n"
                + (f"Source: {att.source_url}" if att.source_url else "Source: (none)")
            )
        context = "\n\n".join(blocks) or "(no relevant context found)"

        messages = [
            {"role": "system", "content": prompts.ANSWER_SYSTEM},
            {
                "role": "user",
                "content": (
                    f"RETRIEVED CONTEXT:\n{context}\n\n"
                    f"User question: {message}\n\n"
                    "Answer only from the context. Use the Opening hours field for "
                    "any hours question. If the context doesn't cover it, say so."
                ),
            },
        ]
        return chat(messages)

    def _handle_recommendation(self, state: ConversationState, message: str) -> str:
        # Merge any new preferences the user just expressed.
        self._update_preferences(state, message)

        query = self._build_pref_query(message, state.preferences)
        ranked = self.retriever.search_attractions(query, top_k=settings.rag_top_k_recommendation)

        logger.info("RAG recommendation query=%r", query)

        if not ranked:
            return "I couldn't find attractions matching that. Want me to suggest a general first-time visit?"

        for i, (att, score) in enumerate(ranked, 1):
            logger.info("  %d. %s (score=%.3f) categories=%s", i, att.name, score, [c.value for c in att.categories])


        context = "\n\n".join(
            f"[{a.name}] {a.description} "
            f"(categories: {', '.join(c.value for c in a.categories)}; "
            f"{'indoor' if a.indoor else 'outdoor'})"
            + (f" (source: {a.source_url})" if a.source_url else "")
            for a, _ in ranked
        )
        messages = [
            {"role": "system", "content": prompts.ANSWER_SYSTEM},
            {
                "role": "user",
                "content": (
                    f"RETRIEVED CONTEXT:\n{context}\n\n"
                    f"User: {message}\n\n"
                    "Give a short, friendly recommendation using only these places. "
                    "Mention 2-4 places and why they fit."
                ),
            },
        ]
        return chat(messages)

    def _handle_plan_request(self, state, message, router):
        self._update_preferences(state, message)

        hours = router.time_window_hours or settings.default_plan_hours
        start = self._resolve_start_time(router)
        end = start + timedelta(hours=hours)

        # --- 1. Retrieve candidates via RAG ---
        query = self._build_pref_query(message, state.preferences, plan=True)
        ranked = self.retriever.search_attractions(query, top_k=settings.rag_top_k_plan)
        candidates = [a for a, _ in ranked]

        logger.info("RAG plan query=%r", query)
        for i, (att, score) in enumerate(ranked, 1):
            logger.info(
                "  %d. %s (score=%.3f) categories=%s",
                i, att.name, score, [c.value for c in att.categories],
            )

        # --- 2. Filter out disliked categories ---
        if state.preferences.dislikes:
            before = len(candidates)
            removed = [
                a.name for a in candidates
                if any(c in state.preferences.dislikes for c in a.categories)
            ]
            candidates = [
                a for a in candidates
                if not any(c in state.preferences.dislikes for c in a.categories)
            ]
            logger.info(
                "Filtered disliked categories %s: %d -> %d candidates",
                [c.value for c in state.preferences.dislikes],
                before, len(candidates),
            )
            if removed:
                logger.info("  Removed: %s", removed)

        # --- 3. Top up from the full KB if filtering left too few ---
        if len(candidates) < 5:
            have = {a.id for a in candidates}
            for a in self.retriever.all_attractions():
                if a.id in have:
                    continue
                if any(c in state.preferences.dislikes for c in a.categories):
                    continue
                candidates.append(a)
                have.add(a.id)              # <-- important: update `have` as you append
                if len(candidates) >= 8:
                    break

        # --- 4. Fallback: if we still have nothing, ask the user to relax ---
        if not candidates:
            return (
                "I couldn't find attractions that fit your request while avoiding "
                f"{', '.join(c.value for c in state.preferences.dislikes)}. "
                "Want me to relax that constraint?"
            )

        # --- 5. Plan deterministically ---
        weather_by_hour = get_hourly_weather(start.date())

        logger.info("Final candidates for planner: %s", [a.name for a in candidates])
        itinerary = plan_day(
            candidates,
            start_time=start,
            end_time=end,
            prefs=state.preferences,
            weather_by_hour=weather_by_hour,
        )
        state.current_itinerary = itinerary
        logger.info("Plan: %s (start=%s)", summarize(itinerary), start.isoformat())
        return self._phrase_itinerary(itinerary, message)

    def _handle_modify(self, state: ConversationState, message: str) -> str:
        if state.current_itinerary is None:
            return (
                "I don't have an itinerary to modify yet. Tell me how many hours you have "
                "and I'll build one."
            )

        self._update_preferences(state, message)
        change = self._extract_change(message, state)

        logger.info(
            "Modify: action=%s target=%r value=%r reason=%r",
            change.action, change.target, change.value, change.reason,
        )

        it = state.current_itinerary
        notes: list[str] = []

        if change.action == "remove" and change.target:
            kept = [a for a in it.activities if change.target.lower() not in a.name.lower()]
            if len(kept) == len(it.activities):
                notes.append(f"I couldn't find '{change.target}' in the current plan.")
            it.activities = kept
        elif change.action == "replace" and change.target:
            kept = [a for a in it.activities if change.target.lower() not in a.name.lower()]
            replacement = self._find_replacement(state, target=change.target)
            if replacement is None:
                notes.append(f"No suitable replacement found for '{change.target}'.")
            else:
                new_it = build_itinerary(
                    [replacement],
                    start_time=it.start_time,
                    end_time=it.end_time,
                    prefs=state.preferences,
                )
                if new_it.activities:
                    it.activities = kept + new_it.activities
                else:
                    notes.append(
                        f"Replacement '{replacement.name}' didn't fit in the window."
                    )
                    it.activities = kept
        elif change.action == "add" and change.target:
            added = self._find_replacement(state, target=change.target)
            if added is None:
                notes.append(f"I couldn't find '{change.target}' in the knowledge base.")
            else:
                # Rebuild from scratch with the current plan's attractions + new one.
                current_ids = {a.attraction_id for a in it.activities}
                pool = [self.retriever.get_attraction(i) for i in current_ids]
                pool = [p for p in pool if p is not None] + [added]
                rebuilt = build_itinerary(
                    pool,
                    start_time=it.start_time,
                    end_time=it.end_time,
                    prefs=state.preferences,
                )
                it.activities = rebuilt.activities
                notes.extend(rebuilt.feasibility_notes)
        elif change.action == "change_pace" and change.value:
            try:
                state.preferences.pace = type(state.preferences.pace)(change.value)
            except ValueError:
                notes.append(f"Unknown pace '{change.value}'.")
            it = self._replan(state, notes)
        elif change.action == "change_transport" and change.value:
            try:
                state.preferences.transport_mode = type(state.preferences.transport_mode)(
                    change.value
                )
            except ValueError:
                notes.append(f"Unknown transport mode '{change.value}'.")
            it = self._replan(state, notes)
        elif change.action == "change_time":
            notes.append(
                "Time changes need a concrete window — tell me the new start and duration."
            )
        else:
            notes.append("I couldn't tell what to change. Try 'remove the museum' or 'make it shorter'.")

        it.feasibility_notes.extend(notes)
        state.current_itinerary = it

        logger.info(
            "Plan after modify: %s",
            [(a.name, a.start_time.strftime("%H:%M"), a.end_time.strftime("%H:%M")) for a in it.activities],
        )
        logger.info("Prefs now: pace=%s transport=%s", state.preferences.pace, state.preferences.transport_mode)        

        return self._phrase_itinerary(it, message)

    # ---------- helpers ----------

    def _default_start_time(self) -> datetime:
        """Default plan start: next round hour at least 1h from now."""
        now = datetime.now().replace(minute=0, second=0, microsecond=0)
        return now + timedelta(hours=settings.default_plan_start_offset_hours)

    def _build_pref_query(
        self, message: str, prefs: UserPreferences, plan: bool = False
    ) -> str:
        parts = [message]
        if prefs.interests:
            parts.append("interests: " + ", ".join(c.value for c in prefs.interests))
        if prefs.has_children:
            parts.append("family friendly")
        if plan:
            parts.append("attractions suitable for a short itinerary")
        return " | ".join(parts)

    def _update_preferences(self, state: ConversationState, message: str) -> None:
        messages = [
            {"role": "system", "content": prompts.PREFERENCE_SYSTEM},
            {"role": "user", "content": message},
        ]
        try:
            extracted = chat_structured(messages, PreferenceOutput)
        except LLMError:
            logger.warning("Preference extraction failed; keeping existing preferences.")
            return

        prefs = state.preferences
        for cat in extracted.interests:
            if cat not in prefs.interests:
                prefs.interests.append(cat)
        for cat in extracted.dislikes:
            if cat not in prefs.dislikes:
                prefs.dislikes.append(cat)
            if cat in prefs.interests:
                prefs.interests.remove(cat)
        if extracted.pace is not None:
            prefs.pace = extracted.pace
        if extracted.transport_mode is not None:
            prefs.transport_mode = extracted.transport_mode
        if extracted.has_children is not None:
            prefs.has_children = extracted.has_children
        if extracted.has_car is not None:
            prefs.has_car = extracted.has_car

    def _extract_change(
        self, message: str, state: ConversationState
    ) -> ModifyOutput:
        current = state.current_itinerary
        plan_txt = (
            "\n".join(
                f"- {a.name} ({a.start_time:%H:%M}-{a.end_time:%H:%M})"
                for a in current.activities
            )
            if current
            else "(no current itinerary)"
        )
        messages = [
            {"role": "system", "content": prompts.MODIFY_SYSTEM},
            {
                "role": "user",
                "content": f"Current plan:\n{plan_txt}\n\nUser request: {message}",
            },
        ]
        try:
            return chat_structured(messages, ModifyOutput)
        except LLMError:
            logger.warning("Modify extraction failed.")
            return ModifyOutput(action="none")

    def _find_replacement(self, state: ConversationState, target: str):
        """Find an attraction matching the target name/category, respecting dislikes."""
        query = target
        # Try exact name match first.
        for a in self.retriever.all_attractions():
            if target.lower() in a.name.lower():
                logger.info("RAG replacement exact match: %s", a.name)
                return a
        # Fall back to semantic search filtered by dislikes.
        for a, score in self.retriever.search_attractions(query, top_k=settings.rag_top_k_replacement):
            if any(c in state.preferences.dislikes for c in a.categories):
                continue
            logger.info("RAG replacement semantic match: %s (score=%.3f)", a.name, score)
            return a
        logger.info("RAG replacement: no match for %r", target)
        return None

    def _replan(self, state: ConversationState, notes: list[str]) -> Itinerary:
        it = state.current_itinerary
        assert it is not None
        current_ids = [a.attraction_id for a in it.activities]
        pool = [self.retriever.get_attraction(i) for i in current_ids]
        pool = [p for p in pool if p is not None]

        logger.info("RAG replan pool: %s", [p.name for p in pool])

        weather = get_hourly_weather(it.start_time.date())
        rebuilt = plan_day(
            pool,
            start_time=it.start_time,
            end_time=it.end_time,
            prefs=state.preferences,
            weather_by_hour=weather,
        )
        notes.extend(rebuilt.feasibility_notes)
        return rebuilt

    def _phrase_itinerary(self, itinerary: Itinerary, user_message: str) -> str:
        lines = []
        for a in itinerary.activities:
            lines.append(
                f"- {a.start_time:%H:%M}-{a.end_time:%H:%M}  {a.name} "
                f"(travel {a.travel_time_before_min} min, "
                f"{'indoor' if a.indoor else 'outdoor'})"
            )
        plan_txt = "\n".join(lines) or "(no activities scheduled)"
        notes_txt = (
            "\n".join(f"- {n}" for n in itinerary.feasibility_notes)
            or "(none)"
        )
        messages = [
            {"role": "system", "content": prompts.ITINERARY_SYSTEM},
            {
                "role": "user",
                "content": (
                    f"Date of plan: {itinerary.start_time:%A, %d %B %Y}\n"
                    f"Window: {itinerary.start_time:%H:%M}–{itinerary.end_time:%H:%M}\n\n"
                    f"Validated plan:\n{plan_txt}\n\n"
                    f"Feasibility notes:\n{notes_txt}\n\n"
                    f"Original user request: {user_message}\n\n"
                    "Phrase this plan in friendly prose. Do not change the schedule. "
                    "Open by stating the date and the start time."
                ),
            },
        ]
        return chat(messages)

    def _resolve_start_time(self, router: RouterOutput) -> datetime:
        """Determine the plan start time."""
        if router.start_date:
            try:
                d = datetime.strptime(router.start_date, "%Y-%m-%d").date()
                return datetime.combine(
                    d,
                    datetime.min.time().replace(hour=settings.default_plan_start_hour),
                )
            except ValueError:
                logger.warning(
                    "Router returned unparseable start_date=%r; falling back to default.",
                    router.start_date,
                )
        return self._default_start_time()

    def _is_supported_destination(self, destination: str | None) -> bool:
        if not destination:
            return True
        d = destination.strip().lower()
        return any(s in d for s in SUPPORTED_DESTINATIONS)

_manager: ConversationManager | None = None


def get_manager() -> ConversationManager:
    global _manager
    if _manager is None:
        _manager = ConversationManager()
    return _manager