from __future__ import annotations

import uuid

from tourist_assistant.models import ConversationState


class SessionStore:
    """In-memory session store. Replace with Redis/DB for production."""

    def __init__(self) -> None:
        self._sessions: dict[str, ConversationState] = {}

    def create(self) -> ConversationState:
        sid = str(uuid.uuid4())
        state = ConversationState(session_id=sid)
        self._sessions[sid] = state
        return state

    def get(self, session_id: str) -> ConversationState | None:
        return self._sessions.get(session_id)

    def get_or_create(self, session_id: str | None) -> ConversationState:
        if session_id and session_id in self._sessions:
            return self._sessions[session_id]
        return self.create()

    def save(self, state: ConversationState) -> None:
        self._sessions[state.session_id] = state


_store = SessionStore()


def get_session_store() -> SessionStore:
    return _store