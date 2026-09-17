from __future__ import annotations

import logging

from rich.console import Console
from rich.prompt import Prompt

from tourist_assistant.conversation.manager import get_manager
from tourist_assistant.security.prompt_guard import load_prompt_guard

console = Console()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    manager = get_manager()
    load_prompt_guard()

    console.print("[bold green]AI Tourist Assistant — Alexandroupolis[/bold green]")
    console.print("Type your question, or 'exit' to quit.\n")

    session_id: str | None = None
    while True:
        try:
            user_input = Prompt.ask("[bold cyan]You[/bold cyan]")
        except (EOFError, KeyboardInterrupt):
            break

        if user_input.strip().lower() in {"exit", "quit"}:
            break
        if not user_input.strip():
            continue

        result = manager.handle(user_input, session_id=session_id)
        session_id = result["session_id"]
        console.print(f"[bold magenta]Assistant[/bold magenta]: {result['reply']}\n")


if __name__ == "__main__":
    main()