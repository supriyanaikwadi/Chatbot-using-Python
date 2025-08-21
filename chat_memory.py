"""
chat_memory.py
---------------
Sliding window conversation memory for short-term context.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class ChatMemory:
    max_turns: int = 4  # number of (user, bot) turns to keep
    history: List[Tuple[str, str]] = field(default_factory=list)

    def add_turn(self, role: str, text: str) -> None:
        """Add a turn (either user or bot)."""
        if role.lower() == "user":
            self.history.append((text.strip(), ""))  # user text, bot blank for now
        elif role.lower() == "bot":
            if self.history:
                u, _ = self.history[-1]
                self.history[-1] = (u, text.strip())  # update last with bot reply
            else:
                self.history.append(("", text.strip()))
        # Trim history
        if len(self.history) > self.max_turns:
            self.history = self.history[-self.max_turns:]

    def clear(self) -> None:
        """Clear all memory."""
        self.history.clear()

    def render(self) -> str:
        """Render history as transcript."""
        lines = []
        for u, b in self.history:
            if u:
                lines.append(f"User: {u}")
            if b:
                lines.append(f"Bot: {b}")
        return "\n".join(lines)

    def format_buffer(self) -> str:
        """Alias for render(), since interface.py expects this method."""
        return self.render()
