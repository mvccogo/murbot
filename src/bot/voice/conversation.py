"""Short, speaker-scoped conversation focus following an addressed request."""
from dataclasses import dataclass


@dataclass
class ConversationFocus:
    user_id: int | None = None
    last_addressed_at: float = 0.0

    def accept(self, user_id: int, now: float):
        self.user_id = user_id
        self.last_addressed_at = now

    def clear(self):
        self.user_id = None
        self.last_addressed_at = 0.0

    def permits(self, user_id: int, started_at: float, window_seconds: float,
                playback_active: bool = False, playback_finished_at: float = 0.0) -> bool:
        if window_seconds <= 0 or self.user_id != user_id:
            return False
        if playback_active:
            return True
        deadline = max(self.last_addressed_at, playback_finished_at) + window_seconds
        return started_at <= deadline
