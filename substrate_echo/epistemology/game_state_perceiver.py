"""GameStatePerceiver — Reads text feedback from SC2.

SC2 provides three channels of text/signal feedback:
1. Chat messages (self.state.chat) — other players/bots
2. Game alerts (self.state.alerts) — structured events like TrainError, UnitUnderAttack
3. Action errors (self.state.action_errors) — feedback on failed commands

This module converts all three into perceived text events that the
cognitive system can reason about. Maps SC2 alerts to human-readable
text projections like "Supply Limit Reached — Build More Supply Depots".
"""
from __future__ import annotations
import enum
from dataclasses import dataclass, field
from typing import List, Dict, Optional


class PerceivedEventType(enum.Enum):
    CHAT = "chat"
    ALERT = "alert"
    ACTION_ERROR = "action_error"
    SCREEN_TEXT = "screen_text"


# Map SC2 Alert codes to human-readable text projections
# These are the text strings SC2 displays on-screen
ALERT_TEXT = {
    3:  "Error occurred",
    4:  "Add-on complete",
    5:  "Building complete",
    6:  "Base is under attack",
    7:  "Larva hatched",
    8:  "Unit merge complete",
    9:  "Minerals exhausted — Expand or harvest more",
    10: "Unit morph complete",
    11: "Mothership summoned",
    12: "MULE expired",
    13: "Nuclear launch detected",
    14: "Nuclear launch detected — Find the silo",
    15: "Supply limit reached — Build more supply structures",
    16: "Training complete",
    17: "Worker training complete",
    18: "Unit transformation complete",
    19: "Unit is under attack",
    20: "Upgrade complete",
    21: "Vespene exhausted",
    22: "Warp-in complete",
}

# Map action error codes to text
ACTION_ERROR_TEXT = {
    0: "Success",
    1: "Not supported",
    2: "Failed — queue is full",
    3: "Invalid — unit cannot perform this action",
    4: "Wireframe error",
    5: "Action error",
}


@dataclass
class PerceivedEvent:
    """A single piece of text feedback from the game."""
    event_type: PerceivedEventType
    text: str
    source_id: int = 0  # player_id for chat, alert code for alerts
    tick: int = 0
    metadata: Dict = field(default_factory=dict)

    def __repr__(self):
        return f"[{self.event_type.value}] {self.text}"


class GameStatePerceiver:
    """Reads text feedback from SC2 game state each tick.

    Produces PerceivedEvent objects from chat, alerts, and action errors.
    Tracks state changes to detect new events (not repeated ones).
    """

    def __init__(self):
        self._seen_alerts: set = set()
        self._seen_chats: list = []
        self._event_log: List[PerceivedEvent] = []

    def perceive(self, game_state, tick: int = 0) -> List[PerceivedEvent]:
        """Read all text channels from SC2 game state.

        Args:
            game_state: self.state from BotAI (has .chat, .alerts, .action_errors)
            tick: current game tick

        Returns:
            List of new PerceivedEvent objects this tick
        """
        events = []

        events.extend(self._read_chat(game_state, tick))
        events.extend(self._read_alerts(game_state, tick))
        events.extend(self._read_action_errors(game_state, tick))

        self._event_log.extend(events)
        return events

    def _read_chat(self, game_state, tick: int) -> List[PerceivedEvent]:
        """Read chat messages from the game."""
        events = []
        for msg in game_state.chat:
            try:
                text = msg.message
                player_id = msg.player_id
                channel = msg.channel
            except AttributeError:
                continue
            event = PerceivedEvent(
                event_type=PerceivedEventType.CHAT,
                text=text,
                source_id=player_id,
                tick=tick,
                metadata={"channel": channel},
            )
            events.append(event)
        return events

    def _read_alerts(self, game_state, tick: int) -> List[PerceivedEvent]:
        """Read game alerts — these map to on-screen text projections."""
        events = []
        for alert_code in game_state.alerts:
            if alert_code in self._seen_alerts:
                continue
            self._seen_alerts.add(alert_code)
            text = ALERT_TEXT.get(alert_code, f"Unknown alert code {alert_code}")
            event = PerceivedEvent(
                event_type=PerceivedEventType.ALERT,
                text=text,
                source_id=alert_code,
                tick=tick,
            )
            events.append(event)
        return events

    def _read_action_errors(self, game_state, tick: int) -> List[PerceivedEvent]:
        """Read action errors — feedback on failed commands."""
        events = []
        for error in game_state.action_errors:
            try:
                result = error.result
                ability_id = error.ability_id
                unit_tag = error.unit_tag
            except AttributeError:
                continue
            text = ACTION_ERROR_TEXT.get(result, f"Action error code {result}")
            event = PerceivedEvent(
                event_type=PerceivedEventType.ACTION_ERROR,
                text=text,
                source_id=result,
                tick=tick,
                metadata={"ability_id": ability_id, "unit_tag": unit_tag},
            )
            events.append(event)
        return events

    def clear_alert_cache(self):
        """Reset alert tracking so alerts can fire again."""
        self._seen_alerts.clear()

    def get_event_log(self) -> List[PerceivedEvent]:
        """Return the full event log."""
        return list(self._event_log)

    def get_recent_events(self, n: int = 10) -> List[PerceivedEvent]:
        """Return the N most recent events."""
        return self._event_log[-n:]

    def summary(self) -> Dict[str, int]:
        """Count events by type."""
        counts = {}
        for e in self._event_log:
            counts[e.event_type.value] = counts.get(e.event_type.value, 0) + 1
        return counts
