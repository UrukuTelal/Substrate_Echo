"""SC2 Adapter — Integrates SC2 bot with Substrate Kernel.

This module provides the bridge between:
- SC2 game observations → Kernel observations
- Kernel decisions → SC2 game actions
"""
from __future__ import annotations
from typing import Dict, Any, Optional, List, Tuple
import numpy as np

from substrate_echo.embodiments.sc2.observation_encoder import SC2ObservationEncoder
from substrate_echo.embodiments.sc2.action_decoder import SC2ActionDecoder, AbstractAction, ActionType


class SC2Adapter:
    """Adapter between SC2 game and Substrate Kernel.

    Translates:
    - Game state → 16D observation vector
    - Kernel decisions → Game actions
    """

    def __init__(self):
        self.encoder = SC2ObservationEncoder()
        self.decoder = SC2ActionDecoder()
        self._step = 0
        self._last_action = None

    def process_observation(self, game_state: Any) -> np.ndarray:
        """Process game observation into kernel-compatible vector."""
        self._step += 1
        return self.encoder.encode(game_state)

    def process_action(self, kernel_decision: Dict[str, Any]) -> AbstractAction:
        """Convert kernel decision to abstract action."""
        action_type_str = kernel_decision.get("type", "hold")
        try:
            action_type = ActionType(action_type_str)
        except ValueError:
            action_type = ActionType.HOLD

        abstract = AbstractAction(
            action_type=action_type,
            target=kernel_decision.get("target"),
            description=kernel_decision.get("description", ""),
            confidence=kernel_decision.get("confidence", 0.5),
            urgency=kernel_decision.get("urgency", 0.5),
        )

        self._last_action = abstract
        return abstract

    def get_concrete_actions(self, abstract: AbstractAction,
                             game_state: Any = None) -> List:
        """Convert abstract action to concrete game actions."""
        return self.decoder.decode(abstract, game_state)

    def get_status(self) -> Dict[str, Any]:
        """Get adapter status."""
        return {
            "step": self._step,
            "last_action": self._last_action.action_type.value if self._last_action else None,
            "encoder_history": len(self.encoder._history),
        }
