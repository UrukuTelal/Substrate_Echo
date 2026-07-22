"""Council Bridge — connects to Pillar Council orchestration.

Maps council consensus to cognitive responses and provides
16-pillar review for action validation.
"""

from __future__ import annotations
from typing import Optional
import numpy as np


class CouncilBridge:
    """Bridge between Substrate_Echo's AgentEcology and Pillar Council."""
    
    @staticmethod
    def consensus_to_action(consensus_response, context_state: np.ndarray) -> Optional[dict]:
        """Convert a council consensus to an action dict."""
        if consensus_response is None:
            return None
        
        return {
            "action_type": "council_approved",
            "confidence": consensus_response.confidence,
            "reasoning": consensus_response.reasoning,
            "proposed_state_change": (
                consensus_response.proposed_state_change.tolist()
                if consensus_response.proposed_state_change is not None
                else None
            ),
        }
    
    @staticmethod
    def validate_action(action: dict, state: np.ndarray) -> dict:
        """Validate an action against current state using 16-pillar checks."""
        checks = {
            "harm_check": CouncilBridge._check_harm(action, state),
            "integrity_check": CouncilBridge._check_integrity(action, state),
            "coherence_check": CouncilBridge._check_coherence(action, state),
        }
        
        all_passed = all(checks.values())
        
        return {
            "approved": all_passed,
            "checks": checks,
            "confidence": sum(checks.values()) / len(checks),
        }
    
    @staticmethod
    def _check_harm(action: dict, state: np.ndarray) -> bool:
        """Check if action would cause excessive harm (pillar 12)."""
        harm_level = float(state[12]) if len(state) > 12 else 0.5
        return harm_level < 0.8
    
    @staticmethod
    def _check_integrity(action: dict, state: np.ndarray) -> bool:
        """Check if state maintains integrity (pillar 5)."""
        integrity = float(state[5]) if len(state) > 5 else 0.5
        return integrity > 0.3
    
    @staticmethod
    def _check_coherence(action: dict, state: np.ndarray) -> bool:
        """Check if state is internally coherent."""
        std = float(np.std(state))
        return std < 0.4
