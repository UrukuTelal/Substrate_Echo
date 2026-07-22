"""Engine Bridge — connects to Van_Nueman_Engine C++ infrastructure.

Maps engine entities to Substrate_Echo format and provides
synchronization with the engine tick loop.
"""

from __future__ import annotations
from typing import Optional
import numpy as np


class EngineBridge:
    """Bridge between Substrate_Echo and the Van_Nueman Engine."""
    
    @staticmethod
    def engine_entity_to_dict(entity_data: dict) -> dict:
        """Convert engine entity data to Substrate_Echo WorldObject format."""
        psv = entity_data.get("psv", [0.0] * 16)
        
        return {
            "object_id": entity_data.get("uid", "unknown"),
            "name": entity_data.get("name", "entity"),
            "object_type": "engine_entity",
            "psv": np.array(psv, dtype=np.float64),
            "position": entity_data.get("position", [0, 0, 0]),
            "metadata": {
                "entity_type": entity_data.get("entity_type", "generic"),
                "mass": entity_data.get("mass", 1.0),
                "charge": entity_data.get("charge", 0.0),
            },
        }
    
    @staticmethod
    def psv_to_cognitive_state(psv: list[float]) -> np.ndarray:
        """Convert engine PSV to cognitive state vector."""
        return np.array(psv, dtype=np.float64)
    
    @staticmethod
    def thought_palace_realm_to_state(realm_id: int) -> np.ndarray:
        """Map ThoughtPalace realm to pillar state bias.
        
        Realms:
        0 = Conscious (balanced)
        1 = Dream (high Distortion, Flux)
        2 = Bicameral (high Willpower, Force)
        3 = Meta-Cognitive (high Awareness, Memory, Depth)
        4 = Paradigmatic (high Integrity, Cohesion)
        """
        realm_profiles = {
            0: {0: 0.7, 8: 0.7, 5: 0.6},  # Conscious
            1: {13: 0.8, 14: 0.8, 15: 0.6},  # Dream
            2: {1: 0.8, 2: 0.8, 3: 0.6},  # Bicameral
            3: {0: 0.8, 10: 0.8, 15: 0.8},  # Meta-Cognitive
            4: {5: 0.8, 6: 0.8, 4: 0.6},  # Paradigmatic
        }
        
        profile = realm_profiles.get(realm_id, {})
        state = np.full(16, 0.5)
        for idx, val in profile.items():
            state[idx] = val
        
        return state
