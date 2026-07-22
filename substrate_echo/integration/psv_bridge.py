"""PSV Bridge — connects Substrate_Echo to BlochPSV, PillarState, and PillarVector16.

Three storage conventions exist in the ecosystem:
1. BlochPSV (DeveloperConsole): theta/phi angles, Z-projection = (cos(theta)+1)/2
2. PillarState (VNES-Lab): theta/phi angles, direct theta access
3. PillarVector16 (C++ Screensaver): raw scalar values [0,1]

Substrate_Echo uses raw scalars [0,1] internally. This bridge converts
to/from the other formats.
"""

from __future__ import annotations
from typing import Optional, Any
import numpy as np
import math


PILLAR_NAMES = [
    "Awareness", "Willpower", "Force", "Influence",
    "Resistance", "Integrity", "Cohesion", "Relation",
    "Presence", "Warmth", "Memory", "Attraction",
    "Harm", "Distortion", "Flux", "Depth",
]

NUM_PILLARS = 16


class PSVBridge:
    """Bridge between Substrate_Echo and existing PSV infrastructure."""
    
    # ── BlochPSV ↔ numpy ──────────────────────────────────────────
    
    @staticmethod
    def blochpsv_to_array(bloch_psv) -> np.ndarray:
        """Convert BlochPSV to numpy array of Z-projections [0,1]."""
        return np.array(bloch_psv.to_scalar_array(), dtype=np.float64)
    
    @staticmethod
    def array_to_blochpsv(arr: np.ndarray):
        """Convert numpy array [0,1] to BlochPSV."""
        # Lazy import to avoid circular deps
        import sys, os
        dev_console = os.path.join(os.path.dirname(__file__), "..", "..", "..", "DeveloperConsole", "backend")
        sys.path.insert(0, dev_console)
        from models.bloch_psv import BlochPSV
        return BlochPSV(arr.tolist())
    
    @staticmethod
    def blochpsv_theta_phi_to_array(bloch_psv) -> np.ndarray:
        """Convert BlochPSV theta values directly (for VNES-Lab compatibility)."""
        return np.array([bloch_psv.get_theta(i) for i in range(NUM_PILLARS)], dtype=np.float64)
    
    # ── PillarState (VNES-Lab) ↔ numpy ────────────────────────────
    
    @staticmethod
    def pillarstate_to_array(pillar_state) -> np.ndarray:
        """Convert VNES-Lab PillarState theta to numpy array [0,1].
        
        PillarState stores raw theta in [0, PI]. We normalize to [0,1]
        for Substrate_Echo compatibility.
        """
        return np.array(pillar_state.theta, dtype=np.float64) / math.pi
    
    @staticmethod
    def array_to_pillarstate(arr: np.ndarray):
        """Convert numpy array [0,1] to VNES-Lab PillarState."""
        import sys, os
        vnes_lab = os.path.join(os.path.dirname(__file__), "..", "..", "..", "VNES-Lab", "lib")
        sys.path.insert(0, vnes_lab)
        from psv_core import PillarState
        theta = (arr * math.pi).tolist()
        return PillarState(theta=theta)
    
    # ── Entity (DeveloperConsole) ↔ numpy ──────────────────────────
    
    @staticmethod
    def entity_to_array(entity) -> np.ndarray:
        """Convert DeveloperConsole Entity PSV to numpy array [0,1].
        
        Entity uses old-style PSV (not BlochPSV). PSV.values are already [0,1].
        """
        if hasattr(entity, 'psv') and hasattr(entity.psv, 'values'):
            return np.array(entity.psv.values, dtype=np.float64)
        elif hasattr(entity, 'psv') and hasattr(entity.psv, 'to_scalar_array'):
            return np.array(entity.psv.to_scalar_array(), dtype=np.float64)
        return np.zeros(NUM_PILLARS)
    
    # ── UniverseEntity (MultiverseScreensaver) ↔ numpy ─────────────
    
    @staticmethod
    def universe_entity_to_array(universe_entity) -> np.ndarray:
        """Convert UniverseEntity PillarVector16 to numpy array.
        
        PillarVector16 stores raw scalars [0,1] — direct compatibility.
        """
        pillar = universe_entity.getPillarState()
        return np.array([pillar[i] for i in range(NUM_PILLARS)], dtype=np.float64)
    
    @staticmethod
    def array_to_pillar_vector16(arr: np.ndarray):
        """Convert numpy array to C++ PillarVector16 (via ctypes or manual)."""
        # In Python, we just return the array — the C++ side reads it
        return arr.tolist()
    
    # ── Similarity & Metrics ───────────────────────────────────────
    
    @staticmethod
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Cosine similarity between two state vectors."""
        n1 = np.linalg.norm(a)
        n2 = np.linalg.norm(b)
        if n1 < 1e-12 or n2 < 1e-12:
            return 0.0
        return float(np.dot(a, b) / (n1 * n2))
    
    @staticmethod
    def coherence(state: np.ndarray) -> float:
        """How internally aligned a state is (0-1)."""
        return float(1.0 - np.std(state))
    
    @staticmethod
    def dominant_pillar(state: np.ndarray) -> int:
        """Index of the highest pillar value."""
        return int(np.argmax(state))
    
    @staticmethod
    def weakest_pillar(state: np.ndarray) -> int:
        """Index of the lowest pillar value."""
        return int(np.argmin(state))
    
    @staticmethod
    def pillar_summary(state: np.ndarray) -> dict[str, float]:
        """Named dictionary of pillar values."""
        return {PILLAR_NAMES[i]: round(float(state[i]), 4) for i in range(NUM_PILLARS)}
