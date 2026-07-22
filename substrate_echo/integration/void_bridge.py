"""Void Bridge — connects to void_svt.py VacuumDefect, VoidFlora, VoidBiosphere.

Maps void objects to Substrate_Echo WorldObjects and extracts
relational state from defect properties.
"""

from __future__ import annotations
from typing import Optional
import numpy as np


class VoidBridge:
    """Bridge between Substrate_Echo and SVT/Void infrastructure."""
    
    @staticmethod
    def defect_to_world_object(defect) -> Optional[dict]:
        """Convert a VacuumDefect to a WorldObject-compatible dict."""
        if not hasattr(defect, 'defect_id'):
            return None
        
        return {
            "object_id": f"defect_{defect.defect_id}",
            "name": f"VacuumDefect({defect.defect_type})",
            "object_type": "vacuum_defect",
            "position": defect.position,
            "metadata": {
                "defect_type": defect.defect_type,
                "linear_mass_density": defect.linear_mass_density,
                "phase_gradient_theta": defect.phase_gradient_theta,
                "phase_gradient_phi": defect.phase_gradient_phi,
                "trapped_baryonic_mass": defect.trapped_baryonic_mass,
                "phonon_flux": defect.phonon_flux,
                "active": defect.active,
            },
            "relational": {
                "familiarity": min(1.0, defect.trapped_baryonic_mass / 10.0),
                "importance": min(1.0, defect.linear_mass_density / 5.0),
            },
        }
    
    @staticmethod
    def flora_to_world_object(flora) -> Optional[dict]:
        """Convert a VoidFlora to a WorldObject-compatible dict."""
        if not hasattr(flora, 'flora_id'):
            return None
        
        return {
            "object_id": f"flora_{flora.flora_id}",
            "name": f"VoidFlora({flora.composition})",
            "object_type": "void_flora",
            "metadata": {
                "defect_id": flora.defect_id,
                "lattice_extent_km": flora.lattice_extent_km,
                "photon_capture_rate": flora.photon_capture_rate,
                "structural_integrity": flora.structural_integrity,
                "composition": flora.composition,
            },
        }
    
    @staticmethod
    def biosphere_to_spatial_objects(biosphere) -> list[dict]:
        """Convert all objects in a VoidBiosphere to WorldObject dicts."""
        objects = []
        
        for defect in biosphere.defects.values():
            obj = VoidBridge.defect_to_world_object(defect)
            if obj:
                objects.append(obj)
        
        for flora in biosphere.flora.values():
            obj = VoidBridge.flora_to_world_object(flora)
            if obj:
                objects.append(obj)
        
        return objects
    
    @staticmethod
    def defect_to_psv(defect) -> np.ndarray:
        """Extract a 16D state vector from a VacuumDefect.
        
    Maps defect properties to pillar-relevant dimensions:
    - Position → spatial pillars (Awareness, Presence, Relation)
    - Mass/energy → force pillars (Force, Resistance, Integrity)
    - Phase → cognitive pillars (Flux, Distortion, Depth)
    """
        pos = np.array(defect.position, dtype=np.float64)
        
        state = np.zeros(16, dtype=np.float64)
        
        # Spatial mapping
        state[0] = np.clip(np.linalg.norm(pos) / 10.0, 0, 1)  # Awareness (distance)
        state[8] = 1.0 - state[0]  # Presence (closeness)
        state[7] = min(1.0, defect.trapped_baryonic_mass / 5.0)  # Relation (mass)
        
        # Force mapping
        state[2] = min(1.0, defect.linear_mass_density / 3.0)  # Force
        state[4] = min(1.0, defect.phonon_flux / 10.0)  # Resistance (phonon flux)
        state[5] = min(1.0, defect.trapped_baryonic_mass / 10.0)  # Integrity
        
        # Phase mapping
        state[14] = (defect.phase_gradient_theta / np.pi) if defect.phase_gradient_theta else 0.5  # Flux
        state[13] = (defect.phase_gradient_phi / (2 * np.pi)) if defect.phase_gradient_phi else 0.5  # Distortion
        state[15] = min(1.0, defect.linear_mass_density / 5.0)  # Depth
        
        return np.clip(state, 0.0, 1.0)
