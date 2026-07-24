"""Observation Layer — Separating telemetry from interpretation.

The observation layer creates explicit separation between raw sensor
data and interpreted features. This prevents premature commitment
to interpretations.

Architecture:
    Raw Observation
         |
    Feature Extraction
         |
    Feature Set (with confidence)
         |
    Interpretation Space
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable
from enum import Enum
import numpy as np
import time


class FeatureType(Enum):
    """Types of features that can be extracted."""
    SPATIAL = "spatial"           # Position, distance, proximity
    TEMPORAL = "temporal"         # Timing, duration, frequency
    RELATIONAL = "relational"     # Between entities
    BEHAVIORAL = "behavioral"     # Movement patterns, actions
    RESOURCE = "resource"         # Quantities, availability
    INTENTIONAL = "intentional"   # Inferred goals, purposes
    UNKNOWN = "unknown"


@dataclass
class RawObservation:
    """Unprocessed sensor data from the environment.
    
    This is what was actually observed, without interpretation.
    """
    data: Dict[str, Any]
    modality: str
    source: str
    timestamp: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_vector(self) -> np.ndarray:
        """Convert to numeric vector for kernel processing."""
        values = []
        for v in self.data.values():
            if isinstance(v, (int, float)):
                values.append(float(v))
            elif isinstance(v, bool):
                values.append(1.0 if v else 0.0)
        return np.array(values, dtype=np.float64) if values else np.zeros(1)


@dataclass
class Feature:
    """An extracted feature from raw observation.
    
    Features are objective measurements, not interpretations.
    """
    name: str
    value: Any
    feature_type: FeatureType
    confidence: float = 1.0  # How certain we are about this feature
    source: str = ""         # Which extraction method produced this
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_numeric(self) -> float:
        """Convert to numeric value for kernel processing."""
        if isinstance(self.value, (int, float)):
            return float(self.value)
        elif isinstance(self.value, bool):
            return 1.0 if self.value else 0.0
        elif isinstance(self.value, str):
            return hash(self.value) % 1000 / 1000.0
        return 0.0


@dataclass
class FeatureSet:
    """A collection of extracted features from a single observation.
    
    Represents the system's objective understanding of what was observed.
    """
    features: Dict[str, Feature] = field(default_factory=dict)
    raw_observation: Optional[RawObservation] = None
    timestamp: float = 0.0
    
    def add(self, name: str, value: Any, feature_type: FeatureType,
            confidence: float = 1.0, source: str = ""):
        """Add a feature to the set."""
        self.features[name] = Feature(
            name=name,
            value=value,
            feature_type=feature_type,
            confidence=confidence,
            source=source,
        )
    
    def get(self, name: str) -> Optional[Feature]:
        """Get a feature by name."""
        return self.features.get(name)
    
    def get_by_type(self, feature_type: FeatureType) -> List[Feature]:
        """Get all features of a specific type."""
        return [f for f in self.features.values() if f.feature_type == feature_type]
    
    def to_vector(self) -> np.ndarray:
        """Convert all features to numeric vector."""
        values = [f.to_numeric() for f in self.features.values()]
        return np.array(values, dtype=np.float64) if values else np.zeros(1)
    
    def get_confidence(self) -> float:
        """Get average confidence across all features."""
        if not self.features:
            return 0.0
        return np.mean([f.confidence for f in self.features.values()])
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            name: {
                "value": f.value,
                "type": f.feature_type.value,
                "confidence": f.confidence,
            }
            for name, f in self.features.items()
        }


class FeatureExtractor:
    """Extracts features from raw observations.
    
    This is where raw sensor data becomes objective measurements.
    The extractor does not interpret - it only measures.
    """
    
    def __init__(self):
        self._extractors: Dict[str, Callable] = {}
        self._extraction_count = 0
    
    def register_extractor(self, feature_name: str, extractor: Callable):
        """Register a feature extraction function.
        
        The extractor function should take raw data and return (value, confidence).
        """
        self._extractors[feature_name] = extractor
    
    def extract(self, raw: RawObservation) -> FeatureSet:
        """Extract features from raw observation."""
        features = FeatureSet(raw_observation=raw, timestamp=raw.timestamp)
        
        # Extract using registered extractors
        for name, extractor in self._extractors.items():
            try:
                value, confidence = extractor(raw.data)
                feature_type = self._infer_feature_type(name, value)
                features.add(name, value, feature_type, confidence, source=name)
            except Exception as e:
                # Extraction failed - record with low confidence
                features.add(name, None, FeatureType.UNKNOWN, 0.0, source=f"error:{e}")
        
        # If no extractors registered, extract all numeric values from raw data
        if not self._extractors:
            for key, value in raw.data.items():
                if isinstance(value, (int, float)):
                    feature_type = self._infer_feature_type(key, value)
                    features.add(key, value, feature_type, confidence=1.0, source="auto_extract")
        
        self._extraction_count += 1
        return features
    
    def _infer_feature_type(self, name: str, value: Any) -> FeatureType:
        """Infer feature type from name and value."""
        name_lower = name.lower()
        
        if any(kw in name_lower for kw in ["position", "distance", "proximity", "location"]):
            return FeatureType.SPATIAL
        elif any(kw in name_lower for kw in ["time", "duration", "frequency", "rate"]):
            return FeatureType.TEMPORAL
        elif any(kw in name_lower for kw in ["ally", "enemy", "relation", "between"]):
            return FeatureType.RELATIONAL
        elif any(kw in name_lower for kw in ["movement", "action", "behavior", "pattern"]):
            return FeatureType.BEHAVIORAL
        elif any(kw in name_lower for kw in ["resource", "mineral", "supply", "count"]):
            return FeatureType.RESOURCE
        elif any(kw in name_lower for kw in ["intent", "goal", "purpose", "plan"]):
            return FeatureType.INTENTIONAL
        
        return FeatureType.UNKNOWN


class ObservationMemory:
    """Memory of past observations and extracted features.
    
    Stores observation history for pattern detection and
    hypothesis generation.
    """
    
    def __init__(self, max_size: int = 1000):
        self._observations: List[FeatureSet] = []
        self._max_size = max_size
    
    def record(self, features: FeatureSet):
        """Record a feature set."""
        self._observations.append(features)
        if len(self._observations) > self._max_size:
            self._observations.pop(0)
    
    def get_recent(self, n: int = 10) -> List[FeatureSet]:
        """Get most recent n observations."""
        return self._observations[-n:]
    
    def get_by_feature(self, feature_name: str,
                       min_confidence: float = 0.5) -> List[FeatureSet]:
        """Get observations containing a specific feature."""
        return [
            obs for obs in self._observations
            if obs.get(feature_name) and obs.get(feature_name).confidence >= min_confidence
        ]
    
    def get_temporal_window(self, start_time: float,
                            end_time: float) -> List[FeatureSet]:
        """Get observations within a time window."""
        return [
            obs for obs in self._observations
            if start_time <= obs.timestamp <= end_time
        ]
    
    def compute_feature_stats(self, feature_name: str) -> Dict[str, float]:
        """Compute statistics for a feature over time."""
        values = []
        for obs in self._observations:
            feature = obs.get(feature_name)
            if feature and isinstance(feature.value, (int, float)):
                values.append(float(feature.value))
        
        if not values:
            return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "count": 0}
        
        return {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "count": len(values),
        }
    
    def detect_trend(self, feature_name: str,
                     window: int = 20) -> Optional[str]:
        """Detect if a feature is trending up, down, or stable."""
        values = []
        for obs in self._observations[-window:]:
            feature = obs.get(feature_name)
            if feature and isinstance(feature.value, (int, float)):
                values.append(float(feature.value))
        
        if len(values) < 5:
            return None
        
        # Simple linear regression
        x = np.arange(len(values))
        slope = np.polyfit(x, values, 1)[0]
        
        if slope > 0.01:
            return "increasing"
        elif slope < -0.01:
            return "decreasing"
        return "stable"