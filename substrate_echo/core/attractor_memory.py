"""Attractor Memory — memory as persistent structures in the ontological field.

Instead of:  Experience → Database Entry → Retrieval
We use:     Experience → State Transformation → Attractor Formation → Recall

Consolidation Pipeline:
1. Periodic sweep: strengthen frequently-accessed, decay rarely-accessed
2. Merging: similar attractors combine (cosine similarity > threshold)
3. Pruning: remove attractors below strength threshold
4. Identity: persistent attractor cluster = identity pattern
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Callable
import numpy as np
import time

from ..models.experience import Experience, ExperienceType
from ..models.memory_trace import MemoryTrace, TraceType
from .ontological_field import OntologicalField, Attractor


@dataclass
class ConsolidationConfig:
    """Configuration for memory consolidation."""
    merge_similarity_threshold: float = 0.9  # cosine similarity to merge
    prune_strength_threshold: float = 0.05  # minimum strength to keep
    max_memories: int = 1000  # maximum memory capacity
    consolidation_interval: float = 60.0  # seconds between consolidations
    decay_halflife: float = 3600.0  # seconds for strength to halve
    identity_min_cluster: int = 3  # minimum attractors for identity
    identity_similarity_threshold: float = 0.7  # similarity to include in identity


class AttractorMemory:
    """Memory implemented as stable attractors in the ontological field.
    
    Experiences are encoded into the field, forming attractors.
    Recall queries the field for nearest attractors to a cue.
    Consolidation maintains memory health through merging, decay, and pruning.
    """
    
    def __init__(self, field: OntologicalField,
                 config: Optional[ConsolidationConfig] = None,
                 transition_manager=None):
        self.field = field
        self.config = config or ConsolidationConfig()
        self.traces: dict[str, MemoryTrace] = {}
        self._transition_manager = transition_manager
        self._consolidation_interval = self.config.consolidation_interval
        self._last_consolidation = time.time()
        
        # Identity tracking
        self._identity_cache: Optional[np.ndarray] = None
        self._identity_dirty: bool = True
    
    def encode(self, experience: Experience) -> Optional[Attractor]:
        """Transform experience into field pattern, form attractor.
        
        Pipeline:
        1. Extract PSV pattern from experience
        2. Inject pattern into field
        3. Form attractor at pattern location
        4. Store memory trace with attractor
        """
        if experience.psv_snapshot is None:
            return None
        
        pattern = np.array(experience.psv_snapshot, dtype=np.float64)
        
        # Determine attractor strength based on experience properties
        strength = experience.importance
        if experience.experience_type == ExperienceType.SURPRISE:
            strength *= 1.5
        if experience.experience_type == ExperienceType.GOAL_ACHIEVED:
            strength *= 1.3
        if experience.experience_type == ExperienceType.GOAL_FAILED:
            strength *= 1.2
        strength = min(1.0, strength)
        
        # Form attractor in field
        label = f"exp_{experience.experience_id}"
        attractor = self.field.form_attractor(pattern, strength=strength, label=label)
        
        # Create memory trace
        trace = MemoryTrace(
            trace_id=experience.experience_id,
            trace_type=self._experience_to_trace_type(experience),
            attractor_center=pattern.tolist(),
            description=experience.description,
            object_ids=experience.object_ids,
            events=[experience.to_dict()],
            emotional_valence=experience.result_valence,
            importance=experience.importance,
            strength=strength,
        )
        self.traces[label] = trace
        self._identity_dirty = True
        
        # Record transition if manager available
        if self._transition_manager is not None:
            from ..dynamics.state_transitions import StateTransition, TransitionCause
            old_state = np.zeros(16)
            if len(self.traces) > 1:
                old_state = list(self.traces.values())[-2].get_center_array() if hasattr(list(self.traces.values())[-2], 'get_center_array') else np.zeros(16)
            self._transition_manager.record(StateTransition(
                source_state=old_state,
                target_state=pattern,
                cause=TransitionCause.MEMORY_UPDATE,
                energy_cost=strength * 0.01,
                information_delta=strength * 0.1,
                description=f"Encoded memory: {experience.description[:50]}",
            ))
        
        return attractor
    
    def recall(self, cue: np.ndarray, k: int = 5) -> list[MemoryTrace]:
        """Recall memories by finding nearest attractors to cue.
        
        1. Find k nearest attractors to the cue pattern
        2. Look up their memory traces
        3. Reinforce recalled memories
        4. Return traces sorted by relevance
        """
        nearest = self.field.find_nearest_attractors(cue, k=k)
        
        results = []
        for att in nearest:
            trace = self.traces.get(att.label)
            if trace and trace.is_viable():
                trace.recall()
                att.access()
                results.append(trace)
        
        return results
    
    def recall_by_association(self, object_id: str, k: int = 5) -> list[MemoryTrace]:
        """Recall memories associated with a specific object."""
        candidates = []
        for trace in self.traces.values():
            if object_id in trace.object_ids and trace.is_viable():
                candidates.append(trace)
        
        candidates.sort(key=lambda t: t.strength * t.importance, reverse=True)
        for t in candidates[:k]:
            t.recall()
        return candidates[:k]
    
    def get_recent(self, n: int = 10) -> list[MemoryTrace]:
        """Get the n most recently formed memories."""
        sorted_traces = sorted(
            self.traces.values(),
            key=lambda t: t.formed_at,
            reverse=True,
        )
        return [t for t in sorted_traces if t.is_viable()][:n]
    
    def consolidate(self, force: bool = False) -> dict:
        """Run full consolidation cycle.
        
        Returns stats about what happened.
        """
        now = time.time()
        if not force and (now - self._last_consolidation) < self._consolidation_interval:
            return {"skipped": True, "reason": "too soon"}
        
        self._last_consolidation = now
        stats = {"merged": 0, "pruned": 0, "decayed": 0}
        
        # Step 1: Decay all traces
        dt = self._consolidation_interval
        for trace in self.traces.values():
            old_strength = trace.strength
            trace.decay(dt)
            if trace.strength < old_strength:
                stats["decayed"] += 1
        
        # Step 2: Merge similar attractors
        merged = self._merge_similar()
        stats["merged"] = merged
        
        # Step 3: Prune weak traces
        before = len(self.traces)
        self.traces = {
            k: v for k, v in self.traces.items()
            if v.is_viable() and v.strength >= self.config.prune_strength_threshold
        }
        stats["pruned"] = before - len(self.traces)
        
        # Step 4: Enforce capacity
        if len(self.traces) > self.config.max_memories:
            self._prune_to_capacity()
            stats["pruned"] += len(self.traces) - self.config.max_memories
        
        # Step 5: Update identity
        self._identity_cache = None
        self._identity_dirty = True
        
        # Consolidate field
        self.field.consolidate(dt)
        
        return stats
    
    def _merge_similar(self) -> int:
        """Merge attractors that are too similar."""
        labels = list(self.traces.keys())
        merged_count = 0
        skip: set[str] = set()
        
        for i in range(len(labels)):
            if labels[i] in skip:
                continue
            for j in range(i + 1, len(labels)):
                if labels[j] in skip:
                    continue
                
                t1 = self.traces.get(labels[i])
                t2 = self.traces.get(labels[j])
                if t1 is None or t2 is None:
                    continue
                
                c1 = np.array(t1.attractor_center)
                c2 = np.array(t2.attractor_center)
                
                similarity = self._cosine_similarity(c1, c2)
                
                if similarity >= self.config.merge_similarity_threshold:
                    # Merge t2 into t1 (keep stronger)
                    if t2.strength > t1.strength:
                        t1, t2 = t2, t1
                        labels[i], labels[j] = labels[j], labels[i]
                    
                    # Weighted merge of centers
                    total = t1.strength + t2.strength
                    w1 = t1.strength / total if total > 0 else 0.5
                    t1.attractor_center = (
                        (w1 * np.array(t1.attractor_center) +
                         (1 - w1) * np.array(t2.attractor_center)).tolist()
                    )
                    t1.strength = min(1.0, total * 0.9)
                    t1.events.extend(t2.events)
                    t1.description += f" + {t2.description}"
                    
                    # Remove t2
                    del self.traces[labels[j]]
                    skip.add(labels[j])
                    merged_count += 1
        
        return merged_count
    
    def _prune_to_capacity(self) -> None:
        """Keep only the strongest memories up to capacity."""
        sorted_traces = sorted(
            self.traces.items(),
            key=lambda kv: kv[1].strength * kv[1].importance,
            reverse=True,
        )
        self.traces = dict(sorted_traces[:self.config.max_memories])
    
    def identity_pattern(self) -> Optional[np.ndarray]:
        """The persistent attractor cluster = the agent's identity.
        
        Identity is the weighted average of the strongest, most stable attractors.
        """
        if not self._identity_dirty and self._identity_cache is not None:
            return self._identity_cache
        
        # Find strong, stable attractors
        strong_traces = [
            t for t in self.traces.values()
            if t.strength >= 0.5 and t.is_viable()
        ]
        
        if len(strong_traces) < self.config.identity_min_cluster:
            self._identity_cache = None
            return None
        
        # Weighted average of their centers
        total_strength = sum(t.strength for t in strong_traces)
        if total_strength < 1e-12:
            self._identity_cache = None
            return None
        
        identity = np.zeros(16)
        for t in strong_traces:
            weight = t.strength / total_strength
            identity += weight * np.array(t.attractor_center)
        
        self._identity_cache = identity
        self._identity_dirty = False
        return identity
    
    def identity_coherence(self) -> float:
        """How internally aligned the identity pattern is (0-1).
        
        High coherence = identity attractors are tightly clustered.
        Low coherence = identity is fragmented.
        """
        identity = self.identity_pattern()
        if identity is None:
            return 0.0
        
        strong_traces = [
            t for t in self.traces.values()
            if t.strength >= 0.5 and t.is_viable()
        ]
        
        if len(strong_traces) < 2:
            return 1.0
        
        # Average distance from each attractor to the identity center
        distances = []
        for t in strong_traces:
            center = np.array(t.attractor_center)
            dist = np.linalg.norm(center - identity)
            distances.append(dist)
        
        avg_distance = np.mean(distances)
        # Normalize: 0 distance = perfect coherence (1.0)
        return float(max(0.0, 1.0 - avg_distance))
    
    def identity_shift_detected(self, threshold: float = 0.3) -> bool:
        """Detect if identity has significantly changed."""
        if self._identity_cache is None:
            return False
        
        new_identity = self.identity_pattern()
        if new_identity is None:
            return True
        
        shift = np.linalg.norm(new_identity - self._identity_cache)
        return shift > threshold
    
    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        n1 = np.linalg.norm(a)
        n2 = np.linalg.norm(b)
        if n1 < 1e-12 or n2 < 1e-12:
            return 0.0
        return float(np.dot(a, b) / (n1 * n2))
    
    def memory_stats(self) -> dict:
        """Statistics about the memory system."""
        active = [t for t in self.traces.values() if t.is_viable()]
        return {
            "total_memories": len(self.traces),
            "active_memories": len(active),
            "total_recalls": sum(t.recall_count for t in active),
            "avg_strength": sum(t.strength for t in active) / max(1, len(active)),
            "identity_coherence": self.identity_coherence(),
            "has_identity": self.identity_pattern() is not None,
            "by_type": {
                tt.name: sum(1 for t in active if t.trace_type == tt)
                for tt in TraceType
            },
        }
    
    @staticmethod
    def _experience_to_trace_type(exp: Experience) -> TraceType:
        mapping = {
            ExperienceType.PERCEPTION: TraceType.EPISODIC,
            ExperienceType.INTERACTION: TraceType.EPISODIC,
            ExperienceType.SOCIAL: TraceType.EPISODIC,
            ExperienceType.REFLECTION: TraceType.SEMANTIC,
            ExperienceType.SURPRISE: TraceType.EMOTIONAL,
            ExperienceType.GOAL_ACHIEVED: TraceType.PROCEDURAL,
            ExperienceType.GOAL_FAILED: TraceType.PROCEDURAL,
            ExperienceType.LEARNING: TraceType.SEMANTIC,
        }
        return mapping.get(exp.experience_type, TraceType.EPISODIC)
