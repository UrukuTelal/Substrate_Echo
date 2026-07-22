"""Dynamics Memory — learns the vector field of experience.

Instead of:  Experience → Pattern → Recall
We use:     Experience → (state, velocity) → Learn V(x) → Predict

This gives the system a world model, not just episodic memory.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import time

from ..models.experience import Experience, ExperienceType
from ..models.memory_trace import MemoryTrace, TraceType


@dataclass
class DynamicsMemoryConfig:
    """Configuration for dynamics memory."""
    max_samples: int = 2000       # max (state, velocity) pairs to retain
    ridge_alpha: float = 0.001    # regularization for linear fit
    min_samples_for_fit: int = 50 # minimum samples before fitting V(x)
    attractor_samples: int = 300  # random starts for attractor discovery
    attractor_integration_steps: int = 200
    dedup_threshold: float = 0.5  # min distance between distinct attractors
    consolidation_interval: float = 60.0
    merge_similarity_threshold: float = 0.9
    prune_strength_threshold: float = 0.05
    max_memories: int = 1000
    # Local linear model params
    model_type: str = "global"    # "global" (ridge) or "local" (k-NN local linear)
    k_neighbors: int = 30         # neighbors for local model
    bandwidth: float = 1.0        # Gaussian kernel bandwidth for local model


class _LocalLinearModel:
    """k-NN local linear vector field model.
    
    For each query point:
    1. Find k nearest neighbors from training data
    2. Fit local linear: v = A*(x - x_center) + v_center
    3. Predict velocity
    
    This captures multi-basin dynamics that a global linear model cannot.
    
    Caches numpy arrays internally; call `_rebuild_index()` after adding samples.
    """
    
    def __init__(self, dim: int, k: int = 20, bandwidth: float = 0.3):
        self.dim = dim
        self.k = k
        self.bandwidth = bandwidth
        self._states: list[np.ndarray] = []
        self._velocities: list[np.ndarray] = []
        self._states_arr: Optional[np.ndarray] = None
        self._vels_arr: Optional[np.ndarray] = None
        self._dirty = True
    
    def add_sample(self, state: np.ndarray, velocity: np.ndarray):
        self._states.append(state.copy())
        self._velocities.append(velocity.copy())
        self._dirty = True
    
    def _rebuild_index(self):
        """Rebuild cached numpy arrays after new samples added."""
        if self._dirty and len(self._states) > 0:
            self._states_arr = np.array(self._states)
            self._vels_arr = np.array(self._velocities)
            self._dirty = False
    
    def predict_velocity(self, state: np.ndarray) -> np.ndarray:
        if len(self._states) < 3:
            return np.zeros(self.dim)
        
        self._rebuild_index()
        
        k = min(self.k, len(self._states))
        dists = np.linalg.norm(self._states_arr - state, axis=1)
        nn_idx = np.argpartition(dists, k)[:k]
        nn_dists = dists[nn_idx]
        
        weights = np.exp(-nn_dists**2 / (2 * self.bandwidth**2))
        wsum = weights.sum()
        if wsum < 1e-10:
            return np.zeros(self.dim)
        weights = weights / wsum
        
        nn_states = self._states_arr[nn_idx]
        nn_vels = self._vels_arr[nn_idx]
        
        x_center = np.average(nn_states, axis=0, weights=weights)
        v_center = np.average(nn_vels, axis=0, weights=weights)
        
        dx = nn_states - x_center
        dv = nn_vels - v_center
        
        W = np.diag(weights)
        Cxx = dx.T @ W @ dx + 1e-6 * np.eye(self.dim)
        Cvx = dv.T @ W @ dx
        try:
            A = Cvx @ np.linalg.inv(Cxx)
        except np.linalg.LinAlgError:
            return v_center
        
        return A @ (state - x_center) + v_center


class DynamicsMemory:
    """Memory that learns the dynamics of experience.
    
    Interface-compatible with AttractorMemory, but internally learns
    a vector field V(x) that maps states to velocities.
    
    This enables:
    - Prediction: "where will this state go?"
    - Attractor discovery: "what are the stable states?"
    - Basin mapping: "which basin is this state in?"
    - Transition prediction: "what's the probability of moving between basins?"
    """
    
    def __init__(self, dim: int = 16, config: Optional[DynamicsMemoryConfig] = None):
        self.dim = dim
        self.config = config or DynamicsMemoryConfig()
        
        # Dynamics model: V(x) = A*x + b
        self.A: Optional[np.ndarray] = None  # (dim, dim) global model
        self.b: Optional[np.ndarray] = None  # (dim,) global model
        self._fitted = False
        
        # Local linear model (for model_type="local")
        self._local_model: Optional[_LocalLinearModel] = None
        if self.config.model_type == "local":
            self._local_model = _LocalLinearModel(
                dim=dim, k=self.config.k_neighbors,
                bandwidth=self.config.bandwidth,
            )
        
        # Training data
        self._states: list[np.ndarray] = []
        self._velocities: list[np.ndarray] = []
        
        # Discovered attractors
        self._attractors: list[np.ndarray] = []
        self._basin_map: dict[int, list[int]] = {}  # attractor_idx -> trace indices
        
        # Memory traces (interface compatibility)
        self.traces: dict[str, MemoryTrace] = {}
        self._prev_psv: Optional[np.ndarray] = None
        
        # Consolidation
        self._last_consolidation = time.time()
        self._consolidation_count = 0
    
    # ── Core Interface ──────────────────────────────────────────
    
    def encode(self, experience: Experience) -> Optional[MemoryTrace]:
        """Encode an experience into dynamics memory.
        
        1. Store (state, velocity) pair
        2. Create memory trace (interface compat)
        3. Refit V(x) if enough data
        """
        if experience.psv_snapshot is None:
            return None
        
        psv = np.array(experience.psv_snapshot, dtype=np.float64)
        
        # Compute velocity from consecutive states
        if self._prev_psv is not None:
            velocity = psv - self._prev_psv
            
            # Store for dynamics learning
            self._states.append(self._prev_psv.copy())
            self._velocities.append(velocity.copy())
            
            # Feed local model
            if self._local_model is not None:
                self._local_model.add_sample(self._prev_psv, velocity)
            
            # Trim to max_samples
            if len(self._states) > self.config.max_samples:
                self._states = self._states[-self.config.max_samples:]
                self._velocities = self._velocities[-self.config.max_samples:]
            
            # Refit global model periodically (if applicable)
            if self.config.model_type == "global":
                if len(self._states) >= self.config.min_samples_for_fit:
                    if len(self._states) % 50 == 0:
                        self._fit_dynamics()
            else:
                # Local model is always "fitted" once we have data
                if len(self._states) >= self.config.min_samples_for_fit:
                    self._fitted = True
        
        self._prev_psv = psv.copy()
        
        # Create memory trace (interface compat with AttractorMemory)
        strength = experience.importance
        if experience.experience_type == ExperienceType.SURPRISE:
            strength *= 1.5
        strength = min(1.0, strength)
        
        label = f"dyn_{experience.experience_id}"
        trace = MemoryTrace(
            trace_id=experience.experience_id,
            trace_type=self._exp_to_trace_type(experience),
            attractor_center=psv.tolist(),
            description=experience.description,
            object_ids=experience.object_ids,
            events=[experience.to_dict()],
            emotional_valence=experience.result_valence,
            importance=experience.importance,
            strength=strength,
        )
        self.traces[label] = trace
        
        return trace
    
    def recall(self, cue: np.ndarray, k: int = 5) -> list[MemoryTrace]:
        """Recall by nearest traces (same as AttractorMemory).
        
        If dynamics model is fitted, also returns predicted future state.
        """
        cue = np.asarray(cue, dtype=np.float64)
        
        # Find k nearest traces
        candidates = []
        for label, trace in self.traces.items():
            center = np.array(trace.attractor_center)
            dist = np.linalg.norm(cue - center)
            candidates.append((dist, trace))
        
        candidates.sort(key=lambda x: x[0])
        
        results = []
        for _, trace in candidates[:k]:
            if trace.is_viable():
                trace.recall()
                results.append(trace)
        
        return results
    
    def recall_by_cue(self, cue: np.ndarray, top_k: int = 5) -> list:
        """Recall compatible with cognitive loop interface.
        
        Returns list of objects with .center and .strength attributes,
        matching what CognitiveLoop.recall_memories() expects.
        """
        cue = np.asarray(cue, dtype=np.float64)
        
        candidates = []
        for label, trace in self.traces.items():
            center = np.array(trace.attractor_center)
            dist = np.linalg.norm(cue - center)
            candidates.append((dist, trace))
        
        candidates.sort(key=lambda x: x[0])
        
        results = []
        for dist, trace in candidates[:top_k]:
            if trace.is_viable():
                trace.recall()
                # Return a simple namespace with .center and .strength
                # so CognitiveLoop.recall_memories() can use it
                results.append(type('RecallResult', (), {
                    'center': np.array(trace.attractor_center),
                    'strength': trace.strength,
                    'description': trace.description,
                    'trace': trace,
                })())
        
        return results
    
    def predict(self, state: np.ndarray, steps: int = 1, dt: float = 0.02) -> np.ndarray:
        """Predict future state by integrating V(x).
        
        This is the key new capability that AttractorMemory lacks.
        """
        if not self._fitted:
            return state.copy()
        
        state = np.asarray(state, dtype=np.float64)
        x = state.copy()
        for _ in range(steps):
            v = self.predict_velocity(x)
            x = x + dt * v
            x = np.clip(x, 0.0, 1.0)  # PSV bounds
        return x
    
    def predict_velocity(self, state: np.ndarray) -> np.ndarray:
        """Predict velocity at a state."""
        if not self._fitted:
            return np.zeros(self.dim)
        state = np.asarray(state, dtype=np.float64)
        if self._local_model is not None:
            return self._local_model.predict_velocity(state)
        return self.A @ state + self.b

    def prediction_error(self, state: np.ndarray,
                         actual_velocity: np.ndarray) -> float:
        """Measure prediction error at a state.

        Returns MSE between predicted and actual velocity.
        High error = model doesn't understand this region = curiosity signal.
        Returns 0.0 if model isn't fitted yet (no basis for comparison).
        """
        if not self._fitted:
            return 0.0
        state = np.asarray(state, dtype=np.float64)
        actual_velocity = np.asarray(actual_velocity, dtype=np.float64)
        predicted = self.predict_velocity(state)
        return float(np.mean((predicted - actual_velocity) ** 2))

    def region_uncertainty(self, state: np.ndarray,
                           n_samples: int = 20,
                           noise_scale: float = 0.05) -> float:
        """Estimate model uncertainty in a region by sampling predictions.

        Injects Gaussian noise around the state and measures variance
        of predicted velocities. High variance = model is uncertain.
        Returns 0.0 if not fitted.
        """
        if not self._fitted:
            return 0.0
        state = np.asarray(state, dtype=np.float64)
        rng = np.random.RandomState(42)
        predictions = []
        for _ in range(n_samples):
            perturbed = state + rng.randn(len(state)) * noise_scale
            perturbed = np.clip(perturbed, 0.0, 1.0)
            predictions.append(self.predict_velocity(perturbed))
        predictions = np.array(predictions)
        # Mean variance across dimensions
        return float(np.mean(np.var(predictions, axis=0)))
    
    def novelty(self, state: np.ndarray) -> float:
        """Measure how novel a state is (distance to nearest training sample).

        High novelty = far from any training data = model hasn't seen
        this region = likely high prediction error.
        This is a better information gain signal than prediction variance,
        because a model can be confidently wrong (low variance, high error).
        Returns 0.0 if no training data.
        """
        if not self._states:
            return 0.0
        state = np.asarray(state, dtype=np.float64)
        states_arr = np.array(self._states)
        distances = np.linalg.norm(states_arr - state, axis=1)
        return float(np.min(distances))
    
    def information_gain(self, state: np.ndarray,
                          n_samples: int = 10,
                          noise_scale: float = 0.05) -> float:
        """Estimate information gain at a state.

        Combines novelty (distance to training data) with prediction
        uncertainty (variance under perturbation). This is the
        exploration reward: high information gain = visit this region.
        """
        nov = self.novelty(state)
        unc = self.region_uncertainty(state, n_samples=n_samples,
                                       noise_scale=noise_scale)
        # Normalize novelty by typical training data spacing
        if self._states and len(self._states) > 1:
            states_arr = np.array(self._states)
            # Average nearest-neighbor distance in training data
            sample_idx = np.random.choice(len(self._states),
                                           min(50, len(self._states)),
                                           replace=False)
            sample_dists = []
            for i in sample_idx:
                dists = np.linalg.norm(states_arr - states_arr[i], axis=1)
                dists_sorted = np.sort(dists)
                if len(dists_sorted) > 1:
                    sample_dists.append(dists_sorted[1])  # nearest non-self
            avg_spacing = np.mean(sample_dists) if sample_dists else 1.0
            nov_normalized = nov / max(avg_spacing, 1e-6)
        else:
            nov_normalized = nov
        
        # Combined: novelty dominates (out-of-distribution),
        # uncertainty adds local information
        return float(nov_normalized + 0.1 * unc)
    
    def discover_attractors(self) -> list[np.ndarray]:
        """Discover attractors from the learned dynamics.
        
        For linear V(x) = Ax + b, fixed points satisfy x* = -A^{-1} b.
        For multi-attractor dynamics (local model), integrates from random starts
        and clusters endpoints with mean-shift.
        """
        if not self._fitted:
            return []
        
        attractors = []
        
        # Analytical fixed point for global linear model only
        if self.A is not None and self.b is not None:
            try:
                fp = -np.linalg.solve(self.A, self.b)
                if np.all(fp >= -0.1) and np.all(fp <= 1.1):
                    fp = np.clip(fp, 0.0, 1.0)
                    attractors.append(fp)
            except np.linalg.LinAlgError:
                pass
            
            # Check if global model needs integration-based discovery
            eigvals = np.linalg.eigvals(self.A)
            has_complext = any(abs(np.imag(e)) > 0.01 for e in eigvals)
            has_positive = any(np.real(e) > 0.01 for e in eigvals)
            needs_integration = has_complext or has_positive or len(attractors) == 0
        else:
            # Local model always uses integration
            needs_integration = True
        
        if needs_integration:
            rng = np.random.RandomState(42)
            endpoints = []
            for _ in range(self.config.attractor_samples):
                x = rng.uniform(0.1, 0.9, self.dim)
                for _ in range(self.config.attractor_integration_steps):
                    v = self.predict_velocity(x)
                    x = x + 0.02 * v
                    x = np.clip(x, 0.0, 1.0)
                endpoints.append(x)
            endpoints = np.array(endpoints)
            
            # Cluster with aggressive mean-shift
            bandwidth = 0.5
            shifted = endpoints.copy()
            for _ in range(30):
                new_shifted = np.zeros_like(shifted)
                for i, ep in enumerate(shifted):
                    dists = np.linalg.norm(endpoints - ep, axis=1)
                    weights = np.exp(-0.5 * (dists / bandwidth) ** 2)
                    wsum = weights.sum()
                    if wsum > 1e-10:
                        new_shifted[i] = np.average(endpoints, weights=weights, axis=0)
                    else:
                        new_shifted[i] = ep
                if np.allclose(shifted, new_shifted, atol=1e-4):
                    break
                shifted = new_shifted
            
            for ep in shifted:
                is_new = True
                for existing in attractors:
                    if np.linalg.norm(ep - existing) < self.config.dedup_threshold:
                        is_new = False
                        break
                if is_new:
                    attractors.append(ep.copy())
        
        self._attractors = attractors
        return attractors
    
    def basin_of(self, state: np.ndarray) -> int:
        """Which basin does this state belong to? Returns index or -1."""
        if not self._attractors:
            return -1
        
        state = np.asarray(state, dtype=np.float64)
        
        # Integrate to convergence
        x = state.copy()
        for _ in range(200):
            v = self.predict_velocity(x)
            x = x + 0.02 * v
            x = np.clip(x, 0.0, 1.0)
        
        # Find nearest attractor
        dists = [np.linalg.norm(x - a) for a in self._attractors]
        best = np.argmin(dists)
        if dists[best] < 0.5:
            return best
        return -1
    
    def transition_probability(self, from_basin: int, steps: int = 100) -> dict[int, float]:
        """Estimate transition probability from one basin to another.
        
        Perturbs from the basin center in multiple directions,
        integrates, and counts where trajectories end up.
        """
        if not self._attractors or from_basin >= len(self._attractors):
            return {}
        
        center = self._attractors[from_basin]
        rng = np.random.RandomState(from_basin)
        
        counts = {}
        n_trials = 20
        
        for _ in range(n_trials):
            # Small perturbation
            x = center + rng.randn(self.dim) * 0.1
            x = np.clip(x, 0.0, 1.0)
            
            for _ in range(steps):
                v = self.predict_velocity(x)
                x = x + 0.02 * v
                x = np.clip(x, 0.0, 1.0)
            
            # Which basin?
            dists = [np.linalg.norm(x - a) for a in self._attractors]
            best = np.argmin(dists)
            counts[best] = counts.get(best, 0) + 1
        
        return {k: v / n_trials for k, v in counts.items()}
    
    def stability_at(self, state: np.ndarray, eps: float = 0.05) -> dict:
        """Analyze stability at a state using the learned Jacobian.
        
        Returns eigenvalues and classification.
        """
        if not self._fitted:
            return {'eigenvalues': [], 'classification': 'unknown'}
        
        state = np.asarray(state, dtype=np.float64)
        
        # Numerical Jacobian
        J = np.zeros((self.dim, self.dim))
        for i in range(self.dim):
            state_plus = state.copy()
            state_minus = state.copy()
            state_plus[i] += eps
            state_minus[i] -= eps
            
            v_plus = self.predict_velocity(state_plus)
            v_minus = self.predict_velocity(state_minus)
            
            J[:, i] = (v_plus - v_minus) / (2 * eps)
        
        eigvals = np.linalg.eigvals(J)
        real_parts = np.real(eigvals)
        
        if all(r < -0.01 for r in real_parts):
            classification = 'attractor'
        elif all(r > 0.01 for r in real_parts):
            classification = 'repellor'
        elif any(r > 0.01 for r in real_parts) and any(r < -0.01 for r in real_parts):
            classification = 'saddle'
        else:
            classification = 'marginal'
        
        return {
            'eigenvalues': eigvals.tolist(),
            'real_parts': real_parts.tolist(),
            'classification': classification,
            'max_eigenvalue_real': float(max(real_parts)),
        }
    
    def consolidate(self, force: bool = False) -> dict:
        """Consolidation cycle."""
        now = time.time()
        if not force and (now - self._last_consolidation) < self.config.consolidation_interval:
            return {"skipped": True}
        
        self._last_consolidation = now
        self._consolidation_count += 1
        stats = {"merged": 0, "pruned": 0, "retracked_attractors": False}
        
        # Prune weak traces
        before = len(self.traces)
        self.traces = {
            k: v for k, v in self.traces.items()
            if v.is_viable() and v.strength >= self.config.prune_strength_threshold
        }
        stats["pruned"] = before - len(self.traces)
        
        # Re-discover attractors periodically
        if self._fitted and self._consolidation_count % 5 == 0:
            self.discover_attractors()
            stats["retracked_attractors"] = True
        
        return stats
    
    def identity_pattern(self) -> Optional[np.ndarray]:
        """Identity = weighted average of discovered attractors."""
        if not self._attractors:
            return None
        
        # Weight by number of traces near each attractor
        weights = []
        for att in self._attractors:
            count = 0
            for trace in self.traces.values():
                center = np.array(trace.attractor_center)
                if np.linalg.norm(center - att) < 1.0:
                    count += 1
            weights.append(max(count, 1))
        
        total = sum(weights)
        identity = np.zeros(self.dim)
        for att, w in zip(self._attractors, weights):
            identity += (w / total) * att
        
        return identity
    
    def identity_coherence(self) -> float:
        """How tightly clustered are the attractors?"""
        if len(self._attractors) < 2:
            return 1.0
        
        identity = self.identity_pattern()
        if identity is None:
            return 0.0
        
        distances = [np.linalg.norm(a - identity) for a in self._attractors]
        avg_dist = np.mean(distances)
        return float(max(0.0, 1.0 - avg_dist * 2))
    
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
            "dynamics_fitted": self._fitted,
            "n_training_samples": len(self._states),
            "n_attractors": len(self._attractors),
            "attractors": [a.tolist() for a in self._attractors],
        }
    
    # ── Internal Methods ────────────────────────────────────────
    
    def _fit_dynamics(self):
        """Fit V(x) = Ax + b using Ridge regression."""
        if len(self._states) < self.config.min_samples_for_fit:
            return
        
        X = np.array(self._states)
        V = np.array(self._velocities)
        
        # Center data
        X_mean = np.mean(X, axis=0)
        V_mean = np.mean(V, axis=0)
        X_centered = X - X_mean
        V_centered = V - V_mean
        
        # Ridge regression: A = (X^T X + alpha*I)^{-1} X^T V
        XtX = X_centered.T @ X_centered
        reg = self.config.ridge_alpha * np.eye(self.dim)
        self.A = np.linalg.solve(XtX + reg, X_centered.T @ V_centered)
        
        # Recover b: V_mean = A * X_mean + b => b = V_mean - A * X_mean
        self.b = V_mean - self.A @ X_mean
        
        self._fitted = True
    
    @staticmethod
    def _exp_to_trace_type(exp: Experience) -> TraceType:
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
