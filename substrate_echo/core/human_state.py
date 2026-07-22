"""HSV — Human State Vector: probabilistic human state estimation.

Stage 2 of the developmental architecture (State Estimation layer).

The HSV estimates a human's internal state from observable behavioral
signals. Each dimension is a Gaussian (mean, variance) capturing both
the best estimate and the uncertainty.

Dimensions:
  0. Arousal        — activation level (calm ↔ excited)
  1. Valence        — emotional tone (negative ↔ positive)
  2. Attention      — focus intensity (scattered ↔ locked-on)
  3. Social Openness — interaction willingness (closed ↔ open)
  4. Fatigue        — tiredness (energetic ↔ exhausted)
  5. Intent Clarity — goal-directedness (aimless ↔ purposeful)
  6. Stability      — emotional stability (volatile ↔ steady)

The estimator is Bayesian: each observation updates the posterior,
and uncertainty shrinks with consistent evidence. Conflicting signals
increase variance, correctly representing "I'm not sure."

HSV plugs into affordance translation:

    Entity → HSV estimate → Intent probability → Affordance weighting

Same behavior (human approaches quickly) can mean friendly or threatening.
HSV provides the context to disambiguate.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import numpy as np


# ── Dimension Definitions ────────────────────────────────────────

HSV_DIM_NAMES = [
    "arousal", "valence", "attention", "social_openness",
    "fatigue", "intent_clarity", "stability",
]
HSV_DIM = len(HSV_DIM_NAMES)


@dataclass
class GaussianDim:
    """A single Gaussian-distributed dimension."""
    mean: float = 0.5      # [0, 1] best estimate
    variance: float = 0.25  # [0, 0.5] uncertainty (max 0.5 = completely unknown)
    
    @property
    def std(self) -> float:
        return np.sqrt(self.variance)
    
    @property
    def confidence(self) -> float:
        """How confident we are (1 - normalized uncertainty)."""
        return max(0.0, 1.0 - self.variance * 2.0)
    
    def sample(self) -> float:
        """Sample from the distribution, clamped to [0, 1]."""
        return float(np.clip(np.random.normal(self.mean, self.std), 0.0, 1.0))
    
    def update(self, observation: float, obs_variance: float) -> None:
        """Bayesian update: combine prior with new observation.
        
        Uses the standard Kalman-style update:
        K = σ_prior² / (σ_prior² + σ_obs²)
        μ_new = μ_prior + K * (z - μ_prior)
        σ_new² = (1 - K) * σ_prior²
        """
        obs_variance = max(obs_variance, 1e-6)
        prior_var = max(self.variance, 1e-6)
        
        kalman_gain = prior_var / (prior_var + obs_variance)
        
        self.mean = self.mean + kalman_gain * (observation - self.mean)
        self.mean = float(np.clip(self.mean, 0.0, 1.0))
        
        self.variance = (1.0 - kalman_gain) * prior_var
        self.variance = float(np.clip(self.variance, 0.0, 0.5))


@dataclass
class HSVState:
    """Full probabilistic human state vector (7D)."""
    arousal: GaussianDim = field(default_factory=GaussianDim)
    valence: GaussianDim = field(default_factory=GaussianDim)
    attention: GaussianDim = field(default_factory=GaussianDim)
    social_openness: GaussianDim = field(default_factory=GaussianDim)
    fatigue: GaussianDim = field(default_factory=GaussianDim)
    intent_clarity: GaussianDim = field(default_factory=GaussianDim)
    stability: GaussianDim = field(default_factory=GaussianDim)
    
    @property
    def means(self) -> np.ndarray:
        """All means as a vector."""
        return np.array([
            self.arousal.mean, self.valence.mean, self.attention.mean,
            self.social_openness.mean, self.fatigue.mean,
            self.intent_clarity.mean, self.stability.mean,
        ])
    
    @property
    def variances(self) -> np.ndarray:
        """All variances as a vector."""
        return np.array([
            self.arousal.variance, self.valence.variance,
            self.attention.variance, self.social_openness.variance,
            self.fatigue.variance, self.intent_clarity.variance,
            self.stability.variance,
        ])
    
    @property
    def uncertainty(self) -> float:
        """Mean uncertainty across all dimensions."""
        return float(np.mean(self.variances))
    
    @property
    def confidence(self) -> float:
        """Mean confidence across all dimensions."""
        dims = [self.arousal, self.valence, self.attention,
                self.social_openness, self.fatigue,
                self.intent_clarity, self.stability]
        return float(np.mean([d.confidence for d in dims]))
    
    def to_array(self) -> np.ndarray:
        """Stacked [means; variances] vector (14D)."""
        return np.concatenate([self.means, self.variances])
    
    @classmethod
    def from_array(cls, arr: np.ndarray) -> HSVState:
        """Reconstruct from 14D [means; variances] vector."""
        state = cls()
        dims = [state.arousal, state.valence, state.attention,
                state.social_openness, state.fatigue,
                state.intent_clarity, state.stability]
        for i, dim in enumerate(dims):
            dim.mean = float(arr[i])
            dim.variance = float(arr[i + 7])
        return state
    
    def __repr__(self):
        m = self.means
        return (f"HSV(A={m[0]:.2f} V={m[1]:.2f} At={m[2]:.2f} "
                f"SO={m[3]:.2f} F={m[4]:.2f} IC={m[5]:.2f} S={m[6]:.2f} "
                f"±{self.uncertainty:.3f})")


# ── Observation Model ────────────────────────────────────────────

@dataclass
class HumanObservation:
    """Raw observable signals from a human.
    
    All values in [0, 1] unless noted. NaN means "not observed."
    """
    # Arousal signals
    speech_level: float = float('nan')      # volume / intensity
    gesture_speed: float = float('nan')     # hand/arm motion
    motion_speed: float = float('nan')      # whole-body movement
    body_tension: float = float('nan')      # muscle tension estimate
    
    # Valence signals
    facial_openness: float = float('nan')   # smile/frown proxy
    vocal_tone: float = float('nan')        # pitch variation
    posture_openness: float = float('nan')  # open vs closed body
    
    # Attention signals
    gaze_stability: float = float('nan')    # how steady their gaze
    orientation_consistency: float = float('nan')  # body facing same dir
    fidget_level: float = float('nan')      # low = focused, high = restless
    
    # Social openness signals
    facing_toward: float = float('nan')     # body facing AI
    approach_behavior: float = float('nan') # moving toward AI
    eye_contact_frequency: float = float('nan')  # how often they look at AI
    
    # Fatigue signals
    blink_rate: float = float('nan')        # high = tired
    posture_slump: float = float('nan')     # high = slouched/tired
    response_latency: float = float('nan')  # high = slow = tired
    
    # Intent clarity signals
    trajectory_directness: float = float('nan')  # straight vs meandering
    gesture_repetition: float = float('nan')     # repeated actions
    gaze_fixation: float = float('nan')          # looking at specific target
    
    # Stability signals
    emotional_variability: float = float('nan')  # high = volatile
    behavioral_consistency: float = float('nan')  # high = predictable
    recovery_speed: float = float('nan')          # how fast they calm down
    
    def get_signal(self, name: str) -> Optional[float]:
        """Get a signal value, returning None if not observed."""
        val = getattr(self, name, None)
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return None
        return float(val)


# ── Signal → Dimension Mapping ───────────────────────────────────

# Which signals contribute to which HSV dimension, and with what weight.
# Positive weight = signal increases the dimension.
# Negative weight = signal decreases the dimension.
SIGNAL_MAP: dict[str, list[tuple[str, float]]] = {
    "arousal": [
        ("speech_level", 0.3),
        ("gesture_speed", 0.25),
        ("motion_speed", 0.25),
        ("body_tension", 0.2),
    ],
    "valence": [
        ("facial_openness", 0.35),
        ("vocal_tone", 0.25),
        ("posture_openness", 0.25),
        ("speech_level", 0.15),
    ],
    "attention": [
        ("gaze_stability", 0.35),
        ("orientation_consistency", 0.3),
        ("fidget_level", -0.35),  # high fidget = low attention
    ],
    "social_openness": [
        ("facing_toward", 0.3),
        ("approach_behavior", 0.3),
        ("eye_contact_frequency", 0.25),
        ("posture_openness", 0.15),
    ],
    "fatigue": [
        ("blink_rate", 0.3),
        ("posture_slump", 0.3),
        ("response_latency", 0.25),
        ("motion_speed", -0.15),  # low motion = high fatigue
    ],
    "intent_clarity": [
        ("trajectory_directness", 0.35),
        ("gesture_repetition", 0.25),
        ("gaze_fixation", 0.25),
        ("orientation_consistency", 0.15),
    ],
    "stability": [
        ("emotional_variability", -0.35),  # high variability = low stability
        ("behavioral_consistency", 0.35),
        ("recovery_speed", 0.3),
    ],
}


# ── Intent Hypothesis ────────────────────────────────────────────

@dataclass
class IntentHypothesis:
    """A probabilistic hypothesis about what a human intends."""
    label: str              # e.g., "friendly_approach", "threatening"
    probability: float      # [0, 1]
    supporting_pillars: list[int] = field(default_factory=list)  # which PSV dims to adjust
    pillar_deltas: list[float] = field(default_factory=list)     # how much to adjust each


# ── Estimator ────────────────────────────────────────────────────

# Intent hypotheses: mapping from HSV region → likely intent
INTENT_HYPOTHESES: list[IntentHypothesis] = [
    IntentHypothesis(
        label="friendly_approach",
        probability=0.0,
        supporting_pillars=[7, 8, 9],  # Relation, Presence, Warmth
        pillar_deltas=[0.15, 0.10, 0.10],
    ),
    IntentHypothesis(
        label="seeking_assistance",
        probability=0.0,
        supporting_pillars=[7, 4, 10],  # Relation, Resistance, Memory
        pillar_deltas=[0.10, 0.05, 0.10],
    ),
    IntentHypothesis(
        label="threatening",
        probability=0.0,
        supporting_pillars=[4, 0, 12],  # Resistance, Awareness, Harm
        pillar_deltas=[0.20, 0.15, 0.10],
    ),
    IntentHypothesis(
        label="fatigued",
        probability=0.0,
        supporting_pillars=[8, 0],  # Presence, Awareness
        pillar_deltas=[0.05, 0.05],
    ),
    IntentHypothesis(
        label="social_engagement",
        probability=0.0,
        supporting_pillars=[7, 8, 9, 3],  # Relation, Presence, Warmth, Influence
        pillar_deltas=[0.20, 0.10, 0.10, 0.05],
    ),
]


class HumanStateEstimator:
    """Estimates human internal state from behavioral observations.
    
    Bayesian estimator: each observation updates a Gaussian posterior.
    Consistent evidence → low variance (high confidence).
    Conflicting evidence → high variance (low confidence).
    
    The estimator also generates intent hypotheses based on the
    HSV state, providing probabilistic intent estimates that feed
    into the affordance translation layer.
    
    Usage:
        estimator = HumanStateEstimator()
        
        obs = HumanObservation(
            speech_level=0.6, gaze_directness=0.8,
            facing_toward=0.9, proximity=0.7)
        estimator.observe(obs)
        
        hsv = estimator.estimate
        print(hsv)  # HSV(A=0.52 V=0.48 At=0.71 ...)
        
        intents = estimator.infer_intents()
        for intent in intents:
            if intent.probability > 0.3:
                print(f"  {intent.label}: {intent.probability:.2f}")
    """
    
    def __init__(self, initial_variance: float = 0.25):
        """Initialize with high uncertainty (no knowledge)."""
        self._state = HSVState()
        self._observation_count = 0
        self._last_observation: Optional[HumanObservation] = None
        self._history: list[HSVState] = []
        self._max_history = 20
        
        # Set all dimensions to prior
        for dim_name in HSV_DIM_NAMES:
            dim = getattr(self._state, dim_name)
            dim.mean = 0.5
            dim.variance = initial_variance
    
    @property
    def estimate(self) -> HSVState:
        return self._state
    
    @property
    def observation_count(self) -> int:
        return self._observation_count
    
    def observe(self, obs: HumanObservation) -> HSVState:
        """Process a new observation and update the state estimate."""
        self._last_observation = obs
        self._observation_count += 1
        
        for dim_name in HSV_DIM_NAMES:
            dim = getattr(self._state, dim_name)
            signal_weights = SIGNAL_MAP.get(dim_name, [])
            
            # Collect available signals
            available = []
            for signal_name, weight in signal_weights:
                val = obs.get_signal(signal_name)
                if val is not None:
                    available.append((val, weight))
            
            if not available:
                # No signals — uncertainty grows slightly
                dim.variance = min(0.5, dim.variance * 1.02)
                continue
            
            # Compute weighted observation
            total_weight = sum(abs(w) for _, w in available)
            if total_weight < 1e-6:
                continue
            
            # Center each signal around 0.5, then weight by direction.
            # signal=0.5 → contribution 0 (neutral)
            # signal=1.0, weight=+0.3 → +0.15 (increases dimension)
            # signal=0.0, weight=+0.3 → -0.15 (decreases dimension)
            # signal=0.0, weight=-0.35 → +0.175 (decreases dimension via negative weight)
            weighted_sum = sum((v - 0.5) * w for v, w in available)
            obs_value = (weighted_sum / total_weight + 1.0) / 2.0
            obs_value = float(np.clip(obs_value, 0.0, 1.0))
            
            # Observation variance: fewer signals = more uncertainty
            sparsity_penalty = 1.0 - len(available) / len(signal_weights)
            base_noise = 0.05
            obs_variance = base_noise + sparsity_penalty * 0.15
            
            # Bayesian update
            dim.update(obs_value, obs_variance)
        
        # Store history
        self._history.append(HSVState(
            arousal=GaussianDim(self._state.arousal.mean, self._state.arousal.variance),
            valence=GaussianDim(self._state.valence.mean, self._state.valence.variance),
            attention=GaussianDim(self._state.attention.mean, self._state.attention.variance),
            social_openness=GaussianDim(self._state.social_openness.mean, self._state.social_openness.variance),
            fatigue=GaussianDim(self._state.fatigue.mean, self._state.fatigue.variance),
            intent_clarity=GaussianDim(self._state.intent_clarity.mean, self._state.intent_clarity.variance),
            stability=GaussianDim(self._state.stability.mean, self._state.stability.variance),
        ))
        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history:]
        
        return self._state
    
    def observe_signals(self, **kwargs) -> HSVState:
        """Convenience: observe with keyword signals."""
        obs = HumanObservation(**kwargs)
        return self.observe(obs)
    
    def infer_intents(self) -> list[IntentHypothesis]:
        """Generate intent hypotheses from current HSV state.
        
        Maps HSV region → probability of each intent.
        Uses fuzzy rules: e.g., high arousal + low valence + low stability
        → higher probability of "threatening" intent.
        
        Also checks raw observation signals for direct threat indicators
        (body tension, low facial openness, fast aggressive approach).
        """
        a = self._state.arousal.mean
        v = self._state.valence.mean
        at = self._state.attention.mean
        so = self._state.social_openness.mean
        f = self._state.fatigue.mean
        ic = self._state.intent_clarity.mean
        s = self._state.stability.mean
        
        # Confidence-weighted probabilities
        conf = self._state.confidence
        
        # Direct observation threat boost: check raw signals
        obs_threat_boost = 0.0
        if self._last_observation is not None:
            obs = self._last_observation
            tension = obs.get_signal("body_tension") or 0.0
            facial = obs.get_signal("facial_openness") or 0.5
            gesture = obs.get_signal("gesture_speed") or 0.0
            motion = obs.get_signal("motion_speed") or 0.0
            # High tension + low facial openness + fast motion = threat
            if tension > 0.6 and facial < 0.3 and motion > 0.5:
                obs_threat_boost = 0.25
            elif tension > 0.5 and facial < 0.3:
                obs_threat_boost = 0.15
        
        hypotheses = []
        
        # Friendly approach: high valence + social openness + low fatigue
        friendly = (v * 0.3 + so * 0.3 + (1 - f) * 0.2 + (1 - a * 0.3) * 0.2)
        hypotheses.append(IntentHypothesis(
            label="friendly_approach",
            probability=friendly * conf,
            supporting_pillars=[7, 8, 9],
            pillar_deltas=[0.15 * friendly, 0.10 * friendly, 0.10 * friendly],
        ))
        
        # Seeking assistance: high attention + moderate arousal + social openness
        assist = (at * 0.3 + a * 0.2 + so * 0.3 + ic * 0.2)
        hypotheses.append(IntentHypothesis(
            label="seeking_assistance",
            probability=assist * conf,
            supporting_pillars=[7, 4, 10],
            pillar_deltas=[0.10 * assist, 0.05 * assist, 0.10 * assist],
        ))
        
        # Threatening: multiplicative interaction of high arousal + low valence
        # This is the key differentiator — threat requires BOTH signals
        threat_interaction = a * (1 - v)  # high when arousal high AND valence low
        threat = (threat_interaction * 0.6 + (1 - s) * 0.15 + (1 - so) * 0.15 + ic * 0.1 + obs_threat_boost)
        hypotheses.append(IntentHypothesis(
            label="threatening",
            probability=threat * conf,
            supporting_pillars=[4, 0, 12],
            pillar_deltas=[0.20 * threat, 0.15 * threat, 0.10 * threat],
        ))
        
        # Fatigued: high fatigue + low arousal + low attention
        fatigued = (f * 0.4 + (1 - a) * 0.3 + (1 - at) * 0.3)
        hypotheses.append(IntentHypothesis(
            label="fatigued",
            probability=fatigued * conf,
            supporting_pillars=[8, 0],
            pillar_deltas=[0.05 * fatigued, 0.05 * fatigued],
        ))
        
        # Social engagement: high social openness + attention + moderate arousal
        social = (so * 0.35 + at * 0.25 + v * 0.2 + (1 - f) * 0.2)
        hypotheses.append(IntentHypothesis(
            label="social_engagement",
            probability=social * conf,
            supporting_pillars=[7, 8, 9, 3],
            pillar_deltas=[0.20 * social, 0.10 * social,
                           0.10 * social, 0.05 * social],
        ))
        
        # Normalize so probabilities sum to 1
        total = sum(h.probability for h in hypotheses)
        if total > 1e-6:
            for h in hypotheses:
                h.probability /= total
        
        # Sort by probability
        hypotheses.sort(key=lambda h: h.probability, reverse=True)
        
        return hypotheses
    
    def predict_next(self, dt: float = 1.0) -> HSVState:
        """Predict what the state will be after dt time.
        
        Without external forces, state decays toward neutral prior (0.5).
        Models the assumption that emotional states are temporary.
        """
        predicted = HSVState()
        decay_rate = 0.05 * dt
        
        for dim_name in HSV_DIM_NAMES:
            src = getattr(self._state, dim_name)
            dst = getattr(predicted, dim_name)
            
            # Mean decays toward 0.5 (neutral)
            dst.mean = src.mean + decay_rate * (0.5 - src.mean)
            dst.mean = float(np.clip(dst.mean, 0.0, 1.0))
            
            # Variance grows (uncertainty increases with prediction)
            dst.variance = min(0.5, src.variance + decay_rate * 0.1)
        
        return predicted
    
    def get_history(self) -> list[HSVState]:
        return list(self._history)
    
    def reset(self) -> None:
        """Reset to uninformative prior."""
        self.__init__()
