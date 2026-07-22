"""Habit Formation — P7.7

Repeated action sequences become automated (lower cost, faster execution).

Core idea: the agent transitions from deliberate planning to habitual
response for frequently-encountered situations. A habit is a compiled
action sequence triggered by context without explicit planning.

This creates:
- Efficiency: habits bypass expensive planning
- Speed: pre-computed action sequences execute faster
- Vulnerability: habits are context-insensitive (may act inappropriately
  when the situation changes but the context matches)
- Learning: habits emerge from repeated successful action sequences

Architecture:
- HabitTracker monitors action history for repeated sequences
- Habit represents a compiled sequence triggered by context similarity
- When context matches a known habit, skip planning and execute directly
- Habits have strength (repetition count) and recency (last used)
- Habits can decay if not used

Usage:
    hf = HabitFormation()
    hf.record(context=current_state, actions=[action1, action2, action3], success=True)
    habit = hf.check_context(current_state)
    if habit:
        return habit.actions  # skip planning
    else:
        plan = planner.plan(intent)
        hf.record(context=current_state, actions=plan.actions, success=outcome)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
from collections import defaultdict


@dataclass
class HabitAction:
    """A single action in a habit sequence."""
    action_type: str          # e.g. "approach", "observe", "communicate"
    target_pillar: int        # which pillar to modify
    magnitude: float          # how much to modify
    duration: int = 1         # how many ticks to maintain


@dataclass
class Habit:
    """A compiled action sequence triggered by context."""
    habit_id: int
    name: str                 # human-readable name

    # Trigger: what context activates this habit
    context_signature: np.ndarray  # compressed context representation
    context_tolerance: float = 0.3  # how similar context must be

    # Actions: what to do when triggered
    actions: list[HabitAction] = field(default_factory=list)

    # Strength: how established this habit is
    strength: float = 0.0     # 0-1, increases with repetition
    use_count: int = 0
    success_count: int = 0
    last_used_tick: int = 0

    # Efficiency: how much faster than planning
    speedup_factor: float = 1.0  # 1.0 = same as planning, 2.0 = twice as fast

    @property
    def success_rate(self) -> float:
        return self.success_count / max(1, self.use_count)

    @property
    def is_established(self) -> bool:
        """Habit is established if used 5+ times with >70% success."""
        return self.use_count >= 5 and self.success_rate > 0.7


@dataclass
class HabitFormationConfig:
    """Configuration for habit formation."""
    context_dim: int = 16          # dimensionality of context
    sequence_window: int = 5       # action sequences to track
    min_repetitions: int = 3       # before considering a habit
    habit_strength_increment: float = 0.1
    habit_decay_rate: float = 0.001  # per tick
    max_habits: int = 50
    context_tolerance: float = 0.3
    planning_cost: float = 1.0     # relative cost of planning
    habit_cost: float = 0.2        # relative cost of executing habit


class HabitFormation:
    """Detects and executes habitual action sequences.

    Monitors the agent's action history for repeated sequences
    in similar contexts. When a sequence is repeated enough times
    with sufficient success, it becomes a habit — a compiled
    action sequence that can be executed without planning.

    This is the cognitive shortcut that enables efficient behavior
    in familiar situations, at the cost of flexibility.

    Usage:
        hf = HabitFormation()

        # Each tick, record what happened
        hf.record(context=state, actions=[a1, a2], success=True, tick=t)

        # Before planning, check if a habit applies
        habit = hf.check_context(state)
        if habit:
            return habit.actions  # fast path

        # Otherwise, plan normally
        plan = planner.plan(intent)
        hf.record(context=state, actions=plan.actions, success=outcome)
    """

    def __init__(self, config: Optional[HabitFormationConfig] = None):
        self.config = config or HabitFormationConfig()
        self._habits: dict[int, Habit] = {}
        self._next_habit_id = 0
        self._action_buffer: list[tuple[np.ndarray, list[HabitAction], bool]] = []
        self._context_action_history: list[tuple[np.ndarray, str]] = []

    def record(self, context: np.ndarray,
               actions: list[dict],
               success: bool,
               tick: int = 0) -> None:
        """Record an action sequence and its outcome.

        Args:
            context: current state when actions were taken
            actions: list of action dicts with keys: type, pillar, magnitude
            success: whether the sequence succeeded
            tick: current tick number
        """
        context = np.asarray(context, dtype=np.float64)

        # Convert actions to HabitAction
        habit_actions = []
        for a in actions:
            habit_actions.append(HabitAction(
                action_type=a.get("type", "unknown"),
                target_pillar=a.get("pillar", 0),
                magnitude=a.get("magnitude", 0.1),
                duration=a.get("duration", 1),
            ))

        if not habit_actions:
            return

        # Add to buffer
        self._action_buffer.append((context.copy(), habit_actions, success))
        if len(self._action_buffer) > self.config.sequence_window * 10:
            self._action_buffer.pop(0)

        # Track context-action pairs
        for a in habit_actions:
            self._context_action_history.append((context.copy(), a.action_type))

        # Look for repeated sequences
        self._detect_habits(tick)

        # Decay existing habits
        for habit in self._habits.values():
            habit.strength *= (1 - self.config.habit_decay_rate)

    def check_context(self, context: np.ndarray) -> Optional[Habit]:
        """Check if current context matches a known habit.

        Returns the strongest matching habit, or None if no match.
        """
        context = np.asarray(context, dtype=np.float64)

        best_habit = None
        best_similarity = 0.0

        for habit in self._habits.values():
            if not habit.is_established:
                continue

            # Compute context similarity
            similarity = self._context_similarity(context, habit.context_signature)

            if similarity > (1.0 - habit.context_tolerance):
                score = similarity * habit.strength
                if score > best_similarity:
                    best_similarity = score
                    best_habit = habit

        return best_habit

    def execute_habit(self, habit: Habit, tick: int = 0) -> list[dict]:
        """Execute a habit's action sequence.

        Returns list of action dicts ready for the planner/controller.
        """
        habit.use_count += 1
        habit.last_used_tick = tick

        result = []
        for a in habit.actions:
            if isinstance(a, dict):
                result.append({
                    "type": a.get("type", "unknown"),
                    "pillar": a.get("pillar", 0),
                    "magnitude": a.get("magnitude", 0.1),
                    "duration": a.get("duration", 1),
                    "source": "habit",
                    "habit_id": habit.habit_id,
                })
            else:
                result.append({
                    "type": a.action_type,
                    "pillar": a.target_pillar,
                    "magnitude": a.magnitude,
                    "duration": a.duration,
                    "source": "habit",
                    "habit_id": habit.habit_id,
                })
        return result

    def report_outcome(self, habit: Habit, success: bool) -> None:
        """Report whether a habit execution succeeded."""
        if success:
            habit.success_count += 1
            habit.strength = min(1.0, habit.strength + self.config.habit_strength_increment)
        else:
            habit.strength = max(0.0, habit.strength - self.config.habit_strength_increment * 2)

    def cost_comparison(self, habit: Habit) -> dict:
        """Compare habit execution cost vs planning cost."""
        habit_cost = self.config.habit_cost * len(habit.actions)
        plan_cost = self.config.planning_cost * len(habit.actions)
        return {
            "habit_cost": habit_cost,
            "planning_cost": plan_cost,
            "savings": plan_cost - habit_cost,
            "speedup": plan_cost / max(habit_cost, 0.01),
        }

    def get_action_preferences(self, context: np.ndarray) -> dict[str, float]:
        """Get action type preferences based on history in similar contexts.

        Returns dict of action_type -> preference score.
        """
        context = np.asarray(context, dtype=np.float64)
        preferences = defaultdict(float)
        count = 0

        for hist_context, action_type in self._context_action_history:
            similarity = self._context_similarity(context, hist_context)
            if similarity > 0.7:
                preferences[action_type] += similarity
                count += 1

        if count > 0:
            for k in preferences:
                preferences[k] /= count

        return dict(preferences)

    def _detect_habits(self, tick: int) -> None:
        """Scan action buffer for repeated sequences."""
        if len(self._action_buffer) < self.config.min_repetitions:
            return

        # Look at recent sequences of length 2-4
        for seq_len in range(2, min(5, len(self._action_buffer) // 2 + 1)):
            sequences = {}
            for i in range(len(self._action_buffer) - seq_len + 1):
                _, actions, success = self._action_buffer[i]
                # Create sequence signature
                sig = tuple(a.action_type for a in actions[:seq_len])
                context = self._action_buffer[i][0]

                if sig not in sequences:
                    sequences[sig] = []
                sequences[sig].append((i, context, actions[:seq_len], success))

            # Check for repetitions
            for sig, instances in sequences.items():
                if len(instances) < self.config.min_repetitions:
                    continue

                # Check context similarity
                contexts = [inst[1] for inst in instances]
                avg_context = np.mean(contexts, axis=0)
                max_spread = max(
                    np.linalg.norm(c - avg_context) for c in contexts)

                if max_spread > self.config.context_tolerance * 2:
                    continue  # contexts too different

                # Check success rate
                successes = sum(1 for inst in instances if inst[3])
                success_rate = successes / len(instances)

                if success_rate < 0.5:
                    continue

                # Create or strengthen habit
                existing = self._find_matching_habit(sig, avg_context)
                if existing:
                    existing.strength = min(
                        1.0, existing.strength + self.config.habit_strength_increment)
                    existing.use_count = len(instances)
                    existing.success_count = successes
                else:
                    if len(self._habits) < self.config.max_habits:
                        self._create_habit(sig, avg_context, instances, tick)

    def _create_habit(self, sig: tuple, context: np.ndarray,
                      instances: list, tick: int) -> Habit:
        """Create a new habit from detected pattern."""
        habit_id = self._next_habit_id
        self._next_habit_id += 1

        # Get actions from first instance
        _, _, first_actions, _ = instances[0]

        # Compute average context
        avg_context = np.mean([inst[1] for inst in instances], axis=0)

        habit = Habit(
            habit_id=habit_id,
            name=f"habit_{habit_id}_{'_'.join(sig)}",
            context_signature=avg_context,
            context_tolerance=self.config.context_tolerance,
            actions=first_actions,
            strength=0.3 + 0.1 * len(instances),
            use_count=len(instances),
            success_count=sum(1 for inst in instances if inst[3]),
            last_used_tick=tick,
        )

        self._habits[habit_id] = habit
        return habit

    def _find_matching_habit(self, sig: tuple,
                             context: np.ndarray) -> Optional[Habit]:
        """Find existing habit matching signature and context."""
        for habit in self._habits.values():
            if habit.actions and isinstance(habit.actions[0], dict):
                habit_sig = tuple(a.get("type", "unknown") for a in habit.actions)
            else:
                habit_sig = tuple(a.action_type for a in habit.actions)
            if habit_sig == sig:
                similarity = self._context_similarity(context, habit.context_signature)
                if similarity > 0.7:
                    return habit
        return None

    def _context_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute context similarity (0-1, 1=identical)."""
        dist = np.linalg.norm(a - b)
        return float(np.exp(-dist / 0.5))

    def get_all_habits(self) -> list[Habit]:
        """Get all tracked habits."""
        return list(self._habits.values())

    def get_established_habits(self) -> list[Habit]:
        """Get only established habits."""
        return [h for h in self._habits.values() if h.is_established]

    def stats(self) -> dict:
        """Summary statistics."""
        habits = list(self._habits.values())
        established = [h for h in habits if h.is_established]
        return {
            "n_habits": len(habits),
            "n_established": len(established),
            "avg_strength": np.mean([h.strength for h in habits]) if habits else 0,
            "avg_success_rate": np.mean([h.success_rate for h in habits]) if habits else 0,
            "buffer_size": len(self._action_buffer),
        }
