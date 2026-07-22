"""Field Renderer — text-based visualization of ontological field state."""

from __future__ import annotations
from typing import Optional
import numpy as np

PILLAR_NAMES = [
    "Awareness", "Willpower", "Force", "Influence",
    "Resistance", "Integrity", "Cohesion", "Relation",
    "Presence", "Warmth", "Memory", "Attraction",
    "Harm", "Distortion", "Flux", "Depth",
]

BAR_WIDTH = 30


class FieldRenderer:
    """Text-based renderer for ontological field state.
    
    Renders 16D state as color-coded bar charts suitable for
    terminal output or logging.
    """
    
    def __init__(self, bar_width: int = BAR_WIDTH):
        self.bar_width = bar_width
    
    def render_state(self, state: np.ndarray, title: str = "Field State") -> str:
        """Render state vector as horizontal bar chart."""
        lines = [f"=== {title} ==="]
        
        for i, name in enumerate(PILLAR_NAMES):
            val = float(state[i]) if i < len(state) else 0.0
            filled = int(val * self.bar_width)
            bar = "█" * filled + "░" * (self.bar_width - filled)
            lines.append(f"  {name:>12s} [{bar}] {val:.3f}")
        
        return "\n".join(lines)
    
    def render_attractors(self, attractors: list, title: str = "Attractors") -> str:
        """Render attractor positions."""
        lines = [f"=== {title} ({len(attractors)} total) ==="]
        
        for att in attractors[:10]:  # show top 10
            center = np.array(att.center) if hasattr(att, 'center') else np.zeros(16)
            dominant = int(np.argmax(center)) if len(center) > 0 else 0
            strength = getattr(att, 'strength', 0.0)
            label = getattr(att, 'label', '?')
            
            lines.append(
                f"  {label:>20s} | strength={strength:.2f} "
                f"| dominant={PILLAR_NAMES[dominant]}"
            )
        
        if len(attractors) > 10:
            lines.append(f"  ... and {len(attractors) - 10} more")
        
        return "\n".join(lines)
    
    def render_diffusion(self, tensor: np.ndarray, title: str = "Diffusion Tensor") -> str:
        """Render diffusion tensor as a compact heatmap."""
        lines = [f"=== {title} ==="]
        lines.append("     " + " ".join(f"{n[:3]:>3s}" for n in PILLAR_NAMES))
        
        for i in range(min(16, len(tensor))):
            row = tensor[i] if i < len(tensor) else np.zeros(16)
            cells = []
            for j in range(min(16, len(row))):
                val = float(row[j])
                if val > 0.5:
                    cells.append("██")
                elif val > 0.1:
                    cells.append("▒▒")
                elif val > 0.01:
                    cells.append("░░")
                else:
                    cells.append("  ")
            lines.append(f"{PILLAR_NAMES[i][:3]:>3s} " + " ".join(cells))
        
        return "\n".join(lines)
    
    def render_trajectory(self, trajectory: list[np.ndarray], title: str = "State Trajectory") -> str:
        """Render state trajectory as sparkline per pillar."""
        if not trajectory:
            return f"=== {title}: empty ==="
        
        lines = [f"=== {title} ({len(trajectory)} steps) ==="]
        
        spark_chars = " ▁▂▃▄▅▆▇█"
        
        for i, name in enumerate(PILLAR_NAMES):
            values = [float(t[i]) if i < len(t) else 0.0 for t in trajectory]
            sparkline = ""
            for v in values[-20:]:  # last 20 steps
                idx = min(int(v * (len(spark_chars) - 1)), len(spark_chars) - 1)
                sparkline += spark_chars[idx]
            lines.append(f"  {name:>12s} {sparkline}")
        
        return "\n".join(lines)
    
    def render_vortices(self, vortices: list, title: str = "Vortex State") -> str:
        """Render vortex positions and winding numbers."""
        lines = [f"=== {title} ({len(vortices)} vortices) ==="]
        
        if not vortices:
            lines.append("  No vortices detected")
            return "\n".join(lines)
        
        for v in vortices[:10]:
            pos = getattr(v, 'position', [0, 0])
            winding = getattr(v, 'winding_number', 0)
            energy = getattr(v, 'energy', 0)
            age = getattr(v, 'age', 0)
            
            sign = "+" if winding > 0 else ""
            lines.append(
                f"  Vortex #{getattr(v, 'id', '?'):>3d} | "
                f"pos=({pos[0]:.2f},{pos[1]:.2f}) | "
                f"W={sign}{winding} | "
                f"E={energy:.3f} | "
                f"age={age:.2f}s"
            )
        
        if len(vortices) > 10:
            lines.append(f"  ... and {len(vortices) - 10} more")
        
        return "\n".join(lines)
    
    def render_conservation(self, results: list, title: str = "Conservation Status") -> str:
        """Render conservation check results."""
        lines = [f"=== {title} ==="]
        
        if not results:
            lines.append("  No conservation checks run")
            return "\n".join(lines)
        
        passed = sum(1 for r in results if r.passed)
        total = len(results)
        
        lines.append(f"  Passed: {passed}/{total}")
        
        for r in results:
            status = "✓" if r.passed else "✗"
            lines.append(
                f"  {status} {r.law_name:>20s} | "
                f"val={r.measured_value:.6f} | "
                f"dev={r.deviation:.2e}"
            )
        
        return "\n".join(lines)
    
    def render_social_field(self, agents: dict, title: str = "Social Field") -> str:
        """Render multi-agent social field state."""
        lines = [f"=== {title} ({len(agents)} agents) ==="]
        
        if not agents:
            lines.append("  No agents in social field")
            return "\n".join(lines)
        
        for agent_id, profile in agents.items():
            role = getattr(profile, 'role', 'unknown')
            rep = getattr(profile, 'reputation', 0)
            state = getattr(profile, 'state', np.zeros(16))
            
            # Dominant pillar
            dominant = PILLAR_NAMES[int(np.argmax(state))] if len(state) > 0 else "?"
            
            lines.append(
                f"  {agent_id:>15s} | "
                f"role={role:>12s} | "
                f"rep={rep:.2f} | "
                f"dominant={dominant}"
            )
        
        return "\n".join(lines)
    
    def render_physics_stats(self, stats: dict, title: str = "Physics Stats") -> str:
        """Render physics integration statistics."""
        lines = [f"=== {title} ==="]
        
        lines.append(f"  Ticks: {stats.get('tick_count', 0)}")
        lines.append(f"  Entities: {stats.get('entity_count', 0)}")
        lines.append(f"  Avg tick: {stats.get('avg_tick_ms', 0):.3f} ms")
        lines.append(f"  Max tick: {stats.get('max_tick_ms', 0):.3f} ms")
        lines.append(f"  Conservation violations: {stats.get('conservation_violations', 0)}")
        
        return "\n".join(lines)
    
    def render_full_dashboard(self, field_state: np.ndarray,
                               attractors: list = None,
                               vortices: list = None,
                               agents: dict = None,
                               conservation: list = None,
                               physics_stats: dict = None) -> str:
        """Render a complete dashboard of all Substrate_Echo state."""
        sections = []
        
        # Field state
        sections.append(self.render_state(field_state))
        
        # Attractors
        if attractors:
            sections.append(self.render_attractors(attractors))
        
        # Vortices
        if vortices:
            sections.append(self.render_vortices(vortices))
        
        # Conservation
        if conservation:
            sections.append(self.render_conservation(conservation))
        
        # Social field
        if agents:
            sections.append(self.render_social_field(agents))
        
        # Physics stats
        if physics_stats:
            sections.append(self.render_physics_stats(physics_stats))
        
        return "\n\n".join(sections)
