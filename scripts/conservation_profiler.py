"""Conservation Profiler v2 — check conservation at natural state boundaries.

One check per tick, comparing post-projection state to previous tick's state.
This eliminates false positives from internal corrections.
"""

from __future__ import annotations
import numpy as np
import time
import json
from dataclasses import dataclass, field

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def profile_conservation_v2(num_ticks: int = 200) -> dict:
    """Run scenario with clean conservation profiling."""
    from substrate_echo.dynamics.field_evolution import FieldEvolver, FieldConfig, SolverType
    from substrate_echo.dynamics.conservation import ConservationFramework
    from substrate_echo.dynamics.pillar_coupling import PillarCoupling
    from substrate_echo.core.cognitive_agents import AgentEcology
    from substrate_echo.core.attractor_memory import AttractorMemory
    from substrate_echo.core.ontological_field import OntologicalField
    from substrate_echo.models.experience import Experience, ExperienceType
    from substrate_echo.core.spatial_world import SpatialWorldModel
    
    field_config = FieldConfig(dt=0.1, gamma=0.01, lambda_gl=1.0, temperature=0.01)
    field_evolver = FieldEvolver(field_config, solver_type=SolverType.CRANK_NICOLSON)
    conservation = ConservationFramework(enabled=True)
    pillar_coupling = PillarCoupling()
    
    ontological_field = OntologicalField()
    memory = AttractorMemory(field=ontological_field)
    agent_ecology = AgentEcology()
    world_model = SpatialWorldModel()
    
    field_state = np.full(16, 0.5)
    pillar_state = np.full(16, 0.5)
    
    # Run one projection to get the baseline norm
    pillar_state = pillar_coupling.project_to_pillars(field_state)
    conservation.set_norm_target(float(np.linalg.norm(pillar_state)))
    
    # Track per-tick norms and energies
    norm_history = [float(np.linalg.norm(pillar_state))]
    energy_history = [field_evolver.compute_energy(field_state)]
    
    violations = {
        "norm": [],
        "energy": [],
        "bounds": [],
    }
    
    for tick in range(num_ticks):
        # Field evolution
        rhs = field_evolver.rhs(field_state, 0.1)
        field_state = field_state + 0.1 * rhs
        field_state = np.clip(field_state, 0.0, 1.0)
        
        # Projection
        new_pillars = pillar_coupling.project_to_pillars(field_state)
        
        # Memory encoding
        if tick % 10 == 0:
            try:
                exp = Experience(
                    experience_id=f"exp_{tick}",
                    experience_type=ExperienceType.LEARNING,
                    psv_snapshot=new_pillars.tolist(),
                    description=f"Tick {tick}",
                )
                memory.encode(exp)
            except Exception:
                pass
        
        # Agent ecology
        responses = agent_ecology.tick(
            new_pillars, world_model=world_model, memory=memory,
        )
        
        # Update state
        pillar_state = new_pillars
        
        # Record
        norm = float(np.linalg.norm(pillar_state))
        energy = field_evolver.compute_energy(field_state)
        norm_history.append(norm)
        energy_history.append(energy)
        
        # Check norm drift
        if len(norm_history) >= 2:
            norm_drift = abs(norm_history[-1] - norm_history[-2])
            if norm_drift > 1e-6:
                violations["norm"].append({
                    "tick": tick,
                    "drift": norm_drift,
                    "from": norm_history[-2],
                    "to": norm_history[-1],
                })
        
        # Check energy stability (non-increasing)
        if len(energy_history) >= 2:
            energy_change = energy_history[-1] - energy_history[-2]
            if energy_change > 0.01:  # 1% tolerance
                violations["energy"].append({
                    "tick": tick,
                    "change": energy_change,
                    "from": energy_history[-2],
                    "to": energy_history[-1],
                })
        
        # Check bounds
        if np.any(pillar_state < 0) or np.any(pillar_state > 1):
            violations["bounds"].append({"tick": tick})
        
        if tick % 50 == 0:
            print(f"  Tick {tick:>4d}: norm={norm:.6f}, energy={energy:.4f}")
    
    # Compute stats
    total_checks = num_ticks * 3  # norm + energy + bounds per tick
    total_violations = sum(len(v) for v in violations.values())
    total_passed = total_checks - total_violations
    
    norm_drifts = [v["drift"] for v in violations["norm"]]
    energy_changes = [v["change"] for v in violations["energy"]]
    
    results = {
        "total_ticks": num_ticks,
        "total_checks": total_checks,
        "total_passed": total_passed,
        "total_failed": total_violations,
        "pass_rate": total_passed / max(1, total_checks),
        "violations": violations,
        "norm_stats": {
            "mean": float(np.mean(norm_history)),
            "std": float(np.std(norm_history)),
            "min": float(np.min(norm_history)),
            "max": float(np.max(norm_history)),
            "drift_count": len(violations["norm"]),
            "avg_drift": float(np.mean(norm_drifts)) if norm_drifts else 0,
            "max_drift": float(np.max(norm_drifts)) if norm_drifts else 0,
        },
        "energy_stats": {
            "mean": float(np.mean(energy_history)),
            "std": float(np.std(energy_history)),
            "min": float(np.min(energy_history)),
            "max": float(np.max(energy_history)),
            "instability_count": len(violations["energy"]),
            "avg_increase": float(np.mean(energy_changes)) if energy_changes else 0,
        },
        "bounds_violations": len(violations["bounds"]),
    }
    
    return results


def print_results(results: dict) -> None:
    print(f"\n{'='*60}")
    print(f"CONSERVATION PROFILE v2: {results['total_ticks']} ticks")
    print(f"{'='*60}")
    
    print(f"\nOverall: {results['total_passed']}/{results['total_checks']} passed "
          f"({results['pass_rate']:.1%})")
    
    ns = results["norm_stats"]
    print(f"\n--- Norm Conservation ---")
    print(f"  Mean: {ns['mean']:.6f}")
    print(f"  Std:  {ns['std']:.6f}")
    print(f"  Drifts: {ns['drift_count']}, avg={ns['avg_drift']:.2e}, max={ns['max_drift']:.2e}")
    
    es = results["energy_stats"]
    print(f"\n--- Energy Stability ---")
    print(f"  Mean: {es['mean']:.6f}")
    print(f"  Std:  {es['std']:.6f}")
    print(f"  Increases: {es['instability_count']}, avg={es['avg_increase']:.4f}")
    
    print(f"\n--- Bounds ---")
    print(f"  Violations: {results['bounds_violations']}")
    
    # Root cause analysis
    print(f"\n--- Root Cause Analysis ---")
    if results["violations"]["norm"]:
        worst = max(results["violations"]["norm"], key=lambda x: x["drift"])
        print(f"  Worst norm drift at tick {worst['tick']}: {worst['drift']:.2e}")
        print(f"    From {worst['from']:.6f} to {worst['to']:.6f}")
    
    if results["violations"]["energy"]:
        worst = max(results["violations"]["energy"], key=lambda x: x["change"])
        print(f"  Worst energy increase at tick {worst['tick']}: {worst['change']:.4f}")


if __name__ == "__main__":
    print("Running conservation profiler v2 (200 ticks)...")
    results = profile_conservation_v2(num_ticks=200)
    print_results(results)
    
    with open("conservation_profile_v2.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nProfile saved to conservation_profile_v2.json")
