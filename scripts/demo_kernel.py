"""Demo: Substrate Kernel — Cognitive Plane streaming.

Demonstrates the kernel as a persistent cognitive backend:
  1. Multiple observations flow through the cognitive plane
  2. Attractors self-organize from convergence
  3. Meta-attractors emerge from correlation
  4. Goals influence action generation
  5. The kernel maintains state across ticks
"""
import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from substrate_echo.kernel import (
    SubstrateKernel, KernelConfig,
    Observation, Goal, EmbodimentState,
)
from substrate_echo.kernel.client import SubstrateClient


def generate_trajectory(n_ticks=2000, seed=42):
    rng = np.random.RandomState(seed)
    state = np.full(16, 0.5)
    targets = {
        0: lambda: _make([0, 1, 2, 3], [0.8, 0.7, 0.6, 0.3]),
        1: lambda: _make([0, 1, 4, 5], [0.75, 0.8, 0.5, 0.4]),
        2: lambda: _make([0, 1, 6, 7], [0.7, 0.75, 0.55, 0.35]),
        3: lambda: _make([0, 1, 2, 3, 4, 5, 6, 7], [0.75]*8),
        4: lambda: _make([10, 11, 12, 13], [0.8, 0.7, 0.6, 0.5]),
    }
    for _ in range(n_ticks):
        r = rng.random()
        mode = 0 if r < 0.30 else 1 if r < 0.55 else 2 if r < 0.75 else 3 if r < 0.85 else 4
        target = targets[mode]()
        state = state + (target - state) * 0.12 + rng.randn(16) * 0.015
        state = np.clip(state, 0.0, 1.0)
        yield state.copy(), mode


def _make(indices, values):
    t = np.zeros(16)
    for i, v in zip(indices, values):
        t[i] = v
    return t


def main():
    print("=" * 70)
    print("SUBSTRATE KERNEL — COGNITIVE PLANE DEMO")
    print("=" * 70)

    # Create kernel
    kernel = SubstrateKernel(KernelConfig(dim=16))

    # Two embodiments sharing one kernel
    desktop = SubstrateClient(kernel, embodiment_id="desktop")
    robot = SubstrateClient(kernel, embodiment_id="robot")

    # Register embodiments
    kernel.publish_embodiment_state(EmbodimentState(
        embodiment_id="desktop", embodiment_type="desktop",
        available_modalities=["text", "audio"]))
    kernel.publish_embodiment_state(EmbodimentState(
        embodiment_id="robot", embodiment_type="robot",
        available_modalities=["proprio", "lidar"]))

    # Set a goal from desktop
    desktop.set_goal(
        target=[0.8, 0.7, 0.6, 0.3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        priority=0.5,
        description="Move toward concept A")

    print(f"Embodiments: {list(kernel._embodiments.keys())}")
    print(f"Goals: {len(kernel._goals)}")

    # Stream observations from both embodiments
    print("\n--- Streaming Cognitive Plane ---")
    rng = np.random.RandomState(99)

    for tick, (state, mode) in enumerate(generate_trajectory(2500)):
        # Alternate between embodiments
        if tick % 3 == 0:
            cs = robot.observe(state.tolist(), modality="proprio")
        else:
            cs = desktop.observe(state.tolist(), modality="text")

        if cs.tick % 500 == 0 and cs.tick > 0:
            print(f"  tick {cs.tick:4d}: "
                  f"attractors={cs.n_attractors} "
                  f"meta={cs.n_meta_attractors} "
                  f"coherence={cs.coherence:.3f} "
                  f"balance={cs.basin_balance:.3f} "
                  f"embodiments={cs.active_embodiments} "
                  f"action_conf={cs.action.confidence:.3f}")

    # Final state
    print("\n--- Kernel State (Control Plane) ---")
    snapshot = kernel.get_snapshot()
    for k, v in snapshot.items():
        print(f"  {k}: {v}")

    print("\n--- Embodiments ---")
    for eid, info in kernel.get_embodiments().items():
        print(f"  {eid}: type={info['type']}, active={info['active']}, "
              f"modalities={info['modalities']}")

    print("\n--- Abstraction Events ---")
    for event in kernel.get_abstraction_events():
        print(f"  tick {event['tick']}: meta={event['meta_id']} "
              f"from {event['children']}")

    print("\n--- Topology History ---")
    for snap in kernel.get_topology_history()[-5:]:
        print(f"  tick {snap['tick']}: attractors={snap['attractors']} "
              f"depth={snap['depth']:.4f} entropy={snap['entropy']:.3f}")

    print("\n" + "=" * 70)
    print("DEMO COMPLETE")
    print("Two embodiments shared one cognitive kernel.")
    print("The kernel maintained state, discovered attractors,")
    print("built abstraction hierarchy, and generated actions.")
    print("=" * 70)


if __name__ == "__main__":
    main()
