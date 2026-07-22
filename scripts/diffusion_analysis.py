#!/usr/bin/env python3
"""
Diffusion Tensor Structuring Analysis.

Feeds correlated observation data into the DiffusionTensor and tracks
how coupling structure emerges over time. BCFVT predicts that observation
correlations should cause D_ij to develop non-trivial structure.
"""
import sys, os, json
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from substrate_echo.dynamics.diffusion import DiffusionTensor
from substrate_echo.dynamics.field_evolution import FieldEvolver, FieldConfig

PILLAR_NAMES = [
    "Awareness", "Willpower", "Force", "Influence",
    "Resistance", "Integrity", "Cohesion", "Relation",
    "Presence", "Warmth", "Memory", "Attraction",
    "Harm", "Distortion", "Flux", "Depth",
]

# Pillar groups for generating correlated data
GROUPS = {
    'cognitive':  [0, 1, 2, 3],     # Awareness, Willpower, Force, Influence
    'structural': [4, 5, 6, 7],     # Resistance, Integrity, Cohesion, Relation
    'emotional':  [8, 9, 10, 11],   # Presence, Warmth, Memory, Attraction
    'metaphoric': [12, 13, 14, 15], # Harm, Distortion, Flux, Depth
}


def generate_correlated_observations(n_obs=500, noise=0.1):
    """Generate observations where pillars within groups are correlated."""
    observations = []

    for _ in range(n_obs):
        # Pick a random group to activate
        group_name = np.random.choice(list(GROUPS.keys()))
        group_indices = GROUPS[group_name]

        # Base state: small random values
        obs = np.random.randn(16) * noise

        # Activate the selected group with correlated signal
        signal = np.random.randn() * 0.5
        for idx in group_indices:
            obs[idx] += signal + np.random.randn() * noise * 0.3

        # Occasionally activate cross-group (inter-group coupling)
        if np.random.random() < 0.2:
            other_group = np.random.choice(
                [g for g in GROUPS if g != group_name]
            )
            signal2 = np.random.randn() * 0.2
            for idx in GROUPS[other_group]:
                obs[idx] += signal2 + np.random.randn() * noise * 0.5

        observations.append(obs)

    return observations


def analyze_structuring():
    print("=" * 60)
    print("DIFFUSION TENSOR STRUCTURING ANALYSIS")
    print("=" * 60)

    diff = DiffusionTensor()

    # Generate correlated observations
    observations = generate_correlated_observations(n_obs=500, noise=0.1)

    # Track tensor evolution
    snapshot_interval = 50
    snapshots = []
    metrics_over_time = []

    for i, obs in enumerate(observations):
        diff.update_from_observation(obs)

        if (i + 1) % snapshot_interval == 0:
            tensor_copy = diff.tensor.copy()

            # Compute metrics
            off_diag = tensor_copy.copy()
            np.fill_diagonal(off_diag, 0)

            frobenius = float(np.linalg.norm(off_diag))
            max_coupling = float(np.max(off_diag))
            sparsity = float(np.sum(np.abs(off_diag) < 1e-6) / off_diag.size)

            # Block structure: how much coupling is within-group vs between-group
            within_group = 0.0
            between_group = 0.0
            within_count = 0
            between_count = 0

            for g_name, g_indices in GROUPS.items():
                for i_idx in g_indices:
                    for j_idx in g_indices:
                        if i_idx != j_idx:
                            within_group += abs(tensor_copy[i_idx, j_idx])
                            within_count += 1

            all_indices = list(range(16))
            for i_idx in all_indices:
                for j_idx in all_indices:
                    if i_idx != j_idx:
                        i_group = None
                        j_group = None
                        for g_name, g_indices in GROUPS.items():
                            if i_idx in g_indices:
                                i_group = g_name
                            if j_idx in g_indices:
                                j_group = g_name
                        if i_group != j_group:
                            between_group += abs(tensor_copy[i_idx, j_idx])
                            between_count += 1

            avg_within = within_group / max(within_count, 1)
            avg_between = between_group / max(between_count, 1)
            block_ratio = avg_within / max(avg_between, 1e-10)

            snapshots.append({
                'step': i + 1,
                'frobenius': frobenius,
                'max_coupling': max_coupling,
                'sparsity': sparsity,
                'block_ratio': block_ratio,
                'avg_within': avg_within,
                'avg_between': avg_between,
            })

            metrics_over_time.append({
                'step': i + 1,
                'frobenius': frobenius,
                'block_ratio': block_ratio,
            })

            if (i + 1) % 200 == 0:
                print(f"  Step {i+1:>4d}: Frobenius={frobenius:.4f} "
                      f"max={max_coupling:.4f} sparsity={sparsity:.2f} "
                      f"block_ratio={block_ratio:.2f}")

    # Final analysis
    final_tensor = diff.tensor
    strongest = diff.get_strongest_couplings(top_k=10)

    print(f"\n--- Strongest Couplings ---")
    for name_i, name_j, strength in strongest:
        print(f"  {name_i:>12s} <-> {name_j:<12s}: {strength:.4f}")

    print(f"\n--- Block Structure ---")
    for g_name, g_indices in GROUPS.items():
        within = []
        between = []
        for i in g_indices:
            for j in g_indices:
                if i != j:
                    within.append(abs(final_tensor[i, j]))
        for i in g_indices:
            for j in range(16):
                if j not in g_indices:
                    between.append(abs(final_tensor[i, j]))
        print(f"  {g_name:>12s}: within={np.mean(within):.4f} "
              f"between={np.mean(between):.4f} "
              f"ratio={np.mean(within)/max(np.mean(between), 1e-10):.2f}")

    print(f"\n--- Emergence Metrics ---")
    if snapshots:
        first = snapshots[0]
        last = snapshots[-1]
        print(f"  Frobenius norm: {first['frobenius']:.4f} -> {last['frobenius']:.4f} "
              f"({last['frobenius']/max(first['frobenius'], 1e-10):.2f}x)")
        print(f"  Block ratio:    {first['block_ratio']:.4f} -> {last['block_ratio']:.4f} "
              f"({last['block_ratio']/max(first['block_ratio'], 1e-10):.2f}x)")
        print(f"  Sparsity:       {first['sparsity']:.2f} -> {last['sparsity']:.2f}")

    # Save
    out = os.path.join(os.path.dirname(__file__), '..', 'diffusion_analysis.json')
    with open(out, 'w') as f:
        json.dump({
            'snapshots': snapshots,
            'strongest_couplings': [(a, b, float(c)) for a, b, c in strongest],
            'final_tensor': final_tensor.tolist(),
        }, f, indent=2)
    print(f"\nResults saved to {out}")

    return snapshots, strongest


if __name__ == '__main__':
    analyze_structuring()
