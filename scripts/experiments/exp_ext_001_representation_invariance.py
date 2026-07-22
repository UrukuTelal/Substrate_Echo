"""EXP-EXT-001: Representation Invariance Test

Question: Does WHT increase invariance to controlled perturbations?

Design:
    1. Generate 1000 synthetic "concepts" as 16D feature vectors
    2. For each concept, create 5 paraphrases (controlled perturbations)
    3. Create adversarial variants (emotional, contradictory, verbose, short)
    4. Measure pairwise distances BEFORE and AFTER WHT
    5. Check: does WHT compress same-concept distances relative to cross-concept?

Key insight: WHT is orthogonal (energy-preserving), so it cannot create or
destroy information. But it CAN redistribute energy across dimensions,
potentially making semantically similar vectors closer in spectral space
than in raw feature space. The question is whether this actually happens
for structured (not random) perturbations.

Metrics:
    - Intra-concept distance: mean distance between variants of same concept
    - Inter-concept distance: mean distance between different concepts
    - Contrast ratio: inter / intra (higher = better separation)
    - Invariance ratio: intra_raw / intra_spectral (higher = WHT helps more)
"""
from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np

_root = str(__import__("pathlib").Path(__file__).resolve().parent.parent)
sys.path.insert(0, _root)
sys.path.insert(0, str(__import__("pathlib").Path(_root).parent / "crowquant"))
from crowquant.core import WHTransform


# ---------------------------------------------------------------------------
# Concept Generation
# ---------------------------------------------------------------------------

PILLAR_NAMES = [
    "awareness", "willpower", "force", "influence", "resistance",
    "integrity", "cohesion", "relation", "presence", "warmth",
    "memory", "attraction", "harm", "distortion", "flux", "depth",
]


@dataclass
class ConceptVariant:
    """A single variant of a concept."""
    concept_id: int
    variant_type: str   # "original", "paraphrase_0..4", "emotional", "contradictory", "verbose", "short"
    vector: np.ndarray  # 16D feature vector
    text_anchor: str    # synthetic text anchor (for logging)


@dataclass
class Concept:
    """A concept with its variants."""
    concept_id: int
    base_vector: np.ndarray
    variants: List[ConceptVariant] = field(default_factory=list)


def generate_concept_vectors(n_concepts: int, dim: int = 16,
                             seed: int = 42) -> np.ndarray:
    """Generate n_concepts random unit vectors in R^dim.

    Each vector represents a 'concept' in the 16D pillar space.
    Vectors are sampled from the unit sphere to ensure diverse directions.
    """
    rng = np.random.default_rng(seed)
    raw = rng.standard_normal((n_concepts, dim))
    norms = np.linalg.norm(raw, axis=1, keepdims=True)
    return raw / (norms + 1e-10)


def make_paraphrase(base: np.ndarray, strength: float,
                    rng: np.random.Generator) -> np.ndarray:
    """Create a paraphrase: same concept, different surface features.

    Paraphrasing = adding structured noise that preserves the dominant
    direction while perturbing secondary components.

    The noise is:
    - Isotropic Gaussian (surface-level variation)
    - Plus a small rotation in the plane perpendicular to the dominant axis
      (structural variation that preserves meaning)
    """
    # Isotropic surface noise
    noise = rng.normal(0, strength, size=base.shape)

    # Structural perturbation: small rotation in a random plane
    v1 = base / (np.linalg.norm(base) + 1e-10)
    perp = rng.standard_normal(base.shape)
    perp = perp - np.dot(perp, v1) * v1  # make orthogonal to base
    perp = perp / (np.linalg.norm(perp) + 1e-10)
    angle = rng.uniform(-0.3, 0.3)  # small rotation
    rotated = base * np.cos(angle) + perp * np.sin(angle)

    # Blend: 70% rotated + 30% noise
    paraphrase = 0.7 * rotated + 0.3 * noise

    # Re-normalize to keep on the unit sphere (approximate)
    paraphrase = paraphrase / (np.linalg.norm(paraphrase) + 1e-10)
    return paraphrase


def make_emotional(base: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Emotional manipulation: amplify high-magnitude components, suppress low.

    Simulates how emotional language amplifies certain features while
    distorting others.
    """
    magnitudes = np.abs(base)
    amplification = 1.0 + 0.5 * magnitudes  # amplify strong features
    suppression = 1.0 - 0.3 * (1.0 - magnitudes)  # suppress weak features
    factor = amplification * suppression
    emotional = base * factor
    emotional = emotional / (np.linalg.norm(emotional) + 1e-10)
    return emotional


def make_contradictory(base: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Contradictory version: flip 2-4 dominant dimensions.

    Simulates how contradictions preserve most structure but negate key claims.
    """
    n_dims = len(base)
    n_flips = rng.integers(2, min(5, n_dims + 1))
    flip_dims = rng.choice(n_dims, size=n_flips, replace=False)
    contradictory = base.copy()
    contradictory[flip_dims] *= -1.0
    contradictory = contradictory / (np.linalg.norm(contradictory) + 1e-10)
    return contradictory


def make_verbose(base: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Verbose version: add low-amplitude noise across all dimensions.

    Simulates how verbose text adds redundant information that dilutes
    the core signal.
    """
    noise = rng.normal(0, 0.15, size=base.shape)
    verbose = base + noise
    verbose = verbose / (np.linalg.norm(verbose) + 1e-10)
    return verbose


def make_short(base: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Short version: keep top-8 dimensions, zero rest.

    Simulates how short text loses nuance but keeps dominant features.
    """
    n_keep = len(base) // 2
    top_dims = np.argsort(np.abs(base))[-n_keep:]
    short = np.zeros_like(base)
    short[top_dims] = base[top_dims]
    norm = np.linalg.norm(short)
    if norm > 1e-10:
        short = short / norm
    return short


def build_concepts(n_concepts: int = 1000, dim: int = 16,
                   n_paraphrases: int = 5, seed: int = 42) -> List[Concept]:
    """Build a full concept bank with all variant types."""
    rng = np.random.default_rng(seed)
    base_vectors = generate_concept_vectors(n_concepts, dim, seed)

    concepts = []
    for cid in range(n_concepts):
        base = base_vectors[cid]
        c = Concept(concept_id=cid, base_vector=base)

        # Original
        c.variants.append(ConceptVariant(
            concept_id=cid, variant_type="original",
            vector=base, text_anchor=f"concept_{cid}_original",
        ))

        # Paraphrases (controlled perturbation, varying strength)
        strengths = [0.1, 0.15, 0.2, 0.25, 0.3]
        for i in range(n_paraphrases):
            strength = strengths[i % len(strengths)]
            p = make_paraphrase(base, strength, rng)
            c.variants.append(ConceptVariant(
                concept_id=cid, variant_type=f"paraphrase_{i}",
                vector=p, text_anchor=f"concept_{cid}_para_{i}",
            ))

        # Emotional manipulation
        c.variants.append(ConceptVariant(
            concept_id=cid, variant_type="emotional",
            vector=make_emotional(base, rng),
            text_anchor=f"concept_{cid}_emotional",
        ))

        # Contradictory
        c.variants.append(ConceptVariant(
            concept_id=cid, variant_type="contradictory",
            vector=make_contradictory(base, rng),
            text_anchor=f"concept_{cid}_contradictory",
        ))

        # Verbose
        c.variants.append(ConceptVariant(
            concept_id=cid, variant_type="verbose",
            vector=make_verbose(base, rng),
            text_anchor=f"concept_{cid}_verbose",
        ))

        # Short
        c.variants.append(ConceptVariant(
            concept_id=cid, variant_type="short",
            vector=make_short(base, rng),
            text_anchor=f"concept_{cid}_short",
        ))

        concepts.append(c)

    return concepts


# ---------------------------------------------------------------------------
# Distance Computation
# ---------------------------------------------------------------------------

def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """1 - cosine_similarity. Range [0, 2]."""
    n1, n2 = np.linalg.norm(a), np.linalg.norm(b)
    if n1 < 1e-12 or n2 < 1e-12:
        return 1.0
    return float(1.0 - np.dot(a, b) / (n1 * n2))


def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Euclidean distance."""
    return float(np.linalg.norm(a - b))


@dataclass
class DistanceMatrix:
    """Pairwise distances between all variants."""
    intra_distances: List[float] = field(default_factory=list)
    inter_distances: List[float] = field(default_factory=list)
    intra_by_type: dict = field(default_factory=dict)
    inter_by_pair: dict = field(default_factory=dict)


def compute_distances(concepts: List[Concept],
                      vectors: np.ndarray,
                      variant_types: List[str]) -> DistanceMatrix:
    """Compute intra- and inter-concept distances.

    Parameters
    ----------
    concepts : list of Concept
        The concept bank.
    vectors : np.ndarray, shape (n_concepts, n_variants_per_concept, dim)
        Pre-extracted vectors for each variant of each concept.
    variant_types : list of str
        The variant type names corresponding to axis 1.
    """
    dm = DistanceMatrix()
    n_concepts = len(concepts)
    n_variants = len(variant_types)

    for i in range(n_concepts):
        for j in range(n_variants):
            for k in range(j + 1, n_variants):
                d = cosine_distance(vectors[i, j], vectors[i, k])
                vtype = f"{variant_types[j]}_vs_{variant_types[k]}"
                dm.intra_distances.append(d)
                if vtype not in dm.intra_by_type:
                    dm.intra_by_type[vtype] = []
                dm.intra_by_type[vtype].append(d)

    rng = np.random.default_rng(99)
    n_pairs = min(20000, n_concepts * (n_concepts - 1) // 2)
    sampled = set()
    while len(sampled) < n_pairs:
        a, b = rng.integers(0, n_concepts, size=2)
        if a != b:
            key = (min(a, b), max(a, b))
            if key not in sampled:
                sampled.add(key)
                vi = rng.integers(0, n_variants)
                vj = rng.integers(0, n_variants)
                d = cosine_distance(vectors[a, vi], vectors[b, vj])
                pair_type = f"{variant_types[vi]}_vs_{variant_types[vj]}"
                dm.inter_distances.append(d)
                if pair_type not in dm.inter_by_pair:
                    dm.inter_by_pair[pair_type] = []
                dm.inter_by_pair[pair_type].append(d)

    return dm


# ---------------------------------------------------------------------------
# WHT Application
# ---------------------------------------------------------------------------

def apply_wht_to_concepts(concepts: List[Concept],
                          seed: int = 42) -> Tuple[np.ndarray, List[str]]:
    """Apply WHT rotation to all concept variant vectors.

    Returns
    -------
    vectors : np.ndarray, shape (n_concepts, n_variants, 16)
        WHT-transformed vectors.
    variant_types : list of str
        Variant type names.
    """
    n_concepts = len(concepts)
    n_variants = len(concepts[0].variants)
    dim = len(concepts[0].base_vector)

    vectors = np.zeros((n_concepts, n_variants, dim))
    variant_types = []

    for i, c in enumerate(concepts):
        for j, v in enumerate(c.variants):
            vectors[i, j] = WHTransform.rotate(v.vector.copy(), seed=seed)[:dim]
        if i == 0:
            variant_types = [v.variant_type for v in c.variants]

    return vectors, variant_types


def extract_raw_concepts(concepts: List[Concept]) -> Tuple[np.ndarray, List[str]]:
    """Extract raw (untransformed) vectors."""
    n_concepts = len(concepts)
    n_variants = len(concepts[0].variants)
    dim = len(concepts[0].base_vector)

    vectors = np.zeros((n_concepts, n_variants, dim))
    variant_types = []

    for i, c in enumerate(concepts):
        for j, v in enumerate(c.variants):
            vectors[i, j] = v.vector
        if i == 0:
            variant_types = [v.variant_type for v in c.variants]

    return vectors, variant_types


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

@dataclass
class ExperimentResult:
    """Results from EXP-EXT-001."""
    n_concepts: int
    n_variants: int
    dim: int

    # Raw space
    raw_intra_mean: float
    raw_inter_mean: float
    raw_contrast: float  # inter / intra
    raw_intra_by_type: dict

    # Spectral space (after WHT)
    spectral_intra_mean: float
    spectral_inter_mean: float
    spectral_contrast: float
    spectral_intra_by_type: dict

    # Invariance ratios
    invariance_ratio: float  # raw_intra / spectral_intra (>1 = WHT helps)
    contrast_ratio: float    # spectral_contrast / raw_contrast (>1 = WHT helps)

    # Per-variant-type analysis
    variant_invariance: dict  # variant_type -> raw_dist / spectral_dist

    # Spectral energy distribution
    energy_before: np.ndarray  # per-dimension energy before WHT
    energy_after: np.ndarray   # per-dimension energy after WHT

    # Timing
    raw_time: float
    spectral_time: float


def analyze(concepts: List[Concept], seed: int = 42) -> ExperimentResult:
    """Run the full analysis."""
    t0 = time.perf_counter()

    # Extract raw vectors
    raw_vectors, variant_types = extract_raw_concepts(concepts)
    raw_dm = compute_distances(concepts, raw_vectors, variant_types)
    raw_time = time.perf_counter() - t0

    # Apply WHT
    t1 = time.perf_counter()
    spectral_vectors, _ = apply_wht_to_concepts(concepts, seed=seed)
    spectral_dm = compute_distances(concepts, spectral_vectors, variant_types)
    spectral_time = time.perf_counter() - t1

    # Energy distribution
    energy_before = np.mean(raw_vectors ** 2, axis=(0, 1))
    energy_after = np.mean(spectral_vectors ** 2, axis=(0, 1))

    # Per-variant invariance
    variant_invariance = {}
    for vtype in ["paraphrase_0", "paraphrase_1", "paraphrase_2",
                   "paraphrase_3", "paraphrase_4", "emotional",
                   "contradictory", "verbose", "short"]:
        raw_key = f"original_vs_{vtype}"
        if raw_key in raw_dm.intra_by_type and raw_key in spectral_dm.intra_by_type:
            r = np.mean(raw_dm.intra_by_type[raw_key])
            s = np.mean(spectral_dm.intra_by_type[raw_key])
            variant_invariance[vtype] = r / (s + 1e-10)

    raw_intra = np.mean(raw_dm.intra_distances) if raw_dm.intra_distances else 0
    raw_inter = np.mean(raw_dm.inter_distances) if raw_dm.inter_distances else 0
    spec_intra = np.mean(spectral_dm.intra_distances) if spectral_dm.intra_distances else 0
    spec_inter = np.mean(spectral_dm.inter_distances) if spectral_dm.inter_distances else 0

    return ExperimentResult(
        n_concepts=len(concepts),
        n_variants=len(concepts[0].variants),
        dim=len(concepts[0].base_vector),
        raw_intra_mean=raw_intra,
        raw_inter_mean=raw_inter,
        raw_contrast=raw_inter / (raw_intra + 1e-10),
        raw_intra_by_type={k: np.mean(v) for k, v in raw_dm.intra_by_type.items()},
        spectral_intra_mean=spec_intra,
        spectral_inter_mean=spec_inter,
        spectral_contrast=spec_inter / (spec_intra + 1e-10),
        spectral_intra_by_type={k: np.mean(v) for k, v in spectral_dm.intra_by_type.items()},
        invariance_ratio=raw_intra / (spec_intra + 1e-10),
        contrast_ratio=(spec_inter / (spec_intra + 1e-10)) / (raw_inter / (raw_intra + 1e-10) + 1e-10),
        variant_invariance=variant_invariance,
        energy_before=energy_before,
        energy_after=energy_after,
        raw_time=raw_time,
        spectral_time=spectral_time,
    )


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_report(r: ExperimentResult) -> None:
    """Print human-readable report."""
    print("=" * 70)
    print("EXP-EXT-001: Representation Invariance Test")
    print("=" * 70)
    print(f"  Concepts: {r.n_concepts}")
    print(f"  Variants per concept: {r.n_variants}")
    print(f"  Dimension: {r.dim}")
    print()

    print("RAW SPACE (before WHT)")
    print("-" * 40)
    print(f"  Intra-concept distance:  {r.raw_intra_mean:.6f}")
    print(f"  Inter-concept distance:  {r.raw_inter_mean:.6f}")
    print(f"  Contrast ratio:          {r.raw_contrast:.4f}")
    print()

    print("SPECTRAL SPACE (after WHT)")
    print("-" * 40)
    print(f"  Intra-concept distance:  {r.spectral_intra_mean:.6f}")
    print(f"  Inter-concept distance:  {r.spectral_inter_mean:.6f}")
    print(f"  Contrast ratio:          {r.spectral_contrast:.4f}")
    print()

    print("INVARIANCE ANALYSIS")
    print("-" * 40)
    verdict = "YES" if r.invariance_ratio > 1.0 else "NO"
    print(f"  Does WHT increase invariance?  {verdict}")
    print(f"    Invariance ratio: {r.invariance_ratio:.4f} "
          f"({'WHT compresses same-concept distances' if r.invariance_ratio > 1.0 else 'WHT does NOT compress'})")
    print()
    verdict2 = "YES" if r.contrast_ratio > 1.0 else "NO"
    print(f"  Does WHT improve contrast?     {verdict2}")
    print(f"    Contrast ratio: {r.contrast_ratio:.4f} "
          f"({'spectral separation improves' if r.contrast_ratio > 1.0 else 'spectral separation degrades'})")
    print()

    print("PER-VARIANT INVARIANCE (raw_dist / spectral_dist)")
    print("-" * 40)
    print(f"  {'Variant':<20} {'Ratio':>8}  {'WHT helps?':<15}")
    for vtype, ratio in sorted(r.variant_invariance.items()):
        helps = "YES" if ratio > 1.0 else "no"
        print(f"  {vtype:<20} {ratio:>8.4f}  {helps:<15}")
    print()

    print("ENERGY DISTRIBUTION")
    print("-" * 40)
    print(f"  Before WHT (per-dim variance): {r.energy_before[:8]}...")
    print(f"  After WHT  (per-dim variance): {r.energy_after[:8]}...")
    energy_entropy_before = -np.sum(r.energy_before * np.log(r.energy_before + 1e-10))
    energy_entropy_after = -np.sum(r.energy_after * np.log(r.energy_after + 1e-10))
    print(f"  Entropy before: {energy_entropy_before:.4f}")
    print(f"  Entropy after:  {energy_entropy_after:.4f}")
    print()

    print("TIMING")
    print("-" * 40)
    print(f"  Raw computation:      {r.raw_time:.3f}s")
    print(f"  Spectral computation: {r.spectral_time:.3f}s")
    print()

    print("CONCLUSION")
    print("-" * 40)
    if r.invariance_ratio > 1.05 and r.contrast_ratio > 1.0:
        print("  WHT provides meaningful invariance improvement.")
        print("  The spectral representation compresses paraphrase")
        print("  variation while preserving cross-concept separation.")
        print("  -> USE WHT in the interaction encoder pipeline.")
    elif r.invariance_ratio > 0.95:
        print("  WHT has marginal effect on invariance.")
        print("  The spectral representation neither helps nor hurts.")
        print("  -> WHT is NEUTRAL for this feature space.")
        print("  -> Consider whether the transform adds value vs complexity.")
    else:
        print("  WHT DECREASES invariance for this feature space.")
        print("  Raw features are better for paraphrase invariance.")
        print("  -> DO NOT use WHT in the interaction encoder pipeline.")
        print("  -> The transform spreads energy but does not filter noise.")

    print()
    print("MATHEMATICAL NOTE:")
    print("  WHT is orthogonal: H^T H = I. Energy is preserved.")
    print("  The 'washing' in the pipeline happens through:")
    print("    1. Feature selection (which dimensions to extract)")
    print("    2. Projection (dimensional reduction)")
    print("    3. Thresholding (evaluation gating)")
    print("    4. Not the transform itself.")
    print("=" * 70)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    n_concepts = 1000
    seed = 42

    print(f"Building {n_concepts} concepts with 9 variants each...")
    concepts = build_concepts(n_concepts=n_concepts, seed=seed)
    print(f"  Total vectors: {n_concepts * len(concepts[0].variants)}")
    print()

    result = analyze(concepts, seed=seed)
    print_report(result)

    # Also test with different seeds to check robustness
    print()
    print("=" * 70)
    print("ROBUSTNESS CHECK (3 seeds)")
    print("=" * 70)
    for s in [42, 123, 999]:
        r = analyze(concepts, seed=s)
        print(f"  Seed {s}: invariance={r.invariance_ratio:.4f}, "
              f"contrast={r.contrast_ratio:.4f}")


if __name__ == "__main__":
    main()
