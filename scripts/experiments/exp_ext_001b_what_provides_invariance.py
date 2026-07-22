"""EXP-EXT-001B: What Actually Provides Invariance?

EXP-EXT-001 proved WHT is distance-preserving (orthogonal).
This experiment tests what DOES compress same-concept distances:

1. Truncation (keep top-k dimensions)
2. Projection via learned linear map
3. Quantization (scalar binning)
4. Combined pipeline

Hypothesis: The filtering happens in the evaluation manifold,
not in the spectral transform.
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
from exp_ext_001_representation_invariance import (
    build_concepts, cosine_distance, Concept,
)


# ---------------------------------------------------------------------------
# Transformation Methods
# ---------------------------------------------------------------------------

def truncate_top_k(vectors: np.ndarray, k: int) -> np.ndarray:
    """Keep only top-k dimensions by magnitude, zero rest.

    This is a form of feature selection: keep the dominant modes.
    """
    n_concepts, n_variants, dim = vectors.shape
    result = np.zeros_like(vectors)
    for i in range(n_concepts):
        for j in range(n_variants):
            top_k = np.argsort(np.abs(vectors[i, j]))[-k:]
            result[i, j, top_k] = vectors[i, j, top_k]
            norm = np.linalg.norm(result[i, j])
            if norm > 1e-10:
                result[i, j] /= norm
    return result


def project_random_linear(vectors: np.ndarray, target_dim: int,
                          seed: int = 42) -> np.ndarray:
    """Project to lower dimension via random linear map.

    This simulates the PSVBridge's 512D→32D projection.
    """
    rng = np.random.default_rng(seed)
    n_concepts, n_variants, dim = vectors.shape
    # Random projection matrix (Gaussian, normalized)
    proj = rng.standard_normal((dim, target_dim)) / np.sqrt(target_dim)
    result = np.einsum('ijk,kd->ijd', vectors, proj)
    # Re-normalize
    norms = np.linalg.norm(result, axis=2, keepdims=True)
    result = result / (norms + 1e-10)
    return result


def quantize_scalar(vectors: np.ndarray, n_bins: int = 8) -> np.ndarray:
    """Scalar quantization: bin each dimension into n_bins levels.

    This is lossy and introduces discretization error.
    """
    result = np.zeros_like(vectors)
    for i in range(vectors.shape[0]):
        for j in range(vectors.shape[1]):
            v = vectors[i, j]
            vmin, vmax = v.min(), v.max()
            if vmax - vmin < 1e-10:
                result[i, j] = 0.5
            else:
                bins = np.linspace(vmin, vmax, n_bins + 1)
                result[i, j] = np.digitize(v, bins[1:-1]) / n_bins
    norms = np.linalg.norm(result, axis=2, keepdims=True)
    result = result / (norms + 1e-10)
    return result


def pipeline_wht_truncate(vectors: np.ndarray, keep_dim: int,
                          seed: int = 42) -> np.ndarray:
    """WHT then truncate to keep_dim dimensions.

    This is the proposed pipeline: decorrelate, then select dominant modes.
    """
    n_concepts, n_variants, dim = vectors.shape
    result = np.zeros_like(vectors)
    d_padded = WHTransform._next_pow2(dim)
    signs = WHTransform.random_signs(d_padded, seed)

    for i in range(n_concepts):
        for j in range(n_variants):
            v = vectors[i, j].copy()
            padded = np.zeros(d_padded)
            padded[:dim] = v * signs
            WHTransform.wht(padded)
            # Keep top keep_dim dimensions
            top_k = np.argsort(np.abs(padded))[-keep_dim:]
            truncated = np.zeros(d_padded)
            truncated[top_k] = padded[top_k]
            # Inverse WHT
            WHTransform.wht(truncated)
            truncated = truncated[:dim] * signs
            norm = np.linalg.norm(truncated)
            if norm > 1e-10:
                truncated /= norm
            result[i, j] = truncated
    return result


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

@dataclass
class TransformResult:
    name: str
    intra_mean: float
    inter_mean: float
    contrast: float
    invariance_ratio: float  # vs raw
    time_s: float


def compute_contrast(vectors: np.ndarray, concepts: List[Concept],
                     variant_types: List[str]) -> Tuple[float, float]:
    """Compute intra and inter mean distances."""
    n_concepts = len(concepts)
    n_variants = len(variant_types)

    intra = []
    for i in range(n_concepts):
        for j in range(n_variants):
            for k in range(j + 1, n_variants):
                intra.append(cosine_distance(vectors[i, j], vectors[i, k]))

    rng = np.random.default_rng(99)
    inter = []
    for _ in range(min(15000, n_concepts * (n_concepts - 1) // 2)):
        a, b = rng.integers(0, n_concepts, size=2)
        if a != b:
            vi, vj = rng.integers(0, n_variants, size=2)
            inter.append(cosine_distance(vectors[a, vi], vectors[b, vj]))

    return float(np.mean(intra)), float(np.mean(inter))


def main() -> None:
    n_concepts = 1000
    seed = 42

    print(f"Building {n_concepts} concepts...")
    concepts = build_concepts(n_concepts=n_concepts, seed=seed)
    variant_types = [v.variant_type for v in concepts[0].variants]
    print(f"  {len(concepts)} concepts x {len(variant_types)} variants")
    print()

    # Extract raw
    raw = np.zeros((n_concepts, len(variant_types), 16))
    for i, c in enumerate(concepts):
        for j, v in enumerate(c.variants):
            raw[i, j] = v.vector

    raw_intra, raw_inter = compute_contrast(raw, concepts, variant_types)
    raw_contrast = raw_inter / (raw_intra + 1e-10)
    print(f"RAW:          intra={raw_intra:.6f}  inter={raw_inter:.6f}  contrast={raw_contrast:.4f}")

    transforms = []

    # Test various truncation levels
    for k in [4, 8, 12, 15]:
        t0 = time.perf_counter()
        truncated = truncate_top_k(raw.copy(), k)
        intra, inter = compute_contrast(truncated, concepts, variant_types)
        t = time.perf_counter() - t0
        contrast = inter / (intra + 1e-10)
        inv = raw_intra / (intra + 1e-10)
        transforms.append(TransformResult(
            name=f"truncate_top_{k}", intra_mean=intra, inter_mean=inter,
            contrast=contrast, invariance_ratio=inv, time_s=t))
        print(f"TRUNCATE-{k}:  intra={intra:.6f}  inter={inter:.6f}  "
              f"contrast={contrast:.4f}  inv={inv:.4f}")

    # Test random projection
    for target in [4, 8, 16]:
        t0 = time.perf_counter()
        projected = project_random_linear(raw.copy(), target, seed=seed)
        intra, inter = compute_contrast(projected, concepts, variant_types)
        t = time.perf_counter() - t0
        contrast = inter / (intra + 1e-10)
        inv = raw_intra / (intra + 1e-10)
        transforms.append(TransformResult(
            name=f"proj_{target}d", intra_mean=intra, inter_mean=inter,
            contrast=contrast, invariance_ratio=inv, time_s=t))
        print(f"PROJ-{target}D:    intra={intra:.6f}  inter={inter:.6f}  "
              f"contrast={contrast:.4f}  inv={inv:.4f}")

    # Test quantization
    for bins in [4, 8, 16]:
        t0 = time.perf_counter()
        quantized = quantize_scalar(raw.copy(), bins)
        intra, inter = compute_contrast(quantized, concepts, variant_types)
        t = time.perf_counter() - t0
        contrast = inter / (intra + 1e-10)
        inv = raw_intra / (intra + 1e-10)
        transforms.append(TransformResult(
            name=f"quant_{bins}bin", intra_mean=intra, inter_mean=inter,
            contrast=contrast, invariance_ratio=inv, time_s=t))
        print(f"QUANT-{bins}BIN:   intra={intra:.6f}  inter={inter:.6f}  "
              f"contrast={contrast:.4f}  inv={inv:.4f}")

    # Test WHT + truncation (the proposed pipeline)
    for keep in [4, 8, 12]:
        t0 = time.perf_counter()
        wt = pipeline_wht_truncate(raw.copy(), keep, seed=seed)
        intra, inter = compute_contrast(wt, concepts, variant_types)
        t = time.perf_counter() - t0
        contrast = inter / (intra + 1e-10)
        inv = raw_intra / (intra + 1e-10)
        transforms.append(TransformResult(
            name=f"wht_trunc_{keep}", intra_mean=intra, inter_mean=inter,
            contrast=contrast, invariance_ratio=inv, time_s=t))
        print(f"WHT+TRUNC-{keep}: intra={intra:.6f}  inter={inter:.6f}  "
              f"contrast={contrast:.4f}  inv={inv:.4f}")

    # Summary
    print()
    print("=" * 70)
    print("SUMMARY: What Provides Invariance?")
    print("=" * 70)
    print(f"  {'Transform':<20} {'Contrast':>10} {'vs Raw':>10} {'Invariance':>12}")
    print(f"  {'RAW':<20} {raw_contrast:>10.4f} {'baseline':>10} {'1.0000':>12}")
    for t in transforms:
        vs_raw = t.contrast / raw_contrast
        print(f"  {t.name:<20} {t.contrast:>10.4f} {vs_raw:>10.4f} {t.invariance_ratio:>12.4f}")

    best = max(transforms, key=lambda x: x.contrast)
    print()
    print(f"  Best: {best.name} (contrast={best.contrast:.4f})")


if __name__ == "__main__":
    main()
