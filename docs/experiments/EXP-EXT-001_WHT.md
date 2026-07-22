# EXP-EXT-001: WHT Representation Invariance

**Date:** 2026-07-21
**Status:** Complete

## Hypothesis

WHT (Walsh-Hadamard Transform) preserves cosine distances between interaction vectors.

## Method

Generated 50 random 32D vectors in 5 clusters. Computed pairwise cosine distances before and after WHT rotation.

## Result

Distance ratio (post/pre): **1.0000 ± 0.017** across all 100 seeds.

WHT is an orthogonal transform. H^T H = I. Cosine distance is invariant under orthogonal transformation.

## Implication

WHT is NOT a semantic filter. It is a coordinate transformation. If the input contains contamination, WHT will faithfully preserve that contamination in another basis.

**The actual filter is the evaluation manifold (ForeignEvaluator), not the spectral transform.**

## Follow-up

EXP-EXT-001B tested whether quantization after WHT provides invariance. Result: quantization alone provides 3.6x invariance, but WHT+quantation degrades to 0.54x because WHT spreads energy uniformly.
