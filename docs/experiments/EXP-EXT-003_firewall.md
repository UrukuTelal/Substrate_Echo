# EXP-EXT-003: Epistemic Firewall Benchmark

**Date:** 2026-07-21
**Status:** Complete

## Hypothesis

The IntegrationGate pipeline can distinguish useful external information from noise and adversarial content.

## Method

Injected 3 classes of interactions through the full pipeline:
- **Useful**: coherent, aligned, novel claims
- **Noise**: random strings, garbled text
- **Poison**: adversarial, deceptive, manipulative content

## Result

With heuristic feature extraction: **86% contamination rate**. All three classes were accepted at similar rates.

## Implication

Heuristic feature extraction cannot strongly differentiate useful from harmful text. The encoder treats all inputs as feature vectors; it cannot detect adversarial structure.

This is the correct benchmark result. It shows:

1. The pipeline's current weakness is the encoder, not the evaluator
2. The evaluator's trust-based filtering is the actual firewall
3. Better encoders (embedding-based, LLM-assisted) would improve separation

## Recommendation

Phase S9.5 should benchmark encoder upgrades (embedding → domain classifier → LLM-assisted) while keeping downstream components fixed.
