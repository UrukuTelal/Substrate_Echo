# ADR-001: Preserve Observation/Latent Separation

**Date:** 2026-07-21
**Status:** Accepted

## Context

The original pipeline applied WHT (Walsh-Hadamard Transform) to external interactions during encoding, before evaluation. EXP-EXT-001 confirmed that WHT is distance-preserving (ratio=1.0000) — it is a coordinate transformation, not a filter.

This means contaminated input produces contaminated output in a different basis. Applying WHT before acceptance allows external perturbations to enter the framework's representational manifold even when rejected.

## Decision

WHT encoding is reserved for post-acceptance only. The pipeline maintains two distinct representational spaces:

- **Observation space** (32D raw features): disposable, used only for evaluation
- **Latent space** (32D WHT-encoded): persistent, participates in attractors and predictions

## Consequences

### Positive
- Rejected information never influences latent attractor dynamics
- The epistemic firewall is at the correct point (evaluation, not transformation)
- Audit trail is clean: observation → evaluation → acceptance → encoding → memory

### Negative
- Two representations to maintain (observation and latent)
- Evaluation must work with raw features, not decorrelated ones

### Risks
- The heuristic encoder (16D semantic + 16D relational) is the weakest component. EXP-EXT-003 showed 86% contamination rate. Better encoders needed.

## Alternatives Considered

1. **WHT before evaluation (original)**: Rejected because it violates the principle that external information is a perturbation source, not an authority.
2. **No WHT at all**: Rejected because WHT decorrelates dimensions and provides numerical benefits for attractor formation.
3. **Learned encoder end-to-end**: Deferred to Phase S9.5 — requires training data from the verification loop.
