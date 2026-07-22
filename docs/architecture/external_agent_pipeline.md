# External Agent Pipeline

## Architecture

```
External Agent
    │
    ▼
Raw Interaction (text)
    │
    ▼
InteractionEncoder
    │ semantic features (16D)
    │ relational features (16D)
    ▼
32D Observation Vector (disposable)
    │
    ▼
ForeignEvaluator
    │ alignment, novelty, risk, coherence
    ▼
IntegrationDecision
    │
    ├── REJECT → discarded
    ├── OBSERVE → log only, accumulate samples
    ├── CANDIDATE → queued, awaiting confirmation
    │
    └── ACCEPT
            │
            ▼
        WHT Encoding (post-acceptance)
            │
            ▼
        32D Latent Vector (persistent)
            │
            ▼
        LatentIntegrationRecord (audit trail)
            │
            ▼
        Memory / Attractor Formation
```

## Key Design Decisions

### WHT is Post-Acceptance Only

WHT (Walsh-Hadamard Transform) is a coordinate transformation, not a filter. EXP-EXT-001 confirmed it is distance-preserving (ratio=1.0000). Therefore:

- **Before acceptance:** observation space (raw 32D features)
- **After acceptance:** latent space (WHT-encoded 32D)

Rejected information never enters the latent representation manifold.

### Evaluation Manifold, Not Spectral Transform

The actual filtering happens in the ForeignEvaluator, not the encoder. EXP-EXT-003 showed that heuristic feature extraction produces 86% contamination rate — the encoder cannot distinguish useful from harmful. The evaluator uses:

- Alignment: attractor similarity in DynamicsMemory
- Novelty: distance to existing training data
- Risk: prediction error from verification loop
- Coherence: internal consistency of claims

### Domain-Conditioned Trust

Agents accumulate trust per domain (physics, ecology, social, strategy). EXP-EXT-005 validated that a physics expert is not trusted equally in social contexts.

### Verification-Driven Trust

The VerificationLoop tests external claims against DynamicsMemory predictions. Verified claims increase trust; failed claims decrease it. The system values predictive usefulness over linguistic presentation.

## Safety Progression

| Mode | Behavior |
|------|----------|
| OBSERVATION_ONLY | Log but never route through queue |
| CANDIDATE_ONLY | Queue fills, nothing accepted |
| FULL | Normal operation |
