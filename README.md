# Substrate_Echo

A dissipative, attractor-based dynamical substrate with emergent memory encoding, topology events, and regime-dependent recovery behavior.

## Current Research Status

Substrate_Echo is an experimental artificial cognition framework investigating:

- Predictive verification over linguistic confidence
- Adaptive reputation systems with domain conditioning
- External information integration through an epistemic firewall
- Multi-agent social identity emergence and maintenance
- Latent state representation dynamics

**Validation:**
- 700+ automated tests
- EXP-EXT external agent experiments (5 experiments, all validated)
- EXP-SOC social ecology experiments (1 experiment, 2000 interactions)
- End-to-end scenario with 10 agents running at 50 t/s

## Architecture

```
External World
    │
    ▼
Observation (32D raw features)
    │
    ▼
Epistemic Firewall (ForeignEvaluator)
    │ alignment, novelty, risk, coherence
    ▼
Acceptance Gate
    │
    ▼
WHT Encoding (post-acceptance only)
    │
    ▼
Latent Space (32D, persistent)
    │
    ▼
Memory / Attractor Formation
```

**Key design principle:** External information is a perturbation source, not an authority. WHT is a coordinate transformation, not a filter. The actual filtering happens in the evaluation manifold.

## Quick Start

```bash
pip install -e .
pytest tests/ -q  # 700+ tests
```

Run an experiment:
```bash
python scripts/experiments/exp_ext_004_prediction_trust.py
python scripts/experiments/exp_soc_001_cognitive_ecology.py
```

## Project Structure

```
substrate_echo/
├── core/           # Cognitive modules (memory, planning, meta-cognition)
├── dynamics/       # Field dynamics, attractors, topology, conservation
├── external/       # Epistemic firewall for external agent integration
├── social/         # Persona genomes, relationship memory, dynamics
├── integration/    # Bridges to other systems
├── models/         # Data structures
├── scenarios/      # End-to-end simulations
└── visualization/  # Field and agent rendering

scripts/experiments/  # Reproducible experiment scripts
tests/                # 700+ automated tests
configs/              # Agent genomes, experiment parameters
docs/
├── architecture/     # System design documentation
├── experiments/      # Experiment reports
└── decisions/ADRs/   # Architecture Decision Records
```

## Experiments

| Experiment | Result | Implication |
|-----------|--------|-------------|
| EXP-EXT-001 | WHT ratio = 1.0000 | WHT is distance-preserving, not a filter |
| EXP-EXT-001B | Quantization 3.6x | Scalar binning preserves contrast |
| EXP-EXT-002 | FAR = 0% | Temporal drift correctly detected |
| EXP-EXT-003 | 86% contamination | Heuristic encoder is weakest component |
| EXP-EXT-004 | B(0.716) > A(0.610) | System values usefulness over presentation |
| EXP-EXT-005 | Physics A>C>B | Domain-specific trust works |
| EXP-SOC-001 | Divergence 0.085 stable | Persona anchoring prevents convergence |

## Architecture Decision Records

- [ADR-001](docs/decisions/ADRs/ADR-002_observation_latent_separation.md): Preserve observation/latent separation
- [ADR-002](docs/decisions/ADRs/ADR-003_verification_trust.md): Verification-driven trust over linguistic quality

## License

MIT
