# Contributing to Substrate_Echo

Substrate_Echo is an experimental research framework. Contributions are welcome, but the bar for merging is high because every change affects the experimental validity of the system.

## Principles

1. **Test everything.** Every new module must have tests. Run the full suite before submitting.
2. **Document decisions.** If you change architecture, write an ADR (Architecture Decision Record).
3. **No secrets.** Never commit API keys, tokens, or personal data.
4. **Reproducibility matters.** Experiments must be runnable with `python scripts/experiments/...`.

## Development Setup

```bash
pip install -e .
pip install pytest numpy
pytest tests/ -q  # should show 700+ passing
```

## Architecture Decision Records (ADRs)

If your change affects the system's design philosophy, create an ADR:

```
docs/decisions/ADRs/ADR-NNN_title.md
```

Follow the template in `docs/decisions/ADRs/ADR-001_template.md`.

## Experiment Protocol

New experiments go in `scripts/experiments/`. Name them:

```
exp_{scope}_{number}_{description}.py
```

Where scope is:
- `ext` — external agent integration
- `soc` — social ecology
- `core` — core dynamics
- `mem` — memory systems

## Code Style

- Python 3.10+
- Type hints on all public methods
- Dataclasses over dicts
- NumPy for vector operations
- No external NLP dependencies (heuristic feature extraction only)
