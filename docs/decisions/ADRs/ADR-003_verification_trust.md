# ADR-002: Verification-Driven Trust Over Linguistic Quality

**Date:** 2026-07-21
**Status:** Accepted

## Context

External agents produce text. The system must decide whether to trust that text. Two approaches:

1. **Linguistic quality**: well-written, confident, structured text is more trustworthy
2. **Predictive accuracy**: text that leads to correct predictions is more trustworthy

EXP-EXT-004 validated that the system correctly values approach #2.

## Decision

Trust is calibrated by predictive usefulness, not presentation quality. An awkward-but-accurate agent is trusted more than a persuasive-but-wrong agent.

## Consequences

### Positive
- Resistant to adversarial persuasion
- Rewards genuinely useful information sources
- Aligns with the scientific principle: what works matters more than what sounds right

### Negative
- Slow to build trust (requires verification cycles)
- Agents with poor presentation but good information are initially underrated

### Risks
- Adversarial agents may learn to provide accurate predictions while embedding subtle misinformation in non-predictive content
- Mitigation: domain-conditioned trust prevents cross-domain exploitation

## Alternatives Considered

1. **Linguistic quality scoring**: Rejected because it rewards adversarial agents who write well.
2. **Reputation-only trust**: Rejected because reputation can be gamed through consistent but subtly wrong predictions.
