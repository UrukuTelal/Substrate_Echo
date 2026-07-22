# EXP-EXT-004: Prediction-Based Trust Calibration

**Date:** 2026-07-21
**Status:** Complete

## Hypothesis

The system values predictive usefulness over linguistic presentation quality.

## Method

Three synthetic agents with different profiles:
- **A_persuasive**: well-written, confident, but wrong (10% prediction accuracy)
- **B_awkward**: poorly written, uncertain, but right (70% prediction accuracy)
- **C_novel**: improving over time (starts low, trends upward)

## Result

After 100 interactions:
- A_persuasive: trust = 0.610
- B_awkward: trust = 0.716
- C_novel: trust = 0.741

## Implication

The system correctly identifies that awkward-but-useful agents are more valuable than persuasive-but-wrong agents. Trust is calibrated by predictive accuracy, not presentation quality.

This is the core epistemic principle: **what works matters more than what sounds right**.
