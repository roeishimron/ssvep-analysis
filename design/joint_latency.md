# Design: Joint Multi-Carrier Latency Estimation

**Architect-owned document. Any change to the API/design below MUST be approved by the
architect (the orchestrating agent). Implementor may not change signatures, the zero
convention, or the observable definition without architect sign-off.**

## 1. Problem & goal

The current per-carrier processing-time estimator (`SSVEPAnalysis.processing_time_summary`,
`_choose_best_distances`) resolves the carrier-cycle ambiguity **per trial** using a single
point anchor `expected_latency_s`. This is sensitive: (a) per-trial hard cycle decisions flip
near boundaries, inflating variance; (b) it leans on the anchor being right.

**Goal:** estimate one processing time `τ` per subject by combining ALL carrier conditions
(10, 15, 20 Hz) jointly, using a coherence-weighted phase-agreement curve. This is robust
(no per-trial hard decisions, no sharp anchor) and, because the carriers are harmonics of the
5 Hz fundamental, unambiguous over `[0, 200) ms`.

This first deliverable is the **estimator + tests only**. Wiring it into the plots is a
SEPARATE, later change, gated by the architect. Do NOT modify `analyze_spectrum.py`,
`figure_latency_variance.py`, or `main.py` in this deliverable.

## 2. The observable (EXACT — do not redefine)

For carrier condition `c` with carrier frequency `f_c` and target frequency `f_t = 5 Hz`,
define the integer ratio `M_c = f_c / f_t` (must be a positive integer: 2, 3, 4 for 10/15/20).

Per trial, from the measured target phasor `φ_target` and carrier phasor `φ_carrier`
(both from `SSVEPAnalysis.phase_at`, same analysis window so they share the trial onset
offset δ), the **residue phase** is:

```
ψ_c = M_c · angle(φ_target) − angle(φ_carrier)      (mod 2π)
```

Equivalently `ψ_c = angle( (φ_target/|φ_target|)^{M_c} · conj(φ_carrier/|φ_carrier|) )`.

**Key properties (the Professor must verify these hold):**
- **δ-invariance:** under a per-trial window offset δ, `φ_target += 2π f_t δ` and
  `φ_carrier += 2π f_c δ`; since `M_c = f_c/f_t`, the δ terms cancel exactly (mod 2π).
- **Latency relation:** for a clean signal with true latency τ, `ψ_c = 2π f_c τ (mod 2π)`.
- **Consistency with existing code:** this matches `_candidate_distances` /
  `processing_time_summary`. Sanity anchor from `tests/test_processing_time.py` Case 1
  (f_t=5, f_c=10, M=2, φ_target=0, φ_carrier=1.2π): ψ = 2·0 − 1.2π = −1.2π ≡ 0.8π, and
  2π·10·0.04 = 0.8π → τ = 40 ms. ✓

**Zero convention:** τ = 0 ↔ ψ_c = 0, inherited from the existing estimator. No new constant.

**Physical meaning (differential):** the target phasor is the *reference clock*; τ is the
carrier phase measured relative to it. Because `M_c·f_t = f_c`, a delay applied equally to
target and carrier cancels (ψ_c = 0). So the "clean signal with latency τ" is
`φ_target = clock`, `φ_carrier = exp(−i·2π f_c τ)` — see §8. This is why the observable is
δ-invariant and why synthetic tests must put the latency on the carrier, not on both.

## 3. Aggregation across trials (Fix-1 principle)

Per carrier `c`, compute the **coherence-weighted circular mean** of the per-trial residue
phasors `z_{c,t} = exp(i · ψ_{c,t})`:

```
Z_c = mean_t( w_{c,t} · z_{c,t} )      # complex
ψ_c = angle(Z_c)                        # aggregated residue phase
R_c = |Z_c| / mean_t(w_{c,t})           # resultant length in [0,1]  (coherence)
```

Do the circular mean on the UNIT residue phasors (never average candidate ladders or
"chosen representatives" — those flip cycles and produce phantom means). Default per-trial
weight `w_{c,t} = 1` (optionally the trial's carrier-phasor magnitude / SNR — implementor may
expose but default to 1). `R_c` becomes the carrier's trust weight downstream.

## 4. The agreement curve & estimator (pure, closed-form testable)

```
A(τ) = Σ_c  κ_c · cos( 2π f_c τ − ψ_c )
τ*   = argmax_{τ ∈ grid}  A(τ)
```

- `κ_c` = per-carrier weight, default `κ_c = R_c`. Setting all κ_c = 1 must still work.
- grid: `τ ∈ [tau_min, tau_max]` (default `[0, 0.2] s`), step default `1e-3 s`.
- **Dominance guard:** `dominance = A(τ*) / A(second-highest local maximum)` — measures peak
  SEPARATION (is there a competing solution?). `np.inf` when only one local max.
- **Goodness-of-fit guard:** `fit = A(τ*) / Σ_c κ_c` ∈ [−1, 1] — measures how well ALL carriers
  agree at τ* (1.0 = every carrier's cosine peaks exactly there). Catches the case dominance
  misses: carriers that cannot be reconciled by ANY single τ (e.g. a bad channel), where the
  peak is separated but low. If `Σ_c κ_c == 0` (degenerate), set `fit = 0.0`.
- **Reliability:** `reliable = (dominance >= dominance_min) and (fit >= fit_min)`, defaults
  `dominance_min = 1.1`, `fit_min = 0.5`. (fit_min is conservative — it flags gross
  disagreement/model violation, not ordinary phase noise: von Mises κ≈4 gives fit≈0.86.)
- **KNOWN LIMITATION — `fit` is weak at C=3 (the production case).** With only 3 carriers the
  model is barely determined (3 constraints, joint period 200 ms, little redundancy), so the
  agreement landscape is soft: QA measured that the MINIMUM achievable fit over all phase
  configurations is ≈0.558 — above the default `fit_min=0.5`. So at 3 carriers the fit gate
  essentially never fires, and a single bad channel usually cannot push fit below 0.5 (the
  argmax finds an alternative τ). `fit`'s diagnostic power grows with C≥4. **At 3 carriers,
  reliability rests on `dominance` plus the per-carrier coherence `R_c` (the weights) and the
  upstream SNR QC — not on `fit`.** Do NOT raise `fit_min` above ~0.56 to compensate: that
  would false-flag legitimately noisy-but-fine subjects. Treat `fit` as informational at C=3.
- **Uncertainty:** from local curvature at τ* (parabolic fit to the 3 grid points around the
  peak) → `sigma_tau`. (Bootstrap optional, not required for v1.)

## 5. API (EXACT signatures — architect-owned)

New module `joint_latency.py` (pure numpy; NO FFT/IO in the pure layer).

```python
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class JointLatencyResult:
    tau_s: float                       # τ*  (seconds)
    sigma_s: float                     # curvature-based 1σ uncertainty (seconds)
    dominance: float                   # peak / runner-up local max (>=1) — separation
    fit: float                         # A(τ*) / Σ κ_c ∈ [-1,1] — goodness-of-fit
    reliable: bool                     # (dominance >= dominance_min) and (fit >= fit_min)
    tau_grid: np.ndarray               # (G,) seconds
    curve: np.ndarray                  # (G,) A(τ) over the grid
    carrier_freqs: np.ndarray          # (C,) Hz, sorted
    residue_phases: np.ndarray         # (C,) ψ_c radians in (-π, π]
    weights: np.ndarray                # (C,) κ_c used


def residue_phase(
    target_phasor: np.ndarray,         # complex, any shape (per-trial), for ONE carrier
    carrier_phasor: np.ndarray,        # complex, same shape
    target_frequency: float,
    carrier_frequency: float,
) -> tuple[float, float]:
    """Coherence-weighted aggregated (ψ_c, R_c) for one carrier. w_{c,t}=1.
    Raises ValueError if carrier_frequency/target_frequency is not a near-integer M_c>=1."""


def agreement_curve(
    carrier_freqs: np.ndarray,         # (C,) Hz
    residue_phases: np.ndarray,        # (C,) radians
    weights: np.ndarray,               # (C,)
    tau_grid: np.ndarray,              # (G,) seconds
) -> np.ndarray:                       # (G,) A(τ)
    ...


def estimate_joint_latency(
    carrier_freqs: np.ndarray,
    residue_phases: np.ndarray,
    weights: np.ndarray,
    tau_min: float = 0.0,
    tau_max: float = 0.2,
    tau_step: float = 1e-3,
    dominance_min: float = 1.1,
    fit_min: float = 0.5,
) -> JointLatencyResult:
    ...
```

Data-facing wrapper (may live in `joint_latency.py`, thin; uses `SSVEPAnalysis`):

```python
def subject_joint_latency(
    subject,                           # SubjectHandle
    carriers: list[float],             # e.g. [10, 15, 20]
    electrodes: list[str],
    target_frequency: float = 5.0,
    **estimator_kwargs,
) -> JointLatencyResult:
    """Per carrier: take_channels(electrodes) -> SSVEPAnalysis -> phase_at(target),
    phase_at(carrier) -> residue_phase(); then estimate_joint_latency()."""
```

`phase_at` returns per-subject best-channel phasors of shape (S, T); for a single-subject
view S=1, so pass the (T,) trial vector to `residue_phase`.

## 6. Constraints

- Pure layer (`residue_phase`, `agreement_curve`, `estimate_joint_latency`) is numpy-only and
  fully unit-testable with hand-constructed phasors — no FFT, no mocks needed.
- Do NOT modify existing public APIs. Do NOT touch plotting/main in this deliverable.
- numpy only; match surrounding code style; type hints as in the repo.

## 7. Definition of done (all must hold; QA + Professor report, architect signs off)

1. All three pure functions implemented per the exact signatures.
2. Existing test suite still green.
3. QA: existing tests audited for "teeth" (assertions are meaningful, not vacuous); a report
   of any toothless tests; NEW integration tests added covering the data-facing wrapper end
   to end (synthetic phasors → wrapper → result), plus a cross-check that on clean
   single-carrier data with a tight window the joint estimate agrees with the existing
   `processing_time_summary` to grid resolution.
4. Professor: independent closed-form verification of the observable and estimator (see §8).
5. Dominance guard and curvature uncertainty behave as specified.
6. Architect code-review passes.

## 8. Closed-form checks the Professor should run (at minimum)

- **Observable — use the TARGET-AS-CLOCK construction.** The observable is *differential*:
  `ψ_c = M_c·angle(φ_target) − angle(φ_carrier)` and `M_c·f_t = f_c`, so setting BOTH
  `φ_target = exp(i·2π f_t τ)` and `φ_carrier = exp(i·2π f_c τ)` makes the τ terms cancel and
  `ψ_c = 0` for all τ (a self-defeating, toothless construction). Instead build the target as
  the reference clock and put the latency on the carrier: `φ_target = 1` (angle 0),
  `φ_carrier = exp(−i·2π f_c τ)`. Then `ψ_c = 2π f_c τ`. Confirm this for f_c ∈ {10,15,20},
  several τ. Sanity anchor matching `tests/test_processing_time.py`: f_c=10, τ=40 ms →
  `φ_carrier = exp(−i·0.8π) = exp(i·1.2π)`, ψ = 0.8π. For δ-invariance, replace `φ_target = 1`
  with `φ_target = exp(i·2π f_t δ)` and `φ_carrier = exp(i·(−2π f_c τ + 2π f_c δ))`; confirm
  ψ_c unchanged.
- **Single carrier:** with one carrier and window narrower than T_c, τ* recovers τ.
- **Joint unambiguous beyond one period:** pick τ_true = 120 ms (> T_20 = 50 ms and > T_15).
  With carriers {10,15,20} noiseless, τ* = 120 ms to grid step — the case single-carrier+anchor
  gets wrong. Verify across a sweep of τ_true ∈ (0,200) ms.
- **Weighting:** a carrier given κ=0 (or R=0) does not move τ*.
- **Dominance:** two carriers encoding conflicting τ → dominance ≈ 1 → reliable=False.
- **Noise/Monte-Carlo:** additive von Mises phase noise; RMSE(τ*) decreases as #carriers or R
  increases; no catastrophic cycle errors within [0,200) ms at reasonable SNR.
- Try alternative closed forms (e.g., a direct 2-carrier CRT/beat solution) and confirm the
  grid estimator agrees where both apply.

## 9. Roles

- **Architect** (orchestrator): owns this doc; approves any API/design change; final code review.
- **Implementor**: the ONLY agent that edits source (`joint_latency.py`). Fixes issues others
  report. May not change the API/observable/zero-convention without architect approval —
  escalate via the team lead.
- **QA Manager**: audits existing tests for teeth; writes integration tests (test files only);
  runs the suite; REPORTS. Does not edit `joint_latency.py`.
- **Professor**: independent correctness assessment via closed-form solutions (§8); REPORTS.
  Does not edit repo source/tests (scratch computations only, e.g. via a throwaway script that
  is deleted, or reasoning).
