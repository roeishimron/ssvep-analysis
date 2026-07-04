"""Standalone comparison of processing-time estimators — NOT part of main analysis.

Compares two ways of resolving the SSVEP carrier-cycle ambiguity:

  OLD  — per-carrier, per-trial anchor resolution (`_candidate_distances` +
         `_choose_best_distances`, biased toward `expected_latency_s`), then
         pooled across carriers by mean. This is what the figures in main.py use.
  JOINT — the multi-carrier phase-agreement estimator in joint_latency.py
         (no per-trial hard decision, no sharp anchor; unambiguous over [0,200) ms).

Deliverables are plots, written to figures/comparison/. Synthetic (ground-truth)
comparisons always run; real-data comparisons run if an experiment folder is given.

Run:
  MPLBACKEND=Agg python compare_methods.py                      # synthetic only
  MPLBACKEND=Agg python compare_methods.py data/hebrew_vs_mirror   # + real data
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless: we save files, never show
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import wilcoxon

from analysis import SSVEPAnalysis
from core import Study
from core_types import ConditionProperties
from joint_latency import residue_phase, estimate_joint_latency
from loader import StudyLoader

# --- fixed paradigm parameters -------------------------------------------------
F_TARGET = 5.0
CARRIERS = [10.0, 15.0, 20.0]
ELECTRODES = ["T5", "T6", "O1", "O2"]
EXPECTED_S = 0.05          # canonical anchor for the OLD method
OUT_DIR = Path("figures/comparison")

OLD_COLOR = "#D81B60"
OLD1_COLOR = "#F9A825"
JOINT_COLOR = "#1E88E5"
NULL_COLOR = "#757575"      # random-phase null distribution
COHERENT_COLOR = "#2E7D32"  # coherent (genuine shared-latency) reference
TEMPORAL_COLOR = "#D81B60"  # temporal (first-half vs second-half) split
RANDOM_COLOR = "#1E88E5"    # random (shuffled) split — the stationarity null


# ==============================================================================
# Synthetic signal generation (target-as-clock convention, see design doc §2/§8)
# ==============================================================================

def make_phasors(f_c, tau_s, n_trials, kappa, jitter_s, rng):
    """Per-trial (target, carrier) unit phasors encoding true latency tau_s.

    target is the reference clock; the latency lives on the carrier. A per-trial
    onset offset delta (jitter) shifts both phases (cancels in both estimators);
    independent von Mises phase noise (concentration kappa; inf = noiseless) is
    added to each.
    """
    if jitter_s > 0:
        delta = rng.uniform(-jitter_s, jitter_s, n_trials)
    else:
        delta = np.zeros(n_trials)
    if np.isfinite(kappa):
        noise_t = rng.vonmises(0.0, kappa, n_trials)
        noise_c = rng.vonmises(0.0, kappa, n_trials)
    else:
        noise_t = np.zeros(n_trials)
        noise_c = np.zeros(n_trials)

    c_target = np.exp(1j * (2 * np.pi * F_TARGET * delta + noise_t))
    c_carrier = np.exp(1j * (-2 * np.pi * f_c * tau_s + 2 * np.pi * f_c * delta + noise_c))
    return c_target, c_carrier


def phasors_by_carrier(tau_s, n_trials, kappa, jitter_s, rng, carriers=CARRIERS):
    return {f_c: make_phasors(f_c, tau_s, n_trials, kappa, jitter_s, rng) for f_c in carriers}


# ==============================================================================
# The two estimators, driven from identical phasors (isolates the resolution step)
# ==============================================================================

def old_single(c_target, c_carrier, f_c, expected_s=EXPECTED_S):
    """OLD estimator for one carrier: anchor-resolved, trial-averaged. Seconds."""
    t_t, t_c = 1.0 / F_TARGET, 1.0 / f_c
    distances = SSVEPAnalysis._candidate_distances(c_target, c_carrier, t_t, t_c)
    chosen = SSVEPAnalysis._choose_best_distances(distances, t_t, expected_s=expected_s)
    return float(np.angle(np.mean(chosen)) * t_t / (2 * np.pi))


def old_pooled(by_carrier, expected_s=EXPECTED_S):
    """OLD estimator pooled across carriers by mean (as the main plots do). Seconds."""
    ests = [old_single(ct, cc, f_c, expected_s) for f_c, (ct, cc) in by_carrier.items()]
    return float(np.mean(ests))


def joint_result(by_carrier, **kwargs):
    """JOINT estimator over all carriers -> JointLatencyResult."""
    freqs, psis, weights = [], [], []
    for f_c, (ct, cc) in by_carrier.items():
        psi, r = residue_phase(ct, cc, F_TARGET, f_c)
        freqs.append(f_c)
        psis.append(psi)
        weights.append(r)
    return estimate_joint_latency(
        np.array(freqs), np.array(psis), np.array(weights), **kwargs
    )


# ==============================================================================
# Helpers
# ==============================================================================

def _window_ms(combo_hz):
    """Unambiguous window W = 1000/gcd(freqs_Hz) in milliseconds."""
    g = int(np.gcd.reduce(np.array([int(f) for f in combo_hz], dtype=np.intp)))
    return 1000.0 / g


def _joint_for_subset(by_carrier, subset_freqs, **kw):
    """Run the JOINT estimator on a named subset of a pre-built by_carrier dict."""
    freqs, psis, wts = [], [], []
    for f_c in subset_freqs:
        ct, cc = by_carrier[f_c]
        psi, r = residue_phase(ct, cc, F_TARGET, f_c)
        freqs.append(f_c)
        psis.append(psi)
        wts.append(r)
    return estimate_joint_latency(
        np.array(freqs), np.array(psis), np.array(wts), **kw
    )


# ==============================================================================
# Synthetic figures
# ==============================================================================

def fig_recovery_vs_true_tau(rng):
    """Estimated vs true latency, noiseless — exposes cycle errors of OLD."""
    true = np.linspace(0.002, 0.198, 60)
    old1, oldp, jnt = [], [], []
    for tau in true:
        by_c = phasors_by_carrier(tau, 24, np.inf, 0.01, rng)
        old1.append(old_single(*by_c[20.0], 20.0) * 1e3)
        oldp.append(old_pooled(by_c) * 1e3)
        jnt.append(joint_result(by_c).tau_s * 1e3)
    true_ms = true * 1e3

    fig, ax = plt.subplots(figsize=(8, 6), label="compare-recovery-vs-true-tau")
    ax.plot(true_ms, true_ms, "k--", alpha=0.5, label="ideal (y = x)")
    ax.plot(true_ms, old1, "o", ms=4, color=OLD1_COLOR, label="OLD single carrier (20 Hz)")
    ax.plot(true_ms, oldp, "o", ms=4, color=OLD_COLOR, label="OLD pooled (10/15/20)")
    ax.plot(true_ms, jnt, ".", ms=6, color=JOINT_COLOR, label="JOINT (10/15/20)")
    ax.set_xlabel("True latency (ms)")
    ax.set_ylabel("Estimated latency (ms)")
    ax.set_title("Latency recovery vs. ground truth (noiseless)\n"
                 "OLD mis-resolves the carrier cycle away from the anchor; JOINT tracks y=x")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)
    return fig


def fig_rmse_vs_noise(rng):
    """RMSE vs phase-noise concentration, Monte-Carlo."""
    kappas = np.array([0.5, 1, 2, 4, 8, 16, 32, 64])
    tau_true = 0.087
    n_reps, n_trials = 200, 30
    rmse_old1, rmse_oldp, rmse_jnt = [], [], []
    for kappa in kappas:
        e1, ep, ej = [], [], []
        for _ in range(n_reps):
            by_c = phasors_by_carrier(tau_true, n_trials, kappa, 0.01, rng)
            e1.append(old_single(*by_c[20.0], 20.0) - tau_true)
            ep.append(old_pooled(by_c) - tau_true)
            ej.append(joint_result(by_c).tau_s - tau_true)
        rmse_old1.append(np.sqrt(np.mean(np.square(e1))) * 1e3)
        rmse_oldp.append(np.sqrt(np.mean(np.square(ep))) * 1e3)
        rmse_jnt.append(np.sqrt(np.mean(np.square(ej))) * 1e3)

    fig, ax = plt.subplots(figsize=(8, 6), label="compare-rmse-vs-noise")
    ax.plot(kappas, rmse_old1, "o-", color=OLD1_COLOR, label="OLD single carrier (20 Hz)")
    ax.plot(kappas, rmse_oldp, "o-", color=OLD_COLOR, label="OLD pooled (10/15/20)")
    ax.plot(kappas, rmse_jnt, "o-", color=JOINT_COLOR, label="JOINT (10/15/20)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("von Mises concentration κ  (low = noisy → high = clean)")
    ax.set_ylabel("RMSE of estimated latency (ms)")
    ax.set_title(f"Estimator robustness to phase noise\n"
                 f"(true τ = {tau_true*1e3:.0f} ms, {n_reps} reps × {n_trials} trials)")
    ax.legend()
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    return fig


def fig_anchor_sensitivity(rng):
    """OLD estimate vs the anchor it depends on; JOINT is anchor-free (flat)."""
    anchors = np.linspace(0.0, 0.16, 80)
    # τ within OLD's representable range (−100,100] ms so the wandering is purely
    # anchor-driven cycle selection, not the range cap (that is shown elsewhere).
    tau_true = 0.07
    by_c = phasors_by_carrier(tau_true, 24, np.inf, 0.01, rng)
    old1 = [old_single(*by_c[20.0], 20.0, a) * 1e3 for a in anchors]
    oldp = [old_pooled(by_c, a) * 1e3 for a in anchors]
    jnt = joint_result(by_c).tau_s * 1e3  # anchor-independent

    fig, ax = plt.subplots(figsize=(8, 6), label="compare-anchor-sensitivity")
    ax.axhline(tau_true * 1e3, color="k", ls="--", alpha=0.5, label=f"true τ = {tau_true*1e3:.0f} ms")
    ax.plot(anchors * 1e3, old1, color=OLD1_COLOR, label="OLD single carrier (20 Hz)")
    ax.plot(anchors * 1e3, oldp, color=OLD_COLOR, label="OLD pooled (10/15/20)")
    ax.axhline(jnt, color=JOINT_COLOR, lw=2, label=f"JOINT (anchor-free) = {jnt:.0f} ms")
    ax.set_xlabel("expected_latency_s anchor (ms)")
    ax.set_ylabel("Estimated latency (ms)")
    ax.set_title("Sensitivity to the anchor\n"
                 "OLD jumps carrier cycles as the anchor moves; JOINT does not use one")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)
    return fig


def fig_per_trial_flips(rng):
    """Per-trial OLD cycle flips near a boundary vs the stable JOINT aggregate."""
    tau_true = 0.045  # near the 20 Hz half-period boundary (25 ms) region
    f_c = 20.0
    t_t, t_c = 1.0 / F_TARGET, 1.0 / f_c
    ct, cc = make_phasors(f_c, tau_true, 4000, 6.0, 0.0, rng)
    distances = SSVEPAnalysis._candidate_distances(ct, cc, t_t, t_c)
    chosen = SSVEPAnalysis._choose_best_distances(distances, t_t, expected_s=EXPECTED_S)
    per_trial_ms = np.angle(chosen) * t_t / (2 * np.pi) * 1e3

    by_c = phasors_by_carrier(tau_true, 4000, 6.0, 0.0, rng)
    jnt = joint_result(by_c).tau_s * 1e3

    fig, ax = plt.subplots(figsize=(8, 6), label="compare-per-trial-flips")
    ax.hist(per_trial_ms, bins=60, color=OLD1_COLOR, alpha=0.8,
            edgecolor="black", label="OLD per-trial estimates (20 Hz)")
    ax.axvline(tau_true * 1e3, color="k", ls="--", label=f"true τ = {tau_true*1e3:.0f} ms")
    ax.axvline(jnt, color=JOINT_COLOR, lw=2, label=f"JOINT aggregate = {jnt:.0f} ms")
    ax.set_xlabel("Estimated latency (ms)")
    ax.set_ylabel("Trial count")
    ax.set_title("Per-trial cycle flips inflate OLD variance\n"
                 "(κ=6 noise near a cycle boundary; JOINT decides once on the aggregate)")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    return fig


def fig_carrier_invariance(rng):
    """Carrier-invariance: τ̂ is a property of the signal, not of the carrier set.

    Panel A — central estimate at fixed τ_true = 80 ms (300 reps, κ = 4):
               violin distributions for each carrier combination; centres should
               coincide; spread narrows with more/higher-frequency carriers.
    Panel B — window-limited sweep (50 reps, κ = 8, τ ∈ (0, 200) ms):
               median τ̂ vs true τ; each combination tracks y = x within its
               unambiguous window W = 1/gcd(freqs_Hz); {10,20} breaks at 100 ms.

    Fairness rule: within each Monte-Carlo rep, phasors are drawn for the FULL
    set of carriers [10, 15, 20, 25] Hz ONCE; every combination uses its own
    subset of those SAME phasors — differences arise from carrier choice only.
    """
    ALL_CARRIERS = [10.0, 15.0, 20.0, 25.0]

    COMBOS = [
        [10.0, 15.0],
        [10.0, 20.0],
        [15.0, 20.0],
        [10.0, 15.0, 20.0],
        [10.0, 15.0, 20.0, 25.0],
    ]
    COMBO_LABELS = ["{10,15}", "{10,20}", "{15,20}", "{10,15,20}", "{10,15,20,25}"]
    COMBO_COLORS = ["#43A047", "#FB8C00", "#8E24AA", JOINT_COLOR, "#E53935"]
    WINDOWS_MS = [_window_ms(c) for c in COMBOS]

    N_TRIALS = 24

    # ---- Panel A: fixed τ_true = 80 ms, 300 reps, κ = 4 ----
    TAU_A_MS = 80.0
    TAU_A_S = TAU_A_MS / 1e3
    N_REPS_A = 300
    KAPPA_A = 4.0

    # Search EVERY combo over the same tightest window (= min 1/gcd = 100 ms, for
    # {10,20}); τ_true = 80 ms stays inside it so all combos are unambiguous. Using
    # one common search range makes the spread a fair function of carrier choice
    # only — not of how wide each combo's grid happens to be. (Window-limited
    # breakdown is shown separately in Panel B.)
    COMMON_TAU_MAX_S = min(WINDOWS_MS) / 1e3
    hat_A = [[] for _ in COMBOS]
    for _ in range(N_REPS_A):
        by_all = {f_c: make_phasors(f_c, TAU_A_S, N_TRIALS, KAPPA_A, 0.01, rng)
                  for f_c in ALL_CARRIERS}
        for i, combo in enumerate(COMBOS):
            hat_A[i].append(
                _joint_for_subset(by_all, combo, tau_max=COMMON_TAU_MAX_S).tau_s * 1e3
            )

    # ---- Panel B: τ sweep 2–198 ms, 50 reps, κ = 8 ----
    TAU_SWEEP_S = np.linspace(0.002, 0.198, 80)
    N_REPS_B = 50
    KAPPA_B = 8.0

    medians_B = np.zeros((len(COMBOS), len(TAU_SWEEP_S)))
    for j, tau_s in enumerate(TAU_SWEEP_S):
        per_combo = [[] for _ in COMBOS]
        for _ in range(N_REPS_B):
            by_all = {f_c: make_phasors(f_c, tau_s, N_TRIALS, KAPPA_B, 0.01, rng)
                      for f_c in ALL_CARRIERS}
            for i, combo in enumerate(COMBOS):
                per_combo[i].append(_joint_for_subset(by_all, combo).tau_s * 1e3)
        for i in range(len(COMBOS)):
            medians_B[i, j] = float(np.median(per_combo[i]))

    true_ms = TAU_SWEEP_S * 1e3

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(14, 6),
                                      label="compare-carrier-invariance")

    # — Panel A: one violin per combination —
    for i, (color, hat) in enumerate(zip(COMBO_COLORS, hat_A)):
        pos = i + 1
        parts = ax_a.violinplot([hat], positions=[pos], widths=0.7,
                                showmedians=True, showextrema=True)
        for body in parts["bodies"]:
            body.set_facecolor(color)
            body.set_edgecolor(color)
            body.set_alpha(0.65)
        for key in ("cmedians", "cbars", "cmins", "cmaxes"):
            if key in parts:
                parts[key].set_color(color)
        m, s = float(np.mean(hat)), float(np.std(hat))
        yhi = float(np.percentile(hat, 99))
        ax_a.text(pos, yhi + 1.5, f"{m:.1f}±{s:.1f}",
                  ha="center", va="bottom", fontsize=7, color=color)

    ax_a.axhline(TAU_A_MS, color="k", ls="--", lw=1.5,
                 label=f"true τ = {TAU_A_MS:.0f} ms")
    ax_a.set_xticks(range(1, len(COMBOS) + 1))
    ax_a.set_xticklabels(COMBO_LABELS, rotation=20, ha="right", fontsize=9)
    ax_a.set_ylabel("τ̂  (ms)")
    ax_a.set_title(
        f"A — Invariance of the central estimate\n"
        f"(τ_true = {TAU_A_MS:.0f} ms,  κ = {KAPPA_A},  N = {N_REPS_A} reps; "
        f"common 100 ms search window)",
        fontsize=9,
    )
    ax_a.legend(fontsize=8)
    ax_a.grid(axis="y", linestyle="--", alpha=0.4)

    # — Panel B: median τ̂ vs true τ, one line per combination —
    ax_b.plot(true_ms, true_ms, "k--", alpha=0.5, lw=1.5, label="y = x  (ideal)")
    for i, (lbl, color, W_ms) in enumerate(
        zip(COMBO_LABELS, COMBO_COLORS, WINDOWS_MS)
    ):
        ax_b.plot(true_ms, medians_B[i], color=color, lw=2,
                  label=f"{lbl}  (W = {W_ms:.0f} ms)")
        if W_ms < 200.0:
            ax_b.axvline(W_ms, color=color, ls=":", lw=1.5, alpha=0.9)
            ax_b.text(W_ms - 2, 10, f"W = {W_ms:.0f} ms",
                      color=color, fontsize=7, ha="right", va="bottom")

    ax_b.set_xlabel("True τ  (ms)")
    ax_b.set_ylabel("Median τ̂  (ms)")
    ax_b.set_title(
        f"B — Invariance up to the window\n"
        f"(κ = {KAPPA_B},  N = {N_REPS_B} reps/point)",
        fontsize=10,
    )
    ax_b.legend(fontsize=7, loc="upper left")
    ax_b.grid(True, linestyle="--", alpha=0.4)

    fig.suptitle(
        "Carrier-invariance of the joint latency estimator\n"
        "τ̂ is a property of the signal, not of which carriers are used\n"
        "— centres coincide (A);  each tracks y = x within its window W (B) —",
        weight="bold", fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    return fig


# ==============================================================================
# Real-data figures
# ==============================================================================

def _common_subjects(study):
    reqs = {ConditionProperties(np.float64(F_TARGET), np.float64(c)) for c in CARRIERS}
    return list(study.filter_subjects(reqs))


def real_old_pooled_ms(subject):
    ests = []
    for f_c in CARRIERS:
        rec = subject[ConditionProperties(np.float64(F_TARGET), np.float64(f_c))].take_channels(ELECTRODES)
        mean_ms, _ = SSVEPAnalysis(rec, expected_latency_s=EXPECTED_S).processing_time_summary()
        ests.append(float(mean_ms[0]))
    return float(np.mean(ests))


def collect_real(study):
    """Per-subject (name, old_ms, joint_result)."""
    from joint_latency import subject_joint_latency
    rows = []
    for subject in _common_subjects(study):
        try:
            res = subject_joint_latency(subject, CARRIERS, ELECTRODES, target_frequency=F_TARGET)
            old_ms = real_old_pooled_ms(subject)
        except Exception as e:  # noqa: BLE001 — skip malformed subjects, report
            print(f"  skipping subject {getattr(subject, 'name', '?')}: {e}")
            continue
        rows.append((subject.name, old_ms, res))
    return rows


def fig_real_scatter(rows):
    old = np.array([r[1] for r in rows])
    jnt = np.array([r[2].tau_s * 1e3 for r in rows])
    min_coh = np.array([float(np.min(r[2].weights)) for r in rows])

    fig, ax = plt.subplots(figsize=(8, 6.5), label="compare-real-old-vs-joint")
    lim = [0, max(200.0, float(np.nanmax([old.max(), jnt.max()])) * 1.05)]
    ax.plot(lim, lim, "k--", alpha=0.5, label="y = x")
    sc = ax.scatter(old, jnt, c=min_coh, cmap="viridis", vmin=0, vmax=1,
                    s=60, edgecolor="black", zorder=3)
    fig.colorbar(sc, ax=ax, label="min carrier coherence  min(R_c)")
    ax.set_xlabel("OLD pooled latency (ms)")
    ax.set_ylabel("JOINT latency (ms)")
    ax.set_title(f"Per-subject latency: OLD vs JOINT  (N = {len(rows)})\n"
                 "off-diagonal = methods disagree; color = joint's weakest carrier")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)
    return fig


def fig_real_agreement_curves(rows):
    n = min(6, len(rows))
    sel = rows[:n]
    ncol = 2
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(11, 2.6 * nrow),
                             squeeze=False, label="compare-real-agreement-curves")
    for ax, (name, old_ms, res) in zip(axes.flatten(), sel):
        grid_ms = res.tau_grid * 1e3
        ax.plot(grid_ms, res.curve, color=JOINT_COLOR)
        ax.axvline(res.tau_s * 1e3, color=JOINT_COLOR, lw=2,
                   label=f"JOINT {res.tau_s*1e3:.0f} ms")
        ax.axvline(old_ms, color=OLD_COLOR, ls="--", lw=2, label=f"OLD {old_ms:.0f} ms")
        ax.set_title(f"{name}  (dom={res.dominance:.2f}, fit={res.fit:.2f}, "
                     f"{'reliable' if res.reliable else 'UNRELIABLE'})",
                     fontsize=9, loc="left")
        ax.set_xlabel("τ (ms)")
        ax.set_ylabel("A(τ)")
        ax.legend(fontsize=7)
        ax.grid(True, linestyle="--", alpha=0.3)
    for ax in axes.flatten()[n:]:
        ax.set_visible(False)
    fig.suptitle("Agreement curves: where OLD and JOINT land on each subject", weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return fig


def fig_real_disagreement(rows):
    disagree = np.array([abs(r[1] - r[2].tau_s * 1e3) for r in rows])
    joint_ms = np.array([r[2].tau_s * 1e3 for r in rows])
    min_coh = np.array([float(np.min(r[2].weights)) for r in rows])

    # OLD's angle->ms output is confined to (-100, 100] ms; a genuine latency
    # beyond that half-period is the structural driver of disagreement.
    OLD_CAP = 100.0

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), label="compare-real-disagreement")

    # Left: the actual driver — latency magnitude vs OLD's representable range.
    axes[0].axvspan(OLD_CAP, joint_ms.max() * 1.05, color="#D81B60", alpha=0.08)
    axes[0].axvline(OLD_CAP, color="#D81B60", ls="--",
                    label="OLD range cap (100 ms)")
    axes[0].scatter(joint_ms, disagree, s=55, color=JOINT_COLOR, edgecolor="black")
    axes[0].set_xlabel("JOINT latency (ms)")
    axes[0].set_ylabel("|OLD − JOINT|  (ms)")
    axes[0].set_title("Disagreement IS driven by latency magnitude\n"
                      "(explodes once τ exceeds OLD's representable range)", fontsize=10)
    axes[0].legend()
    axes[0].grid(True, linestyle="--", alpha=0.4)

    # Right: the null — data quality does NOT explain OLD's error.
    axes[1].scatter(min_coh, disagree, s=55, color=JOINT_COLOR, edgecolor="black")
    axes[1].set_xlabel("min carrier coherence  min(R_c)")
    axes[1].set_ylabel("|OLD − JOINT|  (ms)")
    axes[1].set_title("Data quality does NOT predict it\n"
                      "(OLD's errors are invisible to coherence)", fontsize=10)
    axes[1].grid(True, linestyle="--", alpha=0.4)

    fig.suptitle("What drives OLD–JOINT disagreement? Latency magnitude, not quality",
                 weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return fig


# ==============================================================================
# Random-phase null floor (does JOINT confidence exceed chance?)
# ==============================================================================

def _wrap(a):
    """Wrap angle(s) to (-π, π]."""
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi


def random_phase_null(n_draws, rng, weights_pool=None):
    """Monte-Carlo null: fit & dominance when the carriers share NO latency.

    Each draw sets the three residue phases ψ_c ~ iid Uniform(-π, π] — i.e. the
    carriers carry no joint information. With κ_c = 1 (default) this is the
    primary R_c-invariant null. If `weights_pool` (rows of real R_c triplets) is
    given, each draw instead samples a real subject's weights — the "matched-R_c"
    null — for realism. The DEFAULT window ([0, 0.2] s) is used, matching the
    real-data estimates.

    Returns (fits (n_draws,), dominances (n_draws,)).
    """
    fits = np.empty(n_draws)
    doms = np.empty(n_draws)
    for i in range(n_draws):
        psi = rng.uniform(-np.pi, np.pi, len(CARRIERS))
        if weights_pool is None:
            w = np.ones(len(CARRIERS))
        else:
            w = np.asarray(weights_pool[rng.integers(len(weights_pool))], dtype=float)
        res = estimate_joint_latency(np.array(CARRIERS), psi, w)
        fits[i] = res.fit
        doms[i] = res.dominance
    return fits, doms


def coherent_reference(tau_s=0.09, kappa=50.0, n_draws=1, rng=None):
    """Coherent reference: all carriers encode ONE shared latency τ.

    ψ_c = wrap(2π f_c τ) with light von Mises phase noise (κ large). This
    brackets the TOP of the confidence scale (fit≈1, dominance≈C). With
    kappa=inf / n_draws=1 it is the exact noiseless case (fit=1, dominance=C).

    Returns (fits (n_draws,), dominances (n_draws,)).
    """
    freqs = np.array(CARRIERS)
    fits = np.empty(n_draws)
    doms = np.empty(n_draws)
    for i in range(n_draws):
        if np.isfinite(kappa):
            noise = rng.vonmises(0.0, kappa, len(CARRIERS))
        else:
            noise = np.zeros(len(CARRIERS))
        psi = _wrap(2 * np.pi * freqs * tau_s + noise)
        res = estimate_joint_latency(freqs, psi, np.ones(len(CARRIERS)))
        fits[i] = res.fit
        doms[i] = res.dominance
    return fits, doms


def fig_null_vs_real_confidence(rows):
    """Do JOINT's confidence metrics on real data exceed a random-phase null?

    Two panels (fit | dominance). In each: the random-phase null distribution
    (grey, ψ_c iid uniform, R_c=1, N draws); the 16 real subjects overlaid as a
    rug; the coherent (genuine shared-latency) reference marked in green. The
    null mean and 95th percentile are annotated, together with the fraction of
    real subjects above the null 95th percentile.

    HONEST HEADLINE (measured, C=3): the real fit/dominance distributions sit
    essentially INSIDE the random-phase null and far below the coherent
    reference — at three carriers, fit & dominance computed from phases alone do
    NOT certify a shared latency (design §4 KNOWN LIMITATION). Reliability rests
    on the per-carrier coherence R_c and upstream SNR QC, not on these metrics.
    """
    rng = np.random.default_rng(20260702)
    n_draws = 5000

    null_fit, null_dom = random_phase_null(n_draws, rng)
    # Secondary "matched-R_c" null (samples real subjects' weights) for realism.
    w_pool = [r[2].weights for r in rows]
    matched_fit, matched_dom = random_phase_null(n_draws, rng, weights_pool=w_pool)

    # Coherent reference: noiseless (exact top of scale) + a light-noise cloud.
    coh_fit0, coh_dom0 = coherent_reference(kappa=np.inf, n_draws=1)
    coh_fit_n, coh_dom_n = coherent_reference(kappa=50.0, n_draws=2000, rng=rng)

    real_fit = np.array([r[2].fit for r in rows])
    real_dom = np.array([r[2].dominance for r in rows])
    names = [r[0] for r in rows]

    specs = [
        ("fit  A(τ*) / Σκ_c", null_fit, matched_fit, real_fit,
         float(coh_fit0[0]), coh_fit_n),
        ("dominance  A(τ*) / runner-up", null_dom, matched_dom, real_dom,
         float(coh_dom0[0]), coh_dom_n),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6.5),
                             label="compare-null-vs-real-confidence")

    for ax, (label, null, matched, real, coh0, coh_cloud) in zip(axes, specs):
        p95 = float(np.percentile(null, 95))
        mean_null = float(np.mean(null))
        # Finite range for the histogram (dominance can be +inf when a single
        # local max exists; clip only the display, count infs as "above").
        finite = null[np.isfinite(null)]
        hi = float(np.percentile(finite, 99.5))
        lo = float(np.min(finite))
        bins = np.linspace(lo, max(hi, coh0), 60)

        ax.hist(np.clip(null, bins[0], bins[-1]), bins=bins, density=True,
                color=NULL_COLOR, alpha=0.55,
                label=f"random-phase null (R_c=1, N={n_draws})")
        ax.hist(np.clip(matched, bins[0], bins[-1]), bins=bins, density=True,
                histtype="step", color=NULL_COLOR, ls=":", lw=1.6,
                label="matched-R_c null")

        # Coherent reference: noiseless line + light-noise cloud span.
        ax.axvline(coh0, color=COHERENT_COLOR, lw=2.5,
                   label=f"coherent reference (noiseless) = {coh0:.2f}")
        c_lo, c_hi = np.percentile(coh_cloud, [2.5, 97.5])
        ax.axvspan(c_lo, c_hi, color=COHERENT_COLOR, alpha=0.12)

        # Null summary markers.
        ax.axvline(mean_null, color=NULL_COLOR, ls="--", lw=1.8,
                   label=f"null mean = {mean_null:.2f}")
        ax.axvline(p95, color="black", ls="-.", lw=1.8,
                   label=f"null 95th pct = {p95:.2f}")

        # Real subjects as a rug along the top.
        ymax = ax.get_ylim()[1]
        ax.plot(np.clip(real, bins[0], bins[-1]),
                np.full_like(real, ymax * 0.92), "v",
                color=JOINT_COLOR, ms=9, mec="black", zorder=5,
                label=f"real subjects (N={len(real)})")

        # Label the LOWEST real subjects (the ambiguous ones deepest in the null).
        order = np.argsort(real)
        for k in order[:3]:
            ax.annotate(names[k].split("_")[0],
                        xy=(np.clip(real[k], bins[0], bins[-1]), ymax * 0.92),
                        xytext=(0, -14), textcoords="offset points",
                        ha="center", va="top", fontsize=7, rotation=90,
                        color=JOINT_COLOR)

        n_above = int(np.sum(real > p95))
        ax.set_title(
            f"{label}\n"
            f"real mean = {np.mean(real):.2f}  |  "
            f"{n_above}/{len(real)} real above null 95th pct",
            fontsize=10, loc="left")
        ax.set_xlabel(label.split("  ")[0])
        ax.set_ylabel("density")
        ax.legend(fontsize=7, loc="upper left")
        ax.grid(True, linestyle="--", alpha=0.3)

    fig.suptitle(
        "Random-phase null vs real confidence (3 carriers)\n"
        "Real fit & dominance sit INSIDE the random-phase null and far below the "
        "coherent reference:\n"
        "at C=3 these metrics do not certify a shared latency (design §4) — R_c / "
        "SNR QC carry reliability",
        weight="bold", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    return fig


# ==============================================================================
# Within-trial split-half stationarity (temporal split vs random split)
# ==============================================================================
#
# Question: is the target–carrier residue phase (hence the per-trial
# processing-time latency) STATIONARY within a single trial? A trial holds many
# analysis windows, so this is the one check not throttled by the 3-trial sample.
#
# Method: split a trial's windows two ways and compare the residue phase of the
# two halves. A TEMPORAL split (first half of the time-ordered windows vs second
# half) is sensitive to any within-trial drift. A RANDOM split (shuffle then
# halve) is the matched null — same window count, same noise, but drift is
# averaged out. If the phase is stationary the two splits give the same |Δ|
# distribution; if it drifts, the temporal split |Δ| is systematically larger.
#
# Window independence: adjacent OVERLAPPING windows are correlated, which would
# contaminate the random-split null. `_deoverlap_stride` subsamples to
# non-overlapping windows (factor k). The current analysis.py already builds
# NON-overlapping windows (hop == window_size), so k resolves to 1 on real data.

STATIONARITY_SEED = 20260702


def _deoverlap_stride(n_time, window_size, n_windows):
    """Infer the window hop from cache geometry; return (k, hop_samples).

    Windows are a strided sliding_window_view, so the hop in samples is
    (n_time − window_size)/(n_windows − 1). To keep only NON-overlapping windows
    take every k-th, with k = ceil(window_size / hop); non-overlapping windows
    (hop >= window_size) give k = 1.
    """
    if n_windows <= 1:
        return 1, float(window_size)
    hop = (n_time - window_size) / (n_windows - 1)
    if hop <= 0:
        return 1, float(window_size)
    # -1e-9 guards against ceil() tripping on floating-point 1.0000000001.
    k = int(np.ceil(window_size / hop - 1e-9))
    return max(1, k), hop


def _split_indices(n, rng):
    """(temporal_first, temporal_second, random_first, random_second) index arrays.

    Both splits use equal halves of size n//2 (a middle window is dropped when n
    is odd) so temporal and random are paired on identical sample sizes. The
    temporal halves are the time-ordered first vs last n//2 windows.
    """
    order = np.arange(n)
    half = n // 2
    perm = rng.permutation(n)
    return order[:half], order[half:2 * half], perm[:half], perm[half:2 * half]


def _half_stats(target_w, carrier_w, idx_first, idx_second, f_target, f_carrier):
    """Half-difference of residue / raw target / raw carrier phase between two
    window subsets of one trial.

    Each half's phasor is the AMPLITUDE-WEIGHTED coherent mean (raw complex mean
    — no per-window normalisation, which would deflate the estimate). Returns:
      dtau  : wrap(ψ_first − ψ_second)/(2π f_c)         residue, seconds
      dtar  : wrap(angle(M_target_first) − ..._second)  raw target, radians
      dcar  : wrap(angle(M_carrier_first) − ..._second) raw carrier, radians
    with ψ_half = M_c·angle(mean_target) − angle(mean_carrier), M_c = f_c/f_t.
    """
    M_c = f_carrier / f_target
    mt1, mt2 = target_w[idx_first].mean(), target_w[idx_second].mean()
    mc1, mc2 = carrier_w[idx_first].mean(), carrier_w[idx_second].mean()
    psi1 = M_c * np.angle(mt1) - np.angle(mc1)
    psi2 = M_c * np.angle(mt2) - np.angle(mc2)
    dtau = _wrap(psi1 - psi2) / (2 * np.pi * f_carrier)
    dtar = _wrap(np.angle(mt1) - np.angle(mt2))
    dcar = _wrap(np.angle(mc1) - np.angle(mc2))
    return float(dtau), float(dtar), float(dcar)


def within_trial_split_diffs(target_tw, carrier_tw, f_target, f_carrier, k, rng):
    """Per-trial temporal- and random-split half-differences for one (subject,
    carrier) pair.

    Parameters
    ----------
    target_tw, carrier_tw : (T, W) complex — per-window phasors, best SNR channel
    f_target, f_carrier : float — Hz
    k : int — de-overlap subsample factor (keep every k-th window)
    rng : np.random.Generator — seeded, drives the random split

    Returns
    -------
    dict{quantity -> (temporal_list, random_list)} with quantities
    'residue' (seconds), 'target' (radians), 'carrier' (radians); one entry per
    trial in each list.
    """
    T, W = target_tw.shape
    kept = np.arange(W)[::k]
    out = {q: ([], []) for q in ("residue", "target", "carrier")}
    for t in range(T):
        tw = target_tw[t, kept]
        cw = carrier_tw[t, kept]
        n = len(kept)
        if n < 2:
            continue
        tf, ts, rf, rs = _split_indices(n, rng)
        temporal = _half_stats(tw, cw, tf, ts, f_target, f_carrier)
        random_ = _half_stats(tw, cw, rf, rs, f_target, f_carrier)
        for j, q in enumerate(("residue", "target", "carrier")):
            out[q][0].append(temporal[j])
            out[q][1].append(random_[j])
    return out


def _per_window_phasors(rec, f_target, f_carrier):
    """(target_tw, carrier_tw, k, hop, W) per-window phasors for one recording.

    Mirrors SSVEPAnalysis.phase_at but keeps the window axis instead of coherent-
    averaging over it. Best SNR channel is chosen independently for target and
    carrier (argmax over channels of trial-mean SNR at that bin).
    """
    sp = SSVEPAnalysis(rec)
    freqs = sp.frequencies()
    t_bin = int(np.argmin(np.abs(freqs - f_target)))
    c_bin = int(np.argmin(np.abs(freqs - f_carrier)))

    snr = sp.snr_per_trial()[0]                       # (T, C, F)
    t_ch = int(np.argmax(snr[:, :, t_bin].mean(0)))
    c_ch = int(np.argmax(snr[:, :, c_bin].mean(0)))

    fourier = sp._fourier_cache                       # (S, T, C, W, F)
    target_tw = fourier[0, :, t_ch, :, t_bin]         # (T, W)
    carrier_tw = fourier[0, :, c_ch, :, c_bin]        # (T, W)

    n_time = rec.raw_data().shape[-1]
    W = fourier.shape[3]
    k, hop = _deoverlap_stride(n_time, sp._window_size, W)
    return target_tw, carrier_tw, k, hop, W


def collect_stationarity(study, rng=None):
    """Aggregate temporal/random split diffs across (subject, carrier, trial).

    Returns (agg, meta) where
      agg  : {quantity -> {'temporal': list, 'random': list}}
             residue in seconds, target/carrier in radians (plus 'carrier_ms'
             already converted to ms via the known per-sample carrier freq).
      meta : {'window_counts': [...], 'ks': [...], 'hops': [...], 'n_units': int}
    """
    if rng is None:
        rng = np.random.default_rng(STATIONARITY_SEED)
    agg = {q: {"temporal": [], "random": []}
           for q in ("residue", "target", "carrier", "carrier_ms")}
    window_counts, ks, hops = [], [], []
    n_units = 0
    for subject in _common_subjects(study):
        for f_c in CARRIERS:
            rec = subject[ConditionProperties(
                np.float64(F_TARGET), np.float64(f_c))].take_channels(ELECTRODES)
            try:
                target_tw, carrier_tw, k, hop, W = _per_window_phasors(
                    rec, F_TARGET, f_c)
            except Exception as e:  # noqa: BLE001 — skip malformed, report
                print(f"  skipping {getattr(subject,'name','?')} c={f_c}: {e}")
                continue
            window_counts.append(len(np.arange(W)[::k]))
            ks.append(k)
            hops.append(hop)
            diffs = within_trial_split_diffs(
                target_tw, carrier_tw, F_TARGET, f_c, k, rng)
            n_units += len(diffs["residue"][0])
            for q in ("residue", "target", "carrier"):
                agg[q]["temporal"].extend(diffs[q][0])
                agg[q]["random"].extend(diffs[q][1])
            # Carrier raw-phase difference expressed in ms (per-carrier freq).
            conv = 1.0 / (2 * np.pi * f_c) * 1e3
            agg["carrier_ms"]["temporal"].extend(
                d * conv for d in diffs["carrier"][0])
            agg["carrier_ms"]["random"].extend(
                d * conv for d in diffs["carrier"][1])
    meta = {"window_counts": window_counts, "ks": ks, "hops": hops,
            "n_units": n_units}
    return agg, meta


def stationarity_verdict(temporal, random, ratio_thresh=1.3):
    """Paired |temporal| vs |random| summary: (med_t, med_r, ratio, p, verdict).

    Verdict is NON-STATIONARY when the temporal split is both meaningfully and
    significantly larger than the random null (ratio >= ratio_thresh AND
    Wilcoxon signed-rank p < 0.05); otherwise 'stationary'.
    """
    t = np.abs(np.asarray(temporal, dtype=float))
    r = np.abs(np.asarray(random, dtype=float))
    med_t = float(np.median(t))
    med_r = float(np.median(r))
    ratio = med_t / med_r if med_r > 0 else np.inf
    try:
        p = float(wilcoxon(t, r).pvalue)
    except ValueError:
        p = np.nan
    non_stat = (ratio >= ratio_thresh) and np.isfinite(p) and (p < 0.05)
    return med_t, med_r, ratio, p, ("NON-STATIONARY" if non_stat else "stationary")


def fig_within_trial_stationarity(agg, meta):
    """Three panels (residue ms | raw target rad | raw carrier rad): overlaid
    temporal-split vs random-split |Δ| distributions, with medians, ratio, p and
    an honest per-panel verdict."""
    # (key, display array builder, unit label, panel title)
    panels = [
        ("residue", lambda x: np.abs(x) * 1e3, "|Δτ| residue (ms)",
         "Residue phase → processing-time latency"),
        ("target", lambda x: np.abs(x), "|Δφ| target 5 Hz (rad)",
         "Raw target (5 Hz) phase"),
        ("carrier", lambda x: np.abs(x), "|Δφ| carrier (rad)",
         "Raw carrier phase"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5),
                             label="compare-within-trial-stationarity")
    verdicts = {}
    for ax, (key, disp, xlabel, title) in zip(axes, panels):
        temp = np.asarray(agg[key]["temporal"], dtype=float)
        rand = np.asarray(agg[key]["random"], dtype=float)
        med_t, med_r, ratio, p, verdict = stationarity_verdict(temp, rand)
        verdicts[key] = (med_t, med_r, ratio, p, verdict)

        t_disp = disp(temp)
        r_disp = disp(rand)
        hi = float(np.percentile(np.concatenate([t_disp, r_disp]), 98))
        bins = np.linspace(0, max(hi, 1e-9), 40)

        ax.hist(np.clip(r_disp, 0, bins[-1]), bins=bins, density=True,
                color=RANDOM_COLOR, alpha=0.5,
                label=f"random split (null)\nmedian={disp(np.array([med_r]))[0]:.3g}")
        ax.hist(np.clip(t_disp, 0, bins[-1]), bins=bins, density=True,
                histtype="step", lw=2.0, color=TEMPORAL_COLOR,
                label=f"temporal split\nmedian={disp(np.array([med_t]))[0]:.3g}")
        ax.axvline(disp(np.array([med_r]))[0], color=RANDOM_COLOR, ls="--", lw=1.8)
        ax.axvline(disp(np.array([med_t]))[0], color=TEMPORAL_COLOR, ls="-", lw=1.8)

        pstr = "n/a" if not np.isfinite(p) else (f"{p:.2g}" if p >= 1e-4 else "<1e-4")
        flag = "⚠ " if verdict == "NON-STATIONARY" else ""
        ax.set_title(f"{title}\n{flag}{verdict}  "
                     f"(temporal/random = {ratio:.2f}, p = {pstr})",
                     fontsize=10, loc="left")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("density")
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(True, linestyle="--", alpha=0.3)

    res_v = verdicts["residue"]
    raw_drifts = verdicts["target"][4] == "NON-STATIONARY" or \
        verdicts["carrier"][4] == "NON-STATIONARY"
    if res_v[4] == "stationary":
        headline = (
            "Within-trial processing-time latency IS stationary — "
            "residue temporal split ≈ random null"
        )
        if raw_drifts:
            headline += ("\n(raw target & carrier phases drift within-trial, but "
                         "that common-mode drift cancels in the residue)")
    else:
        headline = (
            "Within-trial processing-time latency is NOT stationary — "
            "residue temporal split ≫ random null (within-trial drift)"
        )
    wc = meta["window_counts"]
    kset = sorted(set(meta["ks"]))
    fig.suptitle(
        "Within-trial split-half stationarity of the processing-time phase\n"
        f"{headline}\n"
        f"(N = {meta['n_units']} subject×carrier×trial units; "
        f"non-overlapping windows/trial: {min(wc)}–{max(wc)}; de-overlap k∈{kset}; "
        f"residue temporal/random ratio = {res_v[2]:.2f})",
        weight="bold", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.83))
    return fig


# ==============================================================================
# SNR vs window size (self-contained rectangular FFT)
# ==============================================================================
#
# User claim to validate/refute: with a RECTANGULAR window (no Kaiser taper),
# LOWERING the window size yields BETTER SNR ("kills more noise without harming
# the signal").
#
# This block does its OWN rectangular FFT at each window size — it does NOT use
# analysis.py._compute_fourier (which is in flux). Only raw_data() is read.
#
# What the physics says (and the code below shows):
#   * Coherent averaging preserves an on-bin tone: its amplitude at the bin is
#     ∝ ws, so on-bin POWER ∝ ws². White-noise power at the bin also ∝ ws². The
#     ratio (SNR) is therefore FLAT in ws — SNR is set by TOTAL integration time,
#     not window size. Lowering ws does NOT kill noise "for free"; it shrinks the
#     coherent signal estimate in exact proportion. → claim FALSE for a stationary
#     signal.
#   * A NON-STATIONARY (chirp / drifting) tone wanders off the fixed measurement
#     bin. Short windows have coarse Δf that tolerates the wander, so their
#     coherent on-bin power survives; long windows resolve the wander off-bin and
#     lose it. → small ws IS better, but only because the signal is non-stationary
#     (a real gain), not because noise was "killed".
#   * CONFOUND: shrinking ws coarsens Δf, so the neighbor-bin noise estimate spans
#     a different Hz band at each ws. Over real EEG's 1/f + alpha spectrum that
#     alone moves the SNR. `rect_snr_fixed_band` re-estimates noise over a FIXED
#     Hz band so the SNR is comparable across ws and the artifact is exposed.

# On-bin sweep: every ws is a multiple of 0.2 s so 5/10/15/20 Hz land exactly on
# an FFT bin (no scalloping confound). 0.2/0.4/0.8 s are kept for the carriers;
# for the 5 Hz target they are UNDEFINED (the bin falls within the neighbor
# guard of DC) and reported as NaN.
SNR_WS_SWEEP = [0.4, 0.8, 1.0, 1.4, 2.0, 3.0, 4.0]
SNR_N_NEIGHBORS = 3
SNR_N_SKIP = 1


def _rect_coherent_power(signal, fs, ws_s):
    """Non-overlapping rectangular windows → per-window-demeaned rfft → COHERENT
    (complex-mean) average over windows → |·|². Returns (freqs, power, n_windows).

    Self-contained; does not touch analysis.py. `signal` is (..., Time); the
    window axis is collapsed, leaving power of shape signal.shape[:-1] + (F,).
    """
    ws = int(round(ws_s * fs))
    n_time = signal.shape[-1]
    n_win = n_time // ws
    if n_win < 1:
        raise ValueError(f"window {ws_s}s ({ws} samples) longer than signal ({n_time})")
    seg = signal[..., :n_win * ws].reshape(*signal.shape[:-1], n_win, ws)
    seg = seg - seg.mean(axis=-1, keepdims=True)
    coherent = np.fft.rfft(seg, axis=-1).mean(axis=-2)
    freqs = np.fft.rfftfreq(ws, d=1.0 / fs)
    return freqs, (np.abs(coherent) ** 2), n_win


def _rect_incoherent_power(signal, fs, ws_s):
    """INCOHERENT variant: mean of per-window |rfft|² (power averaging).

    Provided for contrast only — the analysis pipeline uses coherent averaging.
    Under incoherent averaging a drifting tone still favours short windows, but a
    stationary tone favours LONG windows (opposite of the claim); see the report.
    """
    ws = int(round(ws_s * fs))
    n_time = signal.shape[-1]
    n_win = n_time // ws
    if n_win < 1:
        raise ValueError(f"window {ws_s}s longer than signal")
    seg = signal[..., :n_win * ws].reshape(*signal.shape[:-1], n_win, ws)
    seg = seg - seg.mean(axis=-1, keepdims=True)
    power = (np.abs(np.fft.rfft(seg, axis=-1)) ** 2).mean(axis=-2)
    freqs = np.fft.rfftfreq(ws, d=1.0 / fs)
    return freqs, power, n_win


def _snr_from_power(power, n_neighbors=SNR_N_NEIGHBORS, n_skip=SNR_N_SKIP):
    """Local copy of analysis._into_snr (kept self-contained): each bin's noise is
    the mean of n_neighbors bins on each side, skipping n_skip adjacent bins; edge
    bins are padded with inf → SNR 0."""
    kernel = np.concatenate((
        np.ones(n_neighbors), np.zeros(2 * n_skip + 1), np.ones(n_neighbors)))
    kernel /= kernel.sum()
    mean_noise = np.apply_along_axis(
        lambda x: np.convolve(x, kernel, mode="valid"), axis=-1, arr=power)
    edge = n_neighbors + n_skip
    pad = [(0, 0)] * (mean_noise.ndim - 1) + [(edge, edge)]
    mean_noise = np.pad(mean_noise, pad_width=pad, constant_values=np.inf)
    with np.errstate(divide="ignore", invalid="ignore"):
        return power / mean_noise


def _on_bin_index(freqs, freq):
    idx = int(np.argmin(np.abs(freqs - freq)))
    return idx, abs(float(freqs[idx]) - freq)


def rect_snr_at(signal, fs, ws_s, freq, coherent=True,
                n_neighbors=SNR_N_NEIGHBORS, n_skip=SNR_N_SKIP):
    """Neighbor-bin SNR at `freq` for a rectangular window of ws_s seconds.

    Returns (snr, n_windows). snr is NaN when the target bin is within the
    neighbor guard of DC/Nyquist (so no clean noise estimate exists) — this is
    what makes very small windows UNUSABLE for a 5 Hz target. When `signal` has a
    leading trial axis the SNR is averaged over trials.
    """
    power_fn = _rect_coherent_power if coherent else _rect_incoherent_power
    freqs, power, n_win = power_fn(signal, fs, ws_s)
    snr = _snr_from_power(power, n_neighbors, n_skip)
    idx, off = _on_bin_index(freqs, freq)
    edge = n_neighbors + n_skip
    if idx < edge + 1 or idx > len(freqs) - 1 - edge:
        return np.nan, n_win
    snr_at_bin = snr[..., idx]
    return float(np.mean(snr_at_bin)), n_win


def rect_snr_fixed_band(signal, fs, ws_s, freq, guard_hz=1.5, band_hz=2.0):
    """SNR with noise estimated over a FIXED Hz band (guard..guard+band on each
    side of `freq`) — comparable across ws, removing the moving-noise-band
    confound. NaN if fewer than 2 noise bins land in the band."""
    freqs, power, _ = _rect_coherent_power(signal, fs, ws_s)
    idx, _ = _on_bin_index(freqs, freq)
    d = np.abs(freqs - freq)
    mask = (d > guard_hz) & (d <= guard_hz + band_hz)
    if int(mask.sum()) < 2:
        return np.nan
    sig = np.mean(np.take(power, idx, axis=-1))
    noise = np.mean(power[..., mask])
    return float(sig / noise) if noise > 0 else np.nan


def rect_onbin_power_norm(signal, fs, ws_s, freq):
    """Coherent on-bin power divided by ws² — flat ⟺ signal power ∝ ws²."""
    freqs, power, _ = _rect_coherent_power(signal, fs, ws_s)
    idx, _ = _on_bin_index(freqs, freq)
    ws = int(round(ws_s * fs))
    return float(np.mean(np.take(power, idx, axis=-1))) / (ws ** 2)


# --- generated signals --------------------------------------------------------

def _pink_noise(n, rng):
    """1/f (pink) noise via frequency-domain 1/sqrt(f) shaping of white noise."""
    white = rng.normal(0.0, 1.0, n)
    spec = np.fft.rfft(white)
    f = np.fft.rfftfreq(n)
    f[0] = f[1]
    return np.fft.irfft(spec / np.sqrt(f), n)


def gen_stationary(fs, dur_s, f_target, f_carrier, rng, noise_sd, pink=False):
    """On-bin target+carrier tones, fixed phase, additive noise (white or pink)."""
    t = np.arange(int(round(dur_s * fs))) / fs
    sig = np.cos(2 * np.pi * f_target * t) + np.cos(2 * np.pi * f_carrier * t)
    noise = _pink_noise(len(t), rng) if pink else rng.normal(0.0, 1.0, len(t))
    return sig + noise_sd * noise


def gen_chirp(fs, dur_s, f_target, sweep_hz, f_carrier, rng, noise_sd):
    """Target is a linear chirp f_target→f_target+sweep_hz over the trial (a
    within-trial frequency drift); carrier stays on-bin. Additive white noise."""
    t = np.arange(int(round(dur_s * fs))) / fs
    inst_phase = 2 * np.pi * (f_target * t + 0.5 * (sweep_hz / dur_s) * t ** 2)
    sig = np.cos(inst_phase) + np.cos(2 * np.pi * f_carrier * t)
    return sig + noise_sd * rng.normal(0.0, 1.0, len(t))


def snr_curve_generated(signal, fs, freq, sweep=SNR_WS_SWEEP, coherent=True):
    """SNR at `freq` across the ws sweep for one generated series. NaN where the
    bin is undefined (target too close to DC)."""
    return np.array([rect_snr_at(signal, fs, w, freq, coherent=coherent)[0]
                     for w in sweep])


# --- real data ----------------------------------------------------------------

def _best_channel_real(raw_tc_time, fs, freq, ws_ref):
    """Channel (index into raw_tc_time's C axis) with the highest trial-mean
    neighbor-SNR at `freq`, measured at the reference window ws_ref."""
    n_ch = raw_tc_time.shape[1]
    snrs = []
    for c in range(n_ch):
        s, _ = rect_snr_at(raw_tc_time[:, c, :], fs, ws_ref, freq)
        snrs.append(-np.inf if not np.isfinite(s) else s)
    return int(np.argmax(snrs))


def collect_snr_vs_ws_real(study, sweep=SNR_WS_SWEEP):
    """Real-data SNR vs ws, aggregated across subjects/conditions.

    Returns a dict with, for the target (5 Hz) and the carrier, arrays over the ws
    sweep of the mean neighbor-SNR at each condition's best channel; plus, for the
    target, a fixed-Hz-band SNR curve and a normalised on-bin signal-power curve
    (for the confound decomposition).
    """
    ws_ref = max(sweep)
    tgt = {w: [] for w in sweep}
    car = {w: [] for w in sweep}
    tgt_fixed = {w: [] for w in sweep}
    tgt_sigpow = {w: [] for w in sweep}
    n_cond = 0
    for subject in _common_subjects(study):
        for f_c in CARRIERS:
            rec = subject[ConditionProperties(
                np.float64(F_TARGET), np.float64(f_c))].take_channels(ELECTRODES)
            fs = rec.sample_rate()
            raw = rec.raw_data()[0]  # (T, C, Time)
            n_cond += 1
            ch_t = _best_channel_real(raw, fs, F_TARGET, ws_ref)
            ch_c = _best_channel_real(raw, fs, f_c, ws_ref)
            for w in sweep:
                st, _ = rect_snr_at(raw[:, ch_t, :], fs, w, F_TARGET)
                sc, _ = rect_snr_at(raw[:, ch_c, :], fs, w, f_c)
                tgt[w].append(st)
                car[w].append(sc)
                tgt_fixed[w].append(
                    rect_snr_fixed_band(raw[:, ch_t, :], fs, w, F_TARGET))
                tgt_sigpow[w].append(
                    rect_onbin_power_norm(raw[:, ch_t, :], fs, w, F_TARGET))

    def _nanmean_curve(d):
        return np.array([np.nanmean(d[w]) if np.any(np.isfinite(d[w]))
                         else np.nan for w in sweep])

    return {
        "sweep": np.array(sweep),
        "target": _nanmean_curve(tgt),
        "carrier": _nanmean_curve(car),
        "target_fixed_band": _nanmean_curve(tgt_fixed),
        "target_sigpow_norm": _nanmean_curve(tgt_sigpow),
        "n_conditions": n_cond,
    }


def fig_snr_vs_window_size(gen, real):
    """Two panels: (A) SNR vs ws for generated-stationary / generated-drift / real
    target & carrier with a flat total-integration-time reference; (B) the real
    confound decomposition (neighbor-band vs fixed-Hz-band SNR, and the flat
    signal-power ∝ ws² baseline)."""
    sweep = np.array(SNR_WS_SWEEP)
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(15, 6),
                                   label="compare-snr-vs-window-size")

    # --- Panel A ---
    axA.plot(sweep, gen["stationary"], "o-", color=COHERENT_COLOR,
             label="generated stationary (white)")
    axA.plot(sweep, gen["chirp"], "s-", color=OLD_COLOR,
             label="generated drift / chirp")
    axA.plot(sweep, real["target"], "^-", color=JOINT_COLOR,
             label="real target 5 Hz")
    axA.plot(sweep, real["carrier"], "v-", color=OLD1_COLOR,
             label="real carrier")
    flat = np.nanmean(gen["stationary"])
    axA.axhline(flat, color=NULL_COLOR, ls="--", lw=1.5,
                label=f"total-integration reference (flat ≈ {flat:.0f})")
    axA.set_xscale("log")
    axA.set_yscale("log")
    axA.set_xlabel("window size ws (s)  — smaller ←→ larger")
    axA.set_ylabel("SNR at the bin")
    axA.set_title("A — SNR vs window size (coherent averaging)\n"
                  "stationary is FLAT (SNR set by total time); drift favours small "
                  "ws; 5 Hz undefined below 1 s", fontsize=10, loc="left")
    axA.legend(fontsize=8)
    axA.grid(True, which="both", linestyle="--", alpha=0.4)

    # --- Panel B: real-target confound decomposition ---
    axB.plot(sweep, real["target"], "^-", color=JOINT_COLOR,
             label="real target: neighbor-bin SNR (moving noise band)")
    axB.plot(sweep, real["target_fixed_band"], "^--", color="#5E35B1",
             label="real target: FIXED-Hz-band SNR (comparable)")
    sig = real["target_sigpow_norm"]
    sig_scaled = sig / np.nanmax(sig) * np.nanmax(real["target"])
    axB.plot(sweep, sig_scaled, "o:", color=COHERENT_COLOR,
             label="on-bin signal power ÷ ws²  (flat ⟹ signal ∝ ws², rescaled)")
    axB.set_xscale("log")
    axB.set_yscale("log")
    axB.set_xlabel("window size ws (s)")
    axB.set_ylabel("SNR  /  rescaled signal power")
    axB.set_title("B — Real 5 Hz: is any trend a real gain or a noise-band artifact?\n"
                  "signal power ∝ ws² (flat when ÷ws²); SNR trend tracks the moving "
                  "noise band, not the signal", fontsize=10, loc="left")
    axB.legend(fontsize=8)
    axB.grid(True, which="both", linestyle="--", alpha=0.4)

    # Honest headline.
    st = gen["stationary"]
    stat_ratio = (np.nanmax(st) / np.nanmin(st)) if np.any(np.isfinite(st)) else np.nan
    fig.suptitle(
        "Does a smaller rectangular window give better SNR?  NO for a stationary "
        "signal.\n"
        "Coherent averaging preserves an on-bin tone (power ∝ ws²) and white-noise "
        "power ∝ ws² too → SNR is FLAT in ws "
        f"(generated-stationary max/min = {stat_ratio:.2f}); SNR is set by TOTAL "
        "integration time.\n"
        "Smaller ws only helps a NON-STATIONARY (drifting) signal, and apparent "
        "real-data gains are the moving neighbor-noise band (1/f), not killed noise.",
        weight="bold", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    return fig


# ==============================================================================
# Driver
# ==============================================================================

def _save(fig):
    label = fig.get_label() or f"figure_{fig.number}"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"{label}.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    print(f"  wrote {path}")


def main(argv):
    rng = np.random.default_rng(20260702)

    print("Synthetic comparisons...")
    for builder in (fig_recovery_vs_true_tau, fig_rmse_vs_noise,
                    fig_anchor_sensitivity, fig_per_trial_flips,
                    fig_carrier_invariance):
        _save(builder(rng))

    # SNR vs window size — generated part (self-contained).
    print("SNR vs window size (generated)...")
    snr_rng = np.random.default_rng(20260702)
    fs_gen, dur_gen, n_reps = 300.0, 60.0, 40

    def _avg_curve(make_signal):
        """Mean SNR-vs-ws curve over n_reps noise realisations (nan-aware)."""
        curves = [snr_curve_generated(make_signal(snr_rng), fs_gen, F_TARGET)
                  for _ in range(n_reps)]
        return np.nanmean(np.vstack(curves), axis=0)

    gen_snr = {
        "stationary": _avg_curve(
            lambda r: gen_stationary(fs_gen, dur_gen, F_TARGET, 10.0, r,
                                     noise_sd=6.0)),
        "chirp": _avg_curve(
            lambda r: gen_chirp(fs_gen, dur_gen, F_TARGET, sweep_hz=1.5,
                                f_carrier=10.0, rng=r, noise_sd=1.0)),
        "pink": _avg_curve(
            lambda r: gen_stationary(fs_gen, dur_gen, F_TARGET, 10.0, r,
                                     noise_sd=6.0, pink=True)),
    }
    for name in ("stationary", "chirp", "pink"):
        pretty = {round(w, 1): (None if not np.isfinite(v) else round(float(v), 1))
                  for w, v in zip(SNR_WS_SWEEP, gen_snr[name])}
        print(f"  {name:>11}: {pretty}")

    if len(argv) >= 2:
        root = argv[1]
        if not os.path.isdir(root):
            print(f"Real-data folder {root!r} not found; skipping real comparisons.")
            return
        print(f"Loading study from {root} for real-data comparisons...")
        study = Study(StudyLoader().load(root))
        rows = collect_real(study)
        if not rows:
            print("No subjects with all three carriers; skipping real figures.")
            return
        print(f"  {len(rows)} subjects with all carriers")
        _save(fig_real_scatter(rows))
        _save(fig_real_agreement_curves(rows))
        _save(fig_real_disagreement(rows))
        _save(fig_null_vs_real_confidence(rows))

        print("Within-trial split-half stationarity...")
        agg, meta = collect_stationarity(study)
        wc = meta["window_counts"]
        print(f"  {meta['n_units']} subject×carrier×trial units; "
              f"non-overlapping windows/trial {min(wc)}–{max(wc)}; "
              f"de-overlap k∈{sorted(set(meta['ks']))} "
              f"(inferred hop≈{np.mean(meta['hops']):.0f} samples)")
        report = [("residue (ms)", "residue", 1e3),
                  ("target 5Hz (rad)", "target", 1.0),
                  ("carrier (rad)", "carrier", 1.0),
                  ("carrier (ms)", "carrier_ms", 1.0)]
        for label, key, scale in report:
            med_t, med_r, ratio, p, verdict = stationarity_verdict(
                agg[key]["temporal"], agg[key]["random"])
            print(f"    {label:>18}: temporal={med_t*scale:.4g} "
                  f"random={med_r*scale:.4g} ratio={ratio:.2f} "
                  f"p={p:.3g} -> {verdict}")
        _save(fig_within_trial_stationarity(agg, meta))

        print("SNR vs window size (real)...")
        real_snr = collect_snr_vs_ws_real(study)
        for name in ("target", "carrier", "target_fixed_band"):
            pretty = {round(w, 1): (None if not np.isfinite(v) else round(float(v), 1))
                      for w, v in zip(SNR_WS_SWEEP, real_snr[name])}
            print(f"  {name:>18}: {pretty}")
        _save(fig_snr_vs_window_size(gen_snr, real_snr))
    else:
        print("No experiment folder given; skipping real-data comparisons.")


if __name__ == "__main__":
    main(sys.argv)
