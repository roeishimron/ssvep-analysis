"""Joint multi-carrier processing-time estimation for SSVEP.

Estimates one neural processing latency τ per subject by combining all
carrier conditions jointly, using a coherence-weighted phase-agreement curve.
Unlike the per-carrier estimator this is unambiguous over [0, 200) ms and
avoids per-trial hard cycle decisions.

Pure layer (residue_phase, agreement_curve, estimate_joint_latency):
  - numpy-only, no FFT or IO
  - fully unit-testable with hand-constructed phasors

Data-facing wrapper (subject_joint_latency):
  - uses SSVEPAnalysis.phase_at via the SubjectHandle / ConditionProperties API
"""

from dataclasses import dataclass

import numpy as np

from analysis import SSVEPAnalysis
from core_types import ConditionProperties
from interfaces import SubjectHandle


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class JointLatencyResult:
    """Immutable result of estimate_joint_latency."""

    tau_s: float          # τ*  (seconds)
    sigma_s: float        # curvature-based 1σ uncertainty (seconds)
    dominance: float      # peak / runner-up local max (>= 1)
    fit: float            # A(τ*) / Σ weights; 1.0 = perfect multi-carrier agreement
    reliable: bool        # dominance >= dominance_min AND fit >= fit_min
    tau_grid: np.ndarray  # (G,) seconds
    curve: np.ndarray     # (G,) A(τ) over the grid
    carrier_freqs: np.ndarray   # (C,) Hz, sorted
    residue_phases: np.ndarray  # (C,) ψ_c radians in (-π, π]
    weights: np.ndarray         # (C,) κ_c used


# ---------------------------------------------------------------------------
# Pure layer
# ---------------------------------------------------------------------------

def residue_phase(
    target_phasor: np.ndarray,   # complex, any shape (per-trial), for ONE carrier
    carrier_phasor: np.ndarray,  # complex, same shape
    target_frequency: float,
    carrier_frequency: float,
) -> tuple:
    """Coherence-weighted aggregated (ψ_c, R_c) for one carrier. w_{c,t}=1.

    The δ-invariant observable per trial is:
        ψ_{c,t} = M_c · angle(φ_target) − angle(φ_carrier)   (mod 2π)
    equivalently:
        z_{c,t} = exp(i · ψ_{c,t}) = norm_target^{M_c} · conj(norm_carrier)

    The aggregated result is the circular mean of the per-trial unit phasors:
        Z_c     = mean_t(z_{c,t})
        ψ_c     = angle(Z_c)            in (-π, π]
        R_c     = |Z_c|                 in [0, 1]  (resultant length / coherence)

    Returns
    -------
    (psi_c, R_c) : tuple[float, float]

    Raises
    ------
    ValueError
        If carrier_frequency / target_frequency is not a near-integer M_c >= 1.
    """
    ratio = carrier_frequency / target_frequency
    M_c = int(round(ratio))
    if M_c < 1 or abs(ratio - M_c) > 1e-6:
        raise ValueError(
            f"carrier_frequency / target_frequency = {ratio:.6g} is not a "
            f"near-integer M_c >= 1; got M_c = {M_c}"
        )

    target_phasor = np.asarray(target_phasor, dtype=np.complex128)
    carrier_phasor = np.asarray(carrier_phasor, dtype=np.complex128)

    # Unit phasors: strip magnitude so only phase information is used.
    norm_target = target_phasor / np.abs(target_phasor)
    norm_carrier = carrier_phasor / np.abs(carrier_phasor)

    # Per-trial residue phasor (δ-invariant).
    z = (norm_target ** M_c) * np.conj(norm_carrier)

    # Coherence-weighted circular mean (w_{c,t} = 1 ⟹ plain mean).
    Z_c = np.mean(z)
    psi_c = float(np.angle(Z_c))
    R_c = float(np.abs(Z_c))

    return psi_c, R_c


def agreement_curve(
    carrier_freqs: np.ndarray,   # (C,) Hz
    residue_phases: np.ndarray,  # (C,) radians
    weights: np.ndarray,         # (C,)
    tau_grid: np.ndarray,        # (G,) seconds
) -> np.ndarray:                 # (G,) A(τ)
    """Evaluate the multi-carrier agreement curve.

        A(τ) = Σ_c  κ_c · cos( 2π f_c τ − ψ_c )

    Parameters
    ----------
    carrier_freqs : (C,) Hz
    residue_phases : (C,) radians — aggregated ψ_c per carrier
    weights : (C,) — κ_c per carrier (e.g. resultant lengths R_c)
    tau_grid : (G,) seconds — evaluation points

    Returns
    -------
    (G,) float64 — A(τ) at each grid point
    """
    carrier_freqs = np.asarray(carrier_freqs, dtype=np.float64)
    residue_phases = np.asarray(residue_phases, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    tau_grid = np.asarray(tau_grid, dtype=np.float64)

    # Shape: (C, 1) broadcast with (1, G) → (C, G); sum over C → (G,).
    phase = (
        2.0 * np.pi * carrier_freqs[:, np.newaxis] * tau_grid[np.newaxis, :]
        - residue_phases[:, np.newaxis]
    )
    return np.sum(weights[:, np.newaxis] * np.cos(phase), axis=0)


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
    """Estimate processing time from the multi-carrier agreement curve.

    Finds τ* = argmax A(τ) over a uniform grid, then computes:
      - dominance  : A(τ*) / A(second-highest local maximum)
      - fit        : A(τ*) / Σ_c κ_c  (normalised peak height; 1.0 = perfect agreement)
      - sigma_s    : curvature-based uncertainty (parabolic fit at τ*)

    Parameters
    ----------
    carrier_freqs : (C,) Hz — carrier frequencies
    residue_phases : (C,) radians — aggregated residue phases ψ_c
    weights : (C,) — per-carrier trust weights κ_c (e.g. R_c)
    tau_min, tau_max : float — grid extent in seconds (inclusive endpoints)
    tau_step : float — grid step in seconds
    dominance_min : float — dominance threshold for reliability flag
    fit_min : float — fit threshold for reliability flag; reliable requires
        both dominance >= dominance_min AND fit >= fit_min

    Returns
    -------
    JointLatencyResult
    """
    carrier_freqs = np.asarray(carrier_freqs, dtype=np.float64)
    residue_phases = np.asarray(residue_phases, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)

    tau_grid = np.arange(tau_min, tau_max + tau_step * 0.5, tau_step)
    curve = agreement_curve(carrier_freqs, residue_phases, weights, tau_grid)

    # --- global peak ---
    peak_idx = int(np.argmax(curve))
    tau_star = float(tau_grid[peak_idx])
    A_peak = float(curve[peak_idx])

    # --- local maxima for dominance guard ---
    # A point is a local maximum if it is strictly greater than both neighbours.
    n = len(curve)
    is_local_max = np.zeros(n, dtype=bool)
    if n >= 3:
        is_local_max[1:-1] = (curve[1:-1] > curve[:-2]) & (curve[1:-1] > curve[2:])
    is_local_max[0] = (n == 1) or (curve[0] > curve[1])
    is_local_max[-1] = (n == 1) or (curve[-1] > curve[-2])

    local_max_vals = curve[is_local_max]

    if len(local_max_vals) >= 2:
        # Sort descending; the top two are peak and runner-up.
        sorted_vals = np.sort(local_max_vals)[::-1]
        runner_up_val = float(sorted_vals[1])
        if runner_up_val > 0.0:
            dominance = A_peak / runner_up_val
        elif A_peak > 0.0:
            dominance = np.inf
        else:
            dominance = 1.0
    else:
        # Single local maximum — no runner-up; treat as perfectly dominant.
        dominance = np.inf

    # --- goodness-of-fit: normalised peak height ---
    total_weight = float(np.sum(weights))
    fit = A_peak / total_weight if total_weight != 0.0 else 0.0

    reliable = bool((dominance >= dominance_min) and (fit >= fit_min))

    # --- curvature-based uncertainty (parabolic fit at peak) ---
    if 0 < peak_idx < n - 1:
        y_m = curve[peak_idx - 1]
        y_0 = curve[peak_idx]
        y_p = curve[peak_idx + 1]
        # Second-order finite-difference approximation of d²A/dτ².
        d2 = (y_m - 2.0 * y_0 + y_p) / (tau_step ** 2)
        # At a genuine maximum d2 < 0.  σ² = -1 / d2 (Laplace approximation).
        if d2 < 0.0:
            sigma_tau = float(np.sqrt(-1.0 / d2))
        else:
            # Plateau or numerical coincidence — uncertainty is not defined.
            sigma_tau = np.inf
    else:
        # Peak is at a grid boundary — no neighbours on one side.
        sigma_tau = np.inf

    return JointLatencyResult(
        tau_s=tau_star,
        sigma_s=sigma_tau,
        dominance=float(dominance),
        fit=float(fit),
        reliable=reliable,
        tau_grid=tau_grid,
        curve=curve,
        carrier_freqs=np.sort(carrier_freqs),
        residue_phases=residue_phases[np.argsort(carrier_freqs)],
        weights=weights[np.argsort(carrier_freqs)],
    )


# ---------------------------------------------------------------------------
# Data-facing wrapper
# ---------------------------------------------------------------------------

def subject_joint_latency(
    subject: SubjectHandle,
    carriers: list,         # list[float], e.g. [10.0, 15.0, 20.0]
    electrodes: list,       # list[str]
    target_frequency: float = 5.0,
    **estimator_kwargs,
) -> JointLatencyResult:
    """Estimate joint latency for one subject across multiple carrier conditions.

    For each carrier frequency:
      1. Retrieve the condition recording via subject[ConditionProperties(...)].
      2. Restrict to the requested electrodes via take_channels(electrodes).
      3. Wrap in SSVEPAnalysis and call phase_at(target_frequency) and
         phase_at(carrier) to obtain per-trial phasors.
      4. Squeeze the single-subject axis (S=1 → T) and compute residue_phase.

    The resultant lengths R_c are used as per-carrier weights κ_c.
    Additional keyword arguments are forwarded to estimate_joint_latency.

    Parameters
    ----------
    subject : SubjectHandle — one participant's per-condition recordings
    carriers : list[float] — carrier frequencies to combine (Hz)
    electrodes : list[str] — channel names to pass to take_channels
    target_frequency : float — SSVEP target frequency (Hz), default 5.0
    **estimator_kwargs — forwarded to estimate_joint_latency
        (tau_min, tau_max, tau_step, dominance_min, fit_min)

    Returns
    -------
    JointLatencyResult
    """
    all_carrier_freqs: list = []
    all_psi: list = []
    all_R: list = []

    for carrier in carriers:
        key = ConditionProperties(
            target_frequency=np.float64(target_frequency),
            carrier_frequency=np.float64(carrier),
        )
        rec = subject[key].take_channels(list(electrodes))
        ana = SSVEPAnalysis(rec)

        # phase_at returns (S, T) complex; for single-subject S=1 → squeeze.
        target_phasors, _ = ana.phase_at(target_frequency)  # (S, T)
        carrier_phasors, _ = ana.phase_at(float(carrier))   # (S, T)
        if target_phasors.shape[0] != 1:
            raise ValueError(
                f"subject_joint_latency expects a single-subject view (S=1), "
                f"got S={target_phasors.shape[0]}"
            )
        t_vec = target_phasors[0]   # (T,)
        c_vec = carrier_phasors[0]  # (T,)

        psi_c, R_c = residue_phase(t_vec, c_vec, target_frequency, float(carrier))
        all_carrier_freqs.append(float(carrier))
        all_psi.append(psi_c)
        all_R.append(R_c)

    return estimate_joint_latency(
        carrier_freqs=np.array(all_carrier_freqs, dtype=np.float64),
        residue_phases=np.array(all_psi, dtype=np.float64),
        weights=np.array(all_R, dtype=np.float64),
        **estimator_kwargs,
    )
