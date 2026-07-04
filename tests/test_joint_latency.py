"""Integration tests for joint_latency.py.

Covers:
  - Pure layer: residue_phase, agreement_curve, estimate_joint_latency
  - Data-facing wrapper: subject_joint_latency end-to-end
  - Cross-check: single-carrier joint estimate vs SSVEPAnalysis.processing_time_summary
  - Edge cases: κ=0 carrier, conflicting carriers, curvature uncertainty
"""
import pytest
from unittest.mock import MagicMock, patch

import numpy as np

from analysis import SSVEPAnalysis
from core_types import ConditionProperties
from interfaces import SSVEPRecording
from joint_latency import (
    JointLatencyResult,
    agreement_curve,
    estimate_joint_latency,
    residue_phase,
    subject_joint_latency,
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _clean_phasors(tau_s: float, f_c: float, f_t: float, n_trials: int = 1):
    """Return (target_phasors (n_trials,), carrier_phasors (n_trials,)) encoding τ.

    Convention: target phase = 0, carrier phase = -2π f_c τ, so that
        ψ_c = M_c * 0 − (−2π f_c τ) = 2π f_c τ   (the spec's latency relation).
    """
    t_ph = np.ones(n_trials, dtype=np.complex128)
    c_ph = np.full(n_trials, np.exp(-1j * 2 * np.pi * f_c * tau_s), dtype=np.complex128)
    return t_ph, c_ph


def _ssvep_analysis_with_fourier(
    target_f, carrier_f, target_phase_rad, carrier_phase_rad,
):
    """Build an SSVEPAnalysis whose phase_at returns predetermined phasors.

    Mirrors the helper in test_processing_time.py; reproduced here so
    this module has no cross-test import dependency.
    """
    n_freqs = 30
    freqs_array = np.linspace(1, 30, n_freqs).astype(np.float64)
    t_idx = int(np.argmin(np.abs(freqs_array - target_f)))
    c_idx = int(np.argmin(np.abs(freqs_array - carrier_f)))

    fourier = np.zeros((1, 1, 1, 1, n_freqs), dtype=np.complex64)
    fourier[0, 0, 0, 0, t_idx] = np.exp(1j * target_phase_rad)
    fourier[0, 0, 0, 0, c_idx] = np.exp(1j * carrier_phase_rad)

    rec = MagicMock(spec=SSVEPRecording)
    rec.props.return_value = ConditionProperties(
        target_frequency=np.float64(target_f),
        carrier_frequency=np.float64(carrier_f),
    )
    rec.sample_rate.return_value = 100.0

    ana = SSVEPAnalysis.__new__(SSVEPAnalysis)
    ana.rec = rec
    ana.window_duration_s = 1.0
    ana.n_neighbors = 3
    ana.n_skip = 1
    ana._window_size = 100
    ana._fourier_cache = fourier
    ana.expected_latency_s = 0.05
    ana.frequencies = lambda: freqs_array
    return ana


# ---------------------------------------------------------------------------
# residue_phase — pure unit tests
# ---------------------------------------------------------------------------

class TestResiduePhase:

    def test_sanity_anchor_from_spec(self):
        """Spec §2 anchor: f_t=5, f_c=10, φ_target=0, φ_carrier=1.2π → τ=40 ms."""
        t_ph = np.array([np.exp(0j)])
        c_ph = np.array([np.exp(1j * 1.2 * np.pi)])
        psi, R = residue_phase(t_ph, c_ph, 5.0, 10.0)
        # ψ = M*0 − 1.2π = −1.2π ≡ 0.8π;  τ = 0.8π / (2π·10) = 0.04 s
        expected_psi = 0.8 * np.pi  # in (−π, π]: 0.8π < π, OK
        # np.angle wraps −1.2π to 0.8π (since −1.2π + 2π = 0.8π)
        assert abs(psi - expected_psi) < 1e-6
        assert abs(R - 1.0) < 1e-6

    def test_latency_relation_all_carriers(self):
        """ψ_c = 2π f_c τ (mod 2π, in (−π,π]) for f_c ∈ {10,15,20}, τ=80 ms."""
        tau = 0.08
        f_t = 5.0
        for f_c in [10.0, 15.0, 20.0]:
            t_ph, c_ph = _clean_phasors(tau, f_c, f_t, n_trials=1)
            psi, R = residue_phase(t_ph, c_ph, f_t, f_c)
            raw = (2 * np.pi * f_c * tau) % (2 * np.pi)
            expected = raw - 2 * np.pi if raw > np.pi else raw
            assert abs(psi - expected) < 1e-6, f"f_c={f_c}: psi={psi:.6f}, expected={expected:.6f}"
            assert abs(R - 1.0) < 1e-6

    def test_delta_invariance(self):
        """Per-trial window offset δ cancels in ψ_c (δ-invariance, spec §2)."""
        tau = 0.05
        f_t, f_c = 5.0, 15.0
        n_trials = 10
        rng = np.random.default_rng(7)
        deltas = rng.uniform(0, 0.2, size=n_trials)

        t_ph0, c_ph0 = _clean_phasors(tau, f_c, f_t, n_trials)
        psi_ref, R_ref = residue_phase(t_ph0, c_ph0, f_t, f_c)

        # Shift each trial by its own δ
        t_shifted = t_ph0 * np.exp(1j * 2 * np.pi * f_t * deltas)
        c_shifted = c_ph0 * np.exp(1j * 2 * np.pi * f_c * deltas)
        psi_shifted, R_shifted = residue_phase(t_shifted, c_shifted, f_t, f_c)

        assert abs(psi_shifted - psi_ref) < 1e-6
        assert abs(R_shifted - R_ref) < 1e-6

    def test_coherence_less_than_one_with_noise(self):
        """Noisy phasors (von Mises phase scatter) give R_c < 1."""
        tau = 0.06
        f_t, f_c = 5.0, 10.0
        n_trials = 200
        rng = np.random.default_rng(13)

        t_ph, c_ph = _clean_phasors(tau, f_c, f_t, n_trials)
        # Add ~30° phase noise to carrier
        noise = np.exp(1j * rng.normal(0, np.pi / 6, n_trials))
        c_ph_noisy = c_ph * noise

        _, R = residue_phase(t_ph, c_ph_noisy, f_t, f_c)
        assert 0 < R < 1.0

    def test_invalid_ratio_raises(self):
        """Non-integer M_c raises ValueError."""
        with pytest.raises(ValueError, match="near-integer"):
            residue_phase(np.array([1.0 + 0j]), np.array([1.0 + 0j]), 5.0, 7.0)


# ---------------------------------------------------------------------------
# agreement_curve — pure unit test
# ---------------------------------------------------------------------------

class TestAgreementCurve:

    def test_peaks_at_true_tau_single_carrier(self):
        """For one noiseless carrier, A(τ) has its first peak at τ_true."""
        tau_true = 0.06
        f_c = np.array([10.0])
        t_ph, c_ph = _clean_phasors(tau_true, 10.0, 5.0, n_trials=1)
        psi, _ = residue_phase(t_ph, c_ph, 5.0, 10.0)

        tau_grid = np.arange(0.0, 0.101, 1e-3)
        curve = agreement_curve(f_c, np.array([psi]), np.array([1.0]), tau_grid)
        peak_tau = tau_grid[np.argmax(curve)]

        assert abs(peak_tau - tau_true) <= 1e-3

    def test_zero_weight_carrier_contributes_nothing(self):
        """κ=0 carrier leaves the curve unchanged from the single-carrier case."""
        tau_true = 0.05
        psi_good, _ = residue_phase(
            *_clean_phasors(tau_true, 10.0, 5.0, 1), 5.0, 10.0,
        )
        psi_bad, _ = residue_phase(
            *_clean_phasors(0.15, 15.0, 5.0, 1), 5.0, 15.0,
        )  # encodes a conflicting τ

        tau_grid = np.arange(0.0, 0.201, 1e-3)
        curve_single = agreement_curve(
            np.array([10.0]), np.array([psi_good]), np.array([1.0]), tau_grid,
        )
        curve_with_zero = agreement_curve(
            np.array([10.0, 15.0]),
            np.array([psi_good, psi_bad]),
            np.array([1.0, 0.0]),  # bad carrier zeroed out
            tau_grid,
        )

        np.testing.assert_allclose(curve_with_zero, curve_single, atol=1e-12)


# ---------------------------------------------------------------------------
# estimate_joint_latency — pure integration tests
# ---------------------------------------------------------------------------

class TestEstimateJointLatency:

    def test_three_carriers_recover_tau_120ms(self):
        """τ_true=120 ms is recovered unambiguously from 3 carriers (§8 check).

        120 ms > T_20=50 ms and > T_15≈67 ms; a single carrier would be
        ambiguous without an anchor.
        """
        tau_true = 0.12
        f_t = 5.0
        carriers = [10.0, 15.0, 20.0]

        psi_list = []
        for f_c in carriers:
            t_ph, c_ph = _clean_phasors(tau_true, f_c, f_t, 1)
            psi, _ = residue_phase(t_ph, c_ph, f_t, f_c)
            psi_list.append(psi)

        result = estimate_joint_latency(
            carrier_freqs=np.array(carriers),
            residue_phases=np.array(psi_list),
            weights=np.ones(3),
            tau_min=0.0,
            tau_max=0.2,
            tau_step=1e-3,
        )

        assert abs(result.tau_s - tau_true) <= 1e-3, (
            f"Expected ~120 ms, got {result.tau_s * 1000:.1f} ms"
        )
        assert result.reliable

    @pytest.mark.parametrize("tau_true_ms", [10, 30, 50, 80, 110, 150, 190])
    def test_joint_sweep_across_range(self, tau_true_ms):
        """τ* recovers τ_true across a sweep of [0, 200) ms."""
        tau_true = tau_true_ms / 1000.0
        f_t = 5.0
        carriers = [10.0, 15.0, 20.0]

        psi_list = []
        for f_c in carriers:
            t_ph, c_ph = _clean_phasors(tau_true, f_c, f_t, 1)
            psi, _ = residue_phase(t_ph, c_ph, f_t, f_c)
            psi_list.append(psi)

        result = estimate_joint_latency(
            carrier_freqs=np.array(carriers),
            residue_phases=np.array(psi_list),
            weights=np.ones(3),
            tau_min=0.0,
            tau_max=0.2,
            tau_step=1e-3,
        )

        assert abs(result.tau_s - tau_true) <= 1e-3, (
            f"tau_true={tau_true_ms}ms: got {result.tau_s * 1000:.1f}ms"
        )

    def test_kappa_zero_carrier_does_not_move_tau(self):
        """A κ=0 carrier (conflicting latency) must not displace τ*."""
        tau_true = 0.05
        f_t = 5.0

        psi_good, _ = residue_phase(*_clean_phasors(tau_true, 10.0, f_t, 1), f_t, 10.0)
        psi_conflict, _ = residue_phase(*_clean_phasors(0.15, 15.0, f_t, 1), f_t, 15.0)

        result = estimate_joint_latency(
            carrier_freqs=np.array([10.0, 15.0]),
            residue_phases=np.array([psi_good, psi_conflict]),
            weights=np.array([1.0, 0.0]),  # conflict zeroed out
            tau_min=0.0,
            tau_max=0.1,
            tau_step=1e-3,
        )

        assert abs(result.tau_s - tau_true) <= 1e-3

    def test_single_carrier_ambiguous_flags_unreliable(self):
        """One carrier with tau_max > T_c yields 2 equal-amplitude aliases → dominance=1.

        This is the guaranteed-reliable=False construction: f_c=10, T_c=100 ms.
        With tau_max=0.2 s, the peaks at τ* and τ*+100 ms are identical in the
        agreement curve (same cosine value), so dominance = 1.0 < 1.1 (default).
        """
        tau_true = 0.04
        f_t, f_c = 5.0, 10.0

        psi, _ = residue_phase(*_clean_phasors(tau_true, f_c, f_t, 1), f_t, f_c)

        result = estimate_joint_latency(
            carrier_freqs=np.array([f_c]),
            residue_phases=np.array([psi]),
            weights=np.array([1.0]),
            tau_min=0.0,
            tau_max=0.2,   # two full periods of f_c → two equal peaks
            tau_step=1e-3,
            dominance_min=1.1,
        )

        assert not result.reliable, (
            f"Expected unreliable (single carrier, 2 aliases); dominance={result.dominance:.3f}"
        )
        # Both alias peaks are equal so dominance should be exactly 1.
        assert abs(result.dominance - 1.0) < 1e-9

    def test_sigma_finite_for_interior_peak(self):
        """σ_τ is a finite positive number when the peak is not at a grid boundary."""
        tau_true = 0.08
        f_t = 5.0
        carriers = [10.0, 15.0, 20.0]

        psi_list = []
        for f_c in carriers:
            psi, _ = residue_phase(*_clean_phasors(tau_true, f_c, f_t, 1), f_t, f_c)
            psi_list.append(psi)

        result = estimate_joint_latency(
            carrier_freqs=np.array(carriers),
            residue_phases=np.array(psi_list),
            weights=np.ones(3),
        )

        assert np.isfinite(result.sigma_s), "sigma_s must be finite"
        assert result.sigma_s > 0

    def test_result_fields_populated(self):
        """JointLatencyResult has the required fields and sorted carrier_freqs."""
        tau_true = 0.07
        f_t = 5.0
        carriers = [20.0, 10.0, 15.0]  # unsorted input

        psi_list, w_list = [], []
        for f_c in carriers:
            psi, R = residue_phase(*_clean_phasors(tau_true, f_c, f_t, 3), f_t, f_c)
            psi_list.append(psi)
            w_list.append(R)

        result = estimate_joint_latency(
            carrier_freqs=np.array(carriers),
            residue_phases=np.array(psi_list),
            weights=np.array(w_list),
        )

        assert len(result.tau_grid) == len(result.curve)
        assert len(result.carrier_freqs) == 3
        # carrier_freqs must be sorted ascending
        np.testing.assert_array_equal(result.carrier_freqs, np.sort(carriers))
        assert result.dominance >= 1.0

    def test_single_carrier_still_works(self):
        """estimate_joint_latency with one carrier produces a finite result."""
        tau_true = 0.04
        f_t = 5.0
        psi, _ = residue_phase(*_clean_phasors(tau_true, 10.0, f_t, 1), f_t, 10.0)

        result = estimate_joint_latency(
            carrier_freqs=np.array([10.0]),
            residue_phases=np.array([psi]),
            weights=np.array([1.0]),
            tau_min=0.0,
            tau_max=0.1,
            tau_step=1e-3,
        )

        assert abs(result.tau_s - tau_true) <= 1e-3


# ---------------------------------------------------------------------------
# subject_joint_latency — data-facing wrapper integration tests
# ---------------------------------------------------------------------------

def _make_subject_mock(tau_true, carriers, target_freq=5.0, n_trials=4):
    """Build a SubjectHandle mock and a carrier→SSVEPAnalysis mock factory.

    Returns (subject, ana_factory) where ana_factory(rec) returns a mock
    SSVEPAnalysis whose phase_at returns synthetic phasors encoding tau_true.
    """
    # Per-carrier recording mocks.
    def make_rec(carrier):
        rec = MagicMock(spec=SSVEPRecording)
        rec.props.return_value = ConditionProperties(
            target_frequency=np.float64(target_freq),
            carrier_frequency=np.float64(carrier),
        )
        rec.sample_rate.return_value = 100.0
        rec.take_channels.return_value = rec
        return rec

    rec_by_carrier = {float(c): make_rec(float(c)) for c in carriers}

    subject = MagicMock()
    subject.name = "S_test"
    subject.id = 0

    def getitem(key):
        return rec_by_carrier[float(key.carrier_frequency)]

    subject.__getitem__.side_effect = getitem

    # SSVEPAnalysis mock factory: reads carrier from rec.props().
    def make_ana(rec):
        carrier = float(rec.props().carrier_frequency)
        ana = MagicMock()
        t_phasors = np.ones((1, n_trials), dtype=np.complex128)
        c_phasors = np.tile(
            np.exp(-1j * 2 * np.pi * carrier * tau_true),
            (1, n_trials),
        ).astype(np.complex128)

        def phase_at(freq):
            if abs(freq - target_freq) < 0.1:
                return t_phasors.copy(), 1.0 / target_freq
            return c_phasors.copy(), 1.0 / carrier

        ana.phase_at.side_effect = phase_at
        return ana

    return subject, make_ana


class TestSubjectJointLatency:

    def test_end_to_end_three_carriers(self):
        """subject_joint_latency recovers τ_true=80 ms from 3 noiseless carriers."""
        tau_true = 0.08
        carriers = [10.0, 15.0, 20.0]
        subject, make_ana = _make_subject_mock(tau_true, carriers)

        with patch("joint_latency.SSVEPAnalysis", side_effect=make_ana):
            result = subject_joint_latency(
                subject=subject,
                carriers=carriers,
                electrodes=["O1"],
                target_frequency=5.0,
                tau_min=0.0,
                tau_max=0.2,
                tau_step=1e-3,
            )

        assert isinstance(result, JointLatencyResult)
        assert abs(result.tau_s - tau_true) <= 1e-3, (
            f"Expected ~80 ms, got {result.tau_s * 1000:.1f} ms"
        )
        assert result.reliable
        np.testing.assert_array_equal(result.carrier_freqs, np.sort(carriers))

    def test_end_to_end_tau_120ms_three_carriers(self):
        """τ=120 ms (> T_20) recovered unambiguously via 3-carrier joint estimate."""
        tau_true = 0.12
        carriers = [10.0, 15.0, 20.0]
        subject, make_ana = _make_subject_mock(tau_true, carriers)

        with patch("joint_latency.SSVEPAnalysis", side_effect=make_ana):
            result = subject_joint_latency(
                subject=subject,
                carriers=carriers,
                electrodes=["Oz", "O1"],
                target_frequency=5.0,
                tau_min=0.0,
                tau_max=0.2,
                tau_step=1e-3,
            )

        assert abs(result.tau_s - tau_true) <= 1e-3, (
            f"Expected ~120 ms, got {result.tau_s * 1000:.1f} ms"
        )

    def test_weights_are_resultant_lengths(self):
        """Weights returned by wrapper equal R_c (resultant length from residue_phase)."""
        tau_true = 0.05
        carriers = [10.0, 15.0]
        subject, make_ana = _make_subject_mock(tau_true, carriers, n_trials=1)

        with patch("joint_latency.SSVEPAnalysis", side_effect=make_ana):
            result = subject_joint_latency(
                subject=subject,
                carriers=carriers,
                electrodes=["O1"],
                target_frequency=5.0,
            )

        # Noiseless phasors → R_c = 1.0 for all carriers.
        np.testing.assert_allclose(result.weights, 1.0, atol=1e-6)


# ---------------------------------------------------------------------------
# Cross-check: joint estimate vs processing_time_summary on clean data
# ---------------------------------------------------------------------------

class TestCrossCheckVsProcessingTimeSummary:

    def test_single_carrier_spec_anchor_agrees(self):
        """Spec §2 anchor case: joint τ* agrees with processing_time_summary to 1 ms.

        f_t=5, f_c=10, φ_target=0, φ_carrier=1.2π → both should give τ=40 ms.
        Tight window tau_max=0.1 s eliminates single-carrier ambiguity.
        """
        ana = _ssvep_analysis_with_fourier(5.0, 10.0, 0.0, 1.2 * np.pi)

        # Reference: SSVEPAnalysis.processing_time_summary for subject 0.
        mean_ms, _ = ana.processing_time_summary()
        ref_tau_s = float(mean_ms[0]) / 1000.0

        # Same phasors → joint estimate.
        target_phasors, _ = ana.phase_at(5.0)   # (1, 1)
        carrier_phasors, _ = ana.phase_at(10.0)  # (1, 1)
        psi_c, R_c = residue_phase(
            target_phasors[0], carrier_phasors[0], 5.0, 10.0,
        )

        result = estimate_joint_latency(
            carrier_freqs=np.array([10.0]),
            residue_phases=np.array([psi_c]),
            weights=np.array([R_c]),
            tau_min=0.0,
            tau_max=0.1,   # tight: eliminates the 140 ms alias
            tau_step=1e-3,
        )

        assert abs(result.tau_s - ref_tau_s) <= 1e-3, (
            f"Joint={result.tau_s * 1000:.2f} ms vs summary={ref_tau_s * 1000:.2f} ms"
        )

    def test_single_carrier_3to1_ratio_agrees(self):
        """Joint τ* agrees with processing_time_summary for f_c=15 (M=3).

        Mirrors test_processing_time.py Case test_processing_time_3to1_ratio:
        carrier_t=10 ms → expected latency ≈ T_15 − 10 ms ≈ 56.7 ms.
        """
        carrier_phase = (0.01 / (1 / 15.0)) * 2 * np.pi
        ana = _ssvep_analysis_with_fourier(5.0, 15.0, 0.0, carrier_phase)

        mean_ms, _ = ana.processing_time_summary()
        ref_tau_s = float(mean_ms[0]) / 1000.0

        target_phasors, _ = ana.phase_at(5.0)
        carrier_phasors, _ = ana.phase_at(15.0)
        psi_c, R_c = residue_phase(
            target_phasors[0], carrier_phasors[0], 5.0, 15.0,
        )

        # Tight window: [0, T_15] = [0, 1/15] ≈ [0, 66.7 ms].
        result = estimate_joint_latency(
            carrier_freqs=np.array([15.0]),
            residue_phases=np.array([psi_c]),
            weights=np.array([R_c]),
            tau_min=0.0,
            tau_max=1.0 / 15.0,
            tau_step=1e-3,
        )

        assert abs(result.tau_s - ref_tau_s) <= 1e-3, (
            f"Joint={result.tau_s * 1000:.2f} ms vs summary={ref_tau_s * 1000:.2f} ms"
        )


# ---------------------------------------------------------------------------
# TestFitMetric — fit field and fit gate (design §4)
# ---------------------------------------------------------------------------

class TestFitMetric:

    def test_fit_one_noiseless_multi_carrier(self):
        """Noiseless 3-carrier estimate → fit==1.0 and reliable==True.

        When all carriers agree perfectly at τ_true, every cosine peaks at 1
        so A(τ*) = Σ_c κ_c and fit = A(τ*)/Σ κ_c = 1.0 (spec §4).
        """
        tau_true = 0.08
        f_t = 5.0
        carriers = [10.0, 15.0, 20.0]

        psi_list = []
        for f_c in carriers:
            t_ph, c_ph = _clean_phasors(tau_true, f_c, f_t, n_trials=1)
            psi, _ = residue_phase(t_ph, c_ph, f_t, f_c)
            psi_list.append(psi)

        result = estimate_joint_latency(
            carrier_freqs=np.array(carriers),
            residue_phases=np.array(psi_list),
            weights=np.ones(3),
            tau_min=0.0,
            tau_max=0.2,
            tau_step=1e-3,
        )

        assert abs(result.fit - 1.0) < 1e-6, f"Expected fit=1.0; got {result.fit:.8f}"
        assert result.reliable

    def test_fit_zero_for_all_zero_weights(self):
        """All-zero weights → fit==0.0, reliable==False, no ZeroDivisionError.

        Spec §4: 'If Σ_c κ_c == 0 (degenerate), set fit = 0.0.'
        """
        psi, _ = residue_phase(
            *_clean_phasors(0.05, 10.0, 5.0, n_trials=1), 5.0, 10.0,
        )
        result = estimate_joint_latency(
            carrier_freqs=np.array([10.0]),
            residue_phases=np.array([psi]),
            weights=np.array([0.0]),
            tau_min=0.0,
            tau_max=0.1,
            tau_step=1e-3,
        )

        assert result.fit == 0.0, f"Expected fit=0.0 for zero weights; got {result.fit}"
        assert not result.reliable

    def test_irreconcilable_carriers_fail_fit_gate(self):
        """4 carriers with mutually irreconcilable residue phases fail the fit gate.

        For carriers [10, 15, 20, 25] Hz with equal unit weights, some ψ
        assignments make max_τ A(τ)/Σw (= fit) fall below the default
        fit_min=0.5 — no single τ reconciles all four carriers.  The ψ values
        below were chosen to minimise max_τ A(τ)/Σw over [0, 0.2] s and yield
        fit ≈ 0.45.

        dominance_min=0.0 disables the dominance gate so that reliable=False
        is caused exclusively by the fit gate; this is the 'bad channel'
        scenario described in design §4.
        """
        # Residue phases for [10, 15, 20, 25] Hz that drive fit below 0.5:
        # found by exhaustive grid search over ψ ∈ [−π, π)^4.
        psi_irreconcilable = np.array([2.5633, 2.3618, -2.7418, 1.2077])

        result = estimate_joint_latency(
            carrier_freqs=np.array([10., 15., 20., 25.]),
            residue_phases=psi_irreconcilable,
            weights=np.ones(4),
            tau_min=0.0,
            tau_max=0.2,
            tau_step=5e-4,
            dominance_min=0.0,   # disable dominance gate — isolate fit gate
            fit_min=0.5,
        )

        assert result.fit < 1.0, (
            f"Expected fit<1 for irreconcilable carriers; got {result.fit:.4f}"
        )
        assert result.fit < 0.5, (
            f"Expected fit<0.5 (default fit_min=0.5); got {result.fit:.4f}"
        )
        assert not result.reliable, (
            "Expected reliable=False when fit<fit_min (fit gate, dominance gate disabled)"
        )

    def test_fit_decreases_with_von_mises_noise(self):
        """Mean fit at κ=4 (tight concentration) exceeds mean fit at κ=1 (loose).

        Spec §4 notes 'von Mises κ≈4 gives fit≈0.86'.  Single-trial phasors
        (n_trials=1) amplify noise effects so the monotonicity is detectable
        across a few Monte-Carlo repetitions; mean fit at κ=4 should exceed
        0.8 and exceed mean fit at κ=1.
        """
        tau_true = 0.08
        f_t = 5.0
        carriers = [10.0, 15.0, 20.0]
        n_reps = 50
        rng = np.random.default_rng(17)

        fits_high, fits_low = [], []
        for _ in range(n_reps):
            for kappa, collector in [(4.0, fits_high), (1.0, fits_low)]:
                psi_list = []
                for f_c in carriers:
                    t_ph, c_ph = _clean_phasors(tau_true, f_c, f_t, n_trials=1)
                    noise = rng.vonmises(0.0, kappa, 1)
                    c_noisy = c_ph * np.exp(1j * noise)
                    psi, _ = residue_phase(t_ph, c_noisy, f_t, f_c)
                    psi_list.append(psi)
                # Unit weights: single trial → R_c=1 for all carriers anyway
                r = estimate_joint_latency(
                    carrier_freqs=np.array(carriers),
                    residue_phases=np.array(psi_list),
                    weights=np.ones(3),
                    tau_min=0.0,
                    tau_max=0.2,
                    tau_step=1e-3,
                )
                collector.append(r.fit)

        mean_fit_high = float(np.mean(fits_high))
        mean_fit_low = float(np.mean(fits_low))
        assert mean_fit_high > 0.8, (
            f"Expected mean fit>0.8 at κ=4; got {mean_fit_high:.4f}"
        )
        assert mean_fit_high > mean_fit_low, (
            f"Expected fit to decrease with noise; "
            f"κ=4 mean_fit={mean_fit_high:.3f}, κ=1 mean_fit={mean_fit_low:.3f}"
        )


# ---------------------------------------------------------------------------
# TestCarrierInvariance — τ̂ must be a property of the signal, not the carriers
# ---------------------------------------------------------------------------

class TestCarrierInvariance:
    """Carrier-invariance: different carrier subsets recover the same τ.

    Within each combination's unambiguous window W = 1/gcd(freqs_Hz), the
    joint estimator must yield the same τ̂ regardless of which carriers are
    used — only the spread (σ) should differ.

    Fairness rule: within each Monte-Carlo rep, phasors are drawn for ALL four
    carriers [10, 15, 20, 25] Hz ONCE; every combination uses its own subset
    of those SAME phasors.  Differences arise from carrier choice only.

    τ_true = 60 ms is chosen to sit inside the tightest window:
      {10,20} → W = 1/gcd(10,20) = 1/10 = 100 ms  (60 ms < 100 ms ✓)
      all others → W = 200 ms.
    """

    _F_T = 5.0
    _ALL_CARRIERS = [10.0, 15.0, 20.0, 25.0]
    _COMBOS = {
        "{10,15}": [10.0, 15.0],
        "{10,20}": [10.0, 20.0],
        "{15,20}": [15.0, 20.0],
        "{10,15,20}": [10.0, 15.0, 20.0],
    }
    _TAU_TRUE_S = 0.060   # 60 ms — within ALL combinations' windows
    _N_TRIALS = 30
    _KAPPA = 16.0          # tight noise: σ_angle ≈ 0.25 rad → σ_τ < 1 ms per carrier
    # 3 ms tolerance: 3× grid step, >> theoretical per-carrier σ_τ < 0.8 ms at κ=16/N=30
    _TOL_S = 0.003
    # Restrict the search grid to the TIGHTEST window across all tested combinations
    # ({10,20} → W = 100 ms).  Testing "within the window" requires the grid to stay
    # inside the window; otherwise {10,20} correctly returns an alias at 160 ms which
    # is equally valid but not the canonical representative we compare against.
    _TAU_MAX_S = 0.100

    @classmethod
    def _draw_phasors(cls, rng):
        """Draw (target, carrier) phasors for ALL_CARRIERS ONCE — fairness rule.

        Convention: target = reference clock (phase encodes onset jitter only);
        latency τ lives on the carrier (matches design doc §2 and make_phasors).
        """
        delta = rng.uniform(-0.01, 0.01, cls._N_TRIALS)   # 10 ms onset jitter
        by_carrier: dict = {}
        for f_c in cls._ALL_CARRIERS:
            noise_t = rng.vonmises(0.0, cls._KAPPA, cls._N_TRIALS)
            noise_c = rng.vonmises(0.0, cls._KAPPA, cls._N_TRIALS)
            c_t = np.exp(1j * (2 * np.pi * cls._F_T * delta + noise_t))
            c_c = np.exp(
                1j * (-2 * np.pi * f_c * cls._TAU_TRUE_S
                       + 2 * np.pi * f_c * delta + noise_c)
            )
            by_carrier[f_c] = (c_t, c_c)
        return by_carrier

    @classmethod
    def _estimate(cls, by_carrier, subset_freqs, **kw):
        """Run estimate_joint_latency on a subset of the pre-built phasor dict."""
        freqs, psis, wts = [], [], []
        for f_c in subset_freqs:
            ct, cc = by_carrier[f_c]
            psi, r = residue_phase(ct, cc, cls._F_T, f_c)
            freqs.append(f_c)
            psis.append(psi)
            wts.append(r)
        kw.setdefault("tau_max", cls._TAU_MAX_S)
        return estimate_joint_latency(
            np.array(freqs), np.array(psis), np.array(wts), **kw
        )

    @pytest.mark.parametrize("seed", [42, 137, 999, 2024, 31415])
    def test_each_combo_agrees_with_truth(self, seed):
        """Each carrier combination recovers τ_true = 60 ms within 3 ms.

        The grid is capped at tau_max = 100 ms (the tightest window, for {10,20})
        so the estimator searches within the window where each combination is
        unambiguous.  Tolerance (3 ms) is 3× the 1 ms grid step and far above
        the theoretical noise floor (< 0.8 ms at κ=16, N=30).
        """
        rng = np.random.default_rng(seed)
        by_all = self._draw_phasors(rng)
        for name, combo in self._COMBOS.items():
            res = self._estimate(by_all, combo)
            err = abs(res.tau_s - self._TAU_TRUE_S)
            assert err <= self._TOL_S, (
                f"seed={seed}, combo={name}: "
                f"|τ̂ − τ_true| = {err * 1e3:.2f} ms > {self._TOL_S * 1e3:.0f} ms"
            )

    @pytest.mark.parametrize("seed", [42, 137, 999, 2024, 31415])
    def test_cross_combo_spread(self, seed):
        """Max pairwise spread among all combos ≤ 6 ms (= 2 × tol).

        If all combos agree with truth within 3 ms, their pairwise spread is
        bounded by 6 ms.  This test makes that cross-combination agreement
        explicit and independently reportable.
        """
        rng = np.random.default_rng(seed)
        by_all = self._draw_phasors(rng)
        estimates = {
            name: self._estimate(by_all, combo).tau_s
            for name, combo in self._COMBOS.items()
        }
        vals = list(estimates.values())
        spread = float(max(vals) - min(vals))
        assert spread <= 2 * self._TOL_S, (
            f"seed={seed}: cross-combo spread = {spread * 1e3:.2f} ms "
            f"> {2 * self._TOL_S * 1e3:.0f} ms\n"
            f"  estimates (ms): {[(k, round(v * 1e3, 2)) for k, v in estimates.items()]}"
        )


# ---------------------------------------------------------------------------
# TestRandomPhaseNull — confidence metrics under mutually-random residue phases
# ---------------------------------------------------------------------------

class TestRandomPhaseNull:
    """Null floor for the confidence metrics at the production case C=3.

    When the three carriers {10,15,20} Hz share NO latency, their residue phases
    ψ_c are mutually random (iid Uniform(-π, π]). This class characterises the
    resulting `fit` and `dominance` null and contrasts it with a COHERENT
    reference (all carriers encode one shared τ), which brackets the top of the
    scale.

    HONEST CAVEAT (design §4 KNOWN LIMITATION, confirmed by measurement below):
    with only 3 carriers the agreement landscape is soft, so the random-phase
    null is NOT low — QA measured (N=5000, default [0,0.2] s window):
        fit  : mean≈0.81, 95th pct≈0.98,  P(fit>0.95)≈0.13
        dom  : mean≈1.70, median≈1.51, 95th pct≈2.97
    i.e. a random draw very often looks "confident". These tests therefore
    assert what is TRUE of the null (the aggregate averages are bounded away
    from the coherent ceiling) rather than the naive expectation that random
    phases yield LOW confidence. The thresholds below are set from the measured
    null with comfortable margins so the test is deterministic and robust across
    seeds.
    """

    _FREQS = np.array([10.0, 15.0, 20.0])
    _N = 3000

    @staticmethod
    def _wrap(a):
        return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi

    def _draw_null(self, n=None):
        """Fit & dominance for n draws of mutually-random residue phases (R_c=1).

        A fresh seeded rng, one independent ψ triplet per draw (NOT reused).
        """
        n = n or self._N
        rng = np.random.default_rng(20260702)
        fits = np.empty(n)
        doms = np.empty(n)
        for i in range(n):
            psi = rng.uniform(-np.pi, np.pi, 3)     # independent every draw
            res = estimate_joint_latency(self._FREQS, psi, np.ones(3))
            fits[i] = res.fit
            doms[i] = res.dominance
        return fits, doms

    def test_null_means_bounded_away_from_ceiling(self):
        """Random-phase mean fit & dominance stay below the coherent ceiling.

        Measured: mean fit≈0.81, mean dominance≈1.70. The coherent ceiling is
        fit=1.0, dominance=C=3.0. Thresholds (fit<0.85, dom<2.0) sit above the
        measured means with margin but well below the ceiling — they encode
        'random phases do not, on AVERAGE, reach the confidence of a genuine
        shared latency', which is the honest, defensible claim at C=3.
        """
        fits, doms = self._draw_null()
        mean_fit = float(np.mean(fits))
        finite_dom = doms[np.isfinite(doms)]
        mean_dom = float(np.mean(finite_dom))

        assert mean_fit < 0.85, f"mean random fit={mean_fit:.3f} (expected ~0.81)"
        assert mean_dom < 2.0, f"mean random dominance={mean_dom:.3f} (expected ~1.70)"

    def test_random_fit_is_a_poor_discriminator_at_C3(self):
        """P(random fit > 0.95) is NOT small at C=3 — it is ~0.13.

        This deliberately DOCUMENTS the design §4 limitation: unlike the naive
        expectation ('P(random fit>0.95) < 5%'), at three carriers a random draw
        exceeds fit=0.95 about 13% of the time, so `fit` alone cannot gate a
        shared latency. Asserting the true band (0.05, 0.25) is a regression
        guard on this known property.
        """
        fits, _ = self._draw_null()
        p_high = float(np.mean(fits > 0.95))
        assert 0.05 < p_high < 0.25, (
            f"P(random fit>0.95)={p_high:.3f}; expected ~0.13 (fit is a weak "
            f"discriminator at C=3 — design §4)"
        )

    def test_coherent_reference_tops_the_scale(self):
        """A genuine shared latency gives fit=1.0 and dominance=C=3 (noiseless).

        ψ_c = wrap(2π f_c τ) for a single shared τ makes every cosine peak at
        τ, so A(τ*)=Σκ_c ⇒ fit=1.0, and with equal unit weights the runner-up
        local max sums to 1 ⇒ dominance=C=3. This brackets the TOP of the scale.
        """
        tau = 0.09
        psi = self._wrap(2 * np.pi * self._FREQS * tau)
        res = estimate_joint_latency(self._FREQS, psi, np.ones(3))
        assert abs(res.tau_s - tau) <= 1e-3
        assert res.fit >= 0.98, f"coherent fit={res.fit:.4f}"
        assert abs(res.fit - 1.0) < 1e-6
        assert res.dominance >= 2.9, f"coherent dominance={res.dominance:.3f}"

    def test_coherent_fit_exceeds_random_95th_percentile(self):
        """The coherent (noiseless) fit clears the random-phase 95th percentile.

        Measured null 95th pct fit ≈ 0.98; coherent fit = 1.0. The margin is
        THIN (≈0.02) — an honest reflection of fit's weak separating power at
        C=3 — but the ordering holds and is asserted here.
        """
        fits, _ = self._draw_null()
        p95 = float(np.percentile(fits, 95))
        coherent_fit = 1.0  # noiseless shared latency (see previous test)
        assert coherent_fit > p95, (
            f"coherent fit {coherent_fit} did not exceed null 95th pct {p95:.4f}"
        )
        assert p95 > 0.9, (
            f"null 95th pct fit={p95:.3f}; expected high (~0.98) at C=3 — the "
            f"very reason fit cannot certify a shared latency here"
        )

    def test_deterministic_across_calls(self):
        """Seeded null is reproducible (same rng seed ⇒ identical draws)."""
        f1, d1 = self._draw_null(n=200)
        f2, d2 = self._draw_null(n=200)
        np.testing.assert_array_equal(f1, f2)
        np.testing.assert_array_equal(d1, d2)


# ---------------------------------------------------------------------------
# TestWithinTrialStationarity — split-half detector of within-trial drift
# ---------------------------------------------------------------------------

from scipy.stats import wilcoxon  # noqa: E402
from compare_methods import (  # noqa: E402
    within_trial_split_diffs,
    stationarity_verdict,
    _deoverlap_stride,
)


class TestWithinTrialStationarity:
    """Synthetic ground-truth validation of the temporal-vs-random split test.

    The detector compares, per trial, the residue-phase half-difference of a
    TEMPORAL split (first vs second half of the time-ordered windows) against a
    RANDOM split (shuffle then halve). A STATIONARY phase gives temporal ≈ random
    (ratio ~ 1); a within-trial DRIFT makes the temporal split systematically
    larger (ratio ≫ 1). Deterministic (seeded), fast.
    """

    _F_T = 5.0
    _F_C = 20.0
    _T = 12          # trials
    _W = 28          # windows/trial (matches non-overlapping real recordings)
    _KAPPA = 8.0     # von Mises phase-noise concentration
    _TAU_S = 0.05

    def _make_phasors(self, drift_max_rad, seed):
        """(target_tw, carrier_tw) with a stationary target and a carrier whose
        phase drifts linearly by drift_max_rad from the first to the last window."""
        rng = np.random.default_rng(seed)
        target = np.exp(1j * rng.vonmises(0.0, self._KAPPA, (self._T, self._W)))
        drift = np.linspace(0.0, drift_max_rad, self._W)[None, :]
        carrier = np.exp(1j * (
            -2 * np.pi * self._F_C * self._TAU_S
            + drift
            + rng.vonmises(0.0, self._KAPPA, (self._T, self._W))
        ))
        return target, carrier

    def _run(self, drift_max_rad, split_seed=0):
        target, carrier = self._make_phasors(drift_max_rad, seed=1234)
        rng = np.random.default_rng(split_seed)
        return within_trial_split_diffs(
            target, carrier, self._F_T, self._F_C, k=1, rng=rng)

    def test_stationary_temporal_matches_random(self):
        """No drift → temporal and random split |Δ| are comparable (ratio ~ 1).

        The verdict must be 'stationary' for both the residue and the raw carrier
        phase, and the Wilcoxon paired test must NOT flag a difference.
        """
        diffs = self._run(drift_max_rad=0.0)
        for q in ("residue", "carrier"):
            med_t, med_r, ratio, p, verdict = stationarity_verdict(
                diffs[q][0], diffs[q][1])
            assert 0.5 < ratio < 2.0, f"{q}: stationary ratio {ratio:.2f} not ~1"
            assert p > 0.05, f"{q}: stationary p={p:.3g} unexpectedly significant"
            assert verdict == "stationary", f"{q}: got {verdict}"

    def test_linear_drift_is_detected(self):
        """A linear within-trial phase drift → temporal split ≫ random split.

        The temporal/random ratio must be large and the paired test significant,
        for both residue and raw carrier phase, yielding a NON-STATIONARY verdict.
        """
        diffs = self._run(drift_max_rad=3.0)
        for q in ("residue", "carrier"):
            med_t, med_r, ratio, p, verdict = stationarity_verdict(
                diffs[q][0], diffs[q][1])
            assert ratio > 2.0, f"{q}: drift ratio {ratio:.2f} not ≫ 1"
            assert p < 0.05, f"{q}: drift p={p:.3g} not significant"
            assert verdict == "NON-STATIONARY", f"{q}: got {verdict}"

    def test_drift_temporal_exceeds_stationary_temporal(self):
        """Drift inflates the TEMPORAL split specifically (not the random null).

        The temporal-split median under drift far exceeds the stationary case,
        while the random-split medians stay comparable (drift averaged out).
        """
        stat = self._run(drift_max_rad=0.0)["residue"]
        drift = self._run(drift_max_rad=3.0)["residue"]
        temporal_stat = float(np.median(np.abs(stat[0])))
        temporal_drift = float(np.median(np.abs(drift[0])))
        random_stat = float(np.median(np.abs(stat[1])))
        random_drift = float(np.median(np.abs(drift[1])))

        assert temporal_drift > 2.0 * temporal_stat, (
            f"drift temporal={temporal_drift:.4g} not ≫ "
            f"stationary temporal={temporal_stat:.4g}")
        # Random split is the drift-insensitive null: stays within ~3x.
        assert random_drift < 3.0 * random_stat, (
            f"random split moved too much under drift: "
            f"{random_drift:.4g} vs {random_stat:.4g}")

    def test_deoverlap_stride_non_overlapping(self):
        """Non-overlapping cache (hop == window_size) → k = 1 (no subsampling)."""
        # 28 windows of 600 samples over 17251 time points (a real recording).
        k, hop = _deoverlap_stride(n_time=17251, window_size=600, n_windows=28)
        assert k == 1, f"expected k=1 for non-overlapping windows, got {k}"
        assert hop >= 600.0

    def test_deoverlap_stride_overlapping(self):
        """Overlapping cache (hop < window_size) → k > 1 so kept windows are
        independent."""
        # 600-sample windows at hop ≈ 150 (75% overlap) → ~112 windows.
        n_windows = (17251 - 600) // 150 + 1
        k, hop = _deoverlap_stride(n_time=17251, window_size=600,
                                   n_windows=n_windows)
        assert k >= 3, f"expected k>=3 for 75%-overlap windows, got {k} (hop={hop:.0f})"

    def test_deterministic(self):
        """Same split seed ⇒ identical diffs (reproducible)."""
        a = self._run(drift_max_rad=1.0, split_seed=7)["residue"]
        b = self._run(drift_max_rad=1.0, split_seed=7)["residue"]
        np.testing.assert_array_equal(a[0], b[0])
        np.testing.assert_array_equal(a[1], b[1])


# ---------------------------------------------------------------------------
# TestSnrVsWindowSize — does a smaller rectangular window improve SNR?
# ---------------------------------------------------------------------------

from compare_methods import (  # noqa: E402
    gen_stationary,
    gen_chirp,
    snr_curve_generated,
    rect_onbin_power_norm,
    rect_snr_at,
)


class TestSnrVsWindowSize:
    """Ground-truth validation of the SNR-vs-window-size claim (rectangular, no
    taper), self-contained (own rectangular FFT; independent of analysis.py).

    Claim under test: lowering the window size gives BETTER SNR ("kills more noise
    without harming the signal"). Verdict encoded by the tests:
      * STATIONARY signal → SNR ~FLAT across ws (claim FALSE; SNR is set by total
        integration time, not window size).
      * On-bin signal power ∝ ws² exactly (the mechanism: shrinking ws shrinks the
        coherent signal estimate in proportion — noise is not killed "for free").
      * A DRIFTING (chirp) signal → SNR DECREASES with ws (smaller ws better) —
        the only regime where the claim holds, and only because the signal is
        non-stationary.
    Deterministic (seeded), fast.
    """

    _FS = 300.0
    _DUR = 60.0
    _F_T = 5.0
    _F_C = 10.0
    # Valid ws for a 5 Hz target: bin index 5·ws ≥ 5 ⟹ ws ≥ 1.0 s (below that the
    # neighbor-noise band hits DC and SNR is undefined).
    _WS = [1.0, 2.0, 4.0]

    def test_stationary_snr_is_flat(self):
        """Stationary tone + white noise → SNR flat across ws (max/min < 1.6)."""
        rng = np.random.default_rng(0)
        sig = gen_stationary(self._FS, self._DUR, self._F_T, self._F_C,
                             rng, noise_sd=6.0)
        snr = snr_curve_generated(sig, self._FS, self._F_T, sweep=self._WS)
        assert np.all(np.isfinite(snr)), f"undefined SNR in valid range: {snr}"
        ratio = float(np.max(snr) / np.min(snr))
        assert ratio < 1.6, f"stationary SNR not flat across ws: {snr} (ratio {ratio:.2f})"

    def test_signal_power_scales_with_ws_squared(self):
        """Noiseless on-bin coherent power ∝ ws² (power/ws² constant to <0.1%)."""
        rng = np.random.default_rng(1)
        sig = gen_stationary(self._FS, self._DUR, self._F_T, self._F_C,
                             rng, noise_sd=0.0)
        norm = np.array([rect_onbin_power_norm(sig, self._FS, w, self._F_T)
                         for w in self._WS])
        rel_spread = float(np.std(norm) / np.mean(norm))
        assert rel_spread < 1e-3, (
            f"on-bin power not ∝ ws² (power/ws² spread {rel_spread:.2e}): {norm}")

    def test_drift_snr_decreases_with_ws(self):
        """Chirp (within-trial frequency drift) → SNR decreases as ws grows.

        Smaller ws is better ONLY here (non-stationary signal). Monotone
        decreasing, and the smallest window clears the largest by a wide margin.
        """
        rng = np.random.default_rng(2)
        sig = gen_chirp(self._FS, self._DUR, self._F_T, sweep_hz=1.5,
                        f_carrier=self._F_C, rng=rng, noise_sd=1.0)
        snr = snr_curve_generated(sig, self._FS, self._F_T, sweep=self._WS)
        assert snr[0] > snr[1] > snr[2], f"chirp SNR not monotone-decreasing: {snr}"
        assert snr[0] > 3.0 * snr[2], (
            f"chirp: smallest ws SNR {snr[0]:.2f} not ≫ largest {snr[2]:.2f}")

    def test_target_undefined_for_small_window(self):
        """5 Hz target: ws below 1 s puts the bin in the DC guard → SNR is NaN."""
        rng = np.random.default_rng(3)
        sig = gen_stationary(self._FS, self._DUR, self._F_T, self._F_C,
                             rng, noise_sd=6.0)
        s_small, _ = rect_snr_at(sig, self._FS, 0.4, self._F_T)  # bin 2
        s_ok, _ = rect_snr_at(sig, self._FS, 2.0, self._F_T)     # bin 10
        assert not np.isfinite(s_small), "expected NaN SNR for ws=0.4 s at 5 Hz"
        assert np.isfinite(s_ok) and s_ok > 0

    def test_deterministic(self):
        """Seeded generation ⇒ identical SNR curve."""
        a = snr_curve_generated(
            gen_stationary(self._FS, self._DUR, self._F_T, self._F_C,
                           np.random.default_rng(9), noise_sd=6.0),
            self._FS, self._F_T, sweep=self._WS)
        b = snr_curve_generated(
            gen_stationary(self._FS, self._DUR, self._F_T, self._F_C,
                           np.random.default_rng(9), noise_sd=6.0),
            self._FS, self._F_T, sweep=self._WS)
        np.testing.assert_array_equal(a, b)
