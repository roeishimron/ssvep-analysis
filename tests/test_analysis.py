"""Tests for SSVEPAnalysis snr/power summary methods."""
from unittest.mock import MagicMock

import mne
import numpy as np
import pytest

from analysis import SSVEPAnalysis
from core_types import ConditionProperties
from interfaces import Recording, SSVEPRecording, SSVEPRecording as _SR


def _ssvep_analysis_with_fourier(props_target, props_carrier, fourier_5d, freqs_array):
    """Construct an SSVEPAnalysis with pre-computed FFT data, bypassing the FFT step."""
    rec = MagicMock(spec=SSVEPRecording)
    rec.props.return_value = ConditionProperties(
        target_frequency=np.float64(props_target),
        carrier_frequency=np.float64(props_carrier),
    )
    rec.sample_rate.return_value = 100.0
    rec.name.return_value = "mock"

    ana = SSVEPAnalysis.__new__(SSVEPAnalysis)
    ana.rec = rec
    ana.window_duration_s = 1.0
    ana.n_neighbors = 3
    ana.n_skip = 1
    ana._window_size = 100
    ana._fourier_cache = fourier_5d.astype(np.complex64)
    ana.expected_latency_s = 0.05
    ana.frequencies = lambda: freqs_array.astype(np.float64)
    return ana


def test_snr_at_target_with_two_subjects():
    # Target 5Hz at index 10 (sfreq=100, window=100 → 51 freq bins, 1Hz spacing).
    n_freqs = 101
    freqs = np.linspace(0, 50, n_freqs)
    target_idx = int(np.argmin(np.abs(freqs - 5.0)))
    carrier_idx = int(np.argmin(np.abs(freqs - 10.0)))

    # (S=2, T=3, C=4, W=1, F=101). Single window so coherent-window-average == raw.
    fourier = np.full((2, 3, 4, 1, n_freqs), 1.0 + 0j, dtype=np.complex64)
    # Subject 0: target power 10. Subject 1: target power 20. Noise power 1.
    fourier[0, :, :, :, target_idx] = np.sqrt(10.0)
    fourier[1, :, :, :, target_idx] = np.sqrt(20.0)
    # Carrier frequencies: subject 0 power 100, subject 1 power 200.
    fourier[0, :, :, :, carrier_idx] = np.sqrt(100.0)
    fourier[1, :, :, :, carrier_idx] = np.sqrt(200.0)

    ana = _ssvep_analysis_with_fourier(5.0, 10.0, fourier, freqs)

    snr_mean, snr_sem = ana.snr_at_target()
    assert snr_mean == pytest.approx(15.0)
    assert snr_sem == pytest.approx(5.0)

    p_mean, p_sem = ana.power_at_target()
    assert p_mean == pytest.approx(15.0)
    assert p_sem == pytest.approx(5.0)

    snr_c_mean, snr_c_sem = ana.snr_at_carrier()
    assert snr_c_mean == pytest.approx(150.0)
    assert snr_c_sem == pytest.approx(50.0)


def test_subject_level_sem_vs_trial_level():
    # Verify SEM is across subjects, not across trials.
    n_freqs = 101
    freqs = np.linspace(0, 50, n_freqs)
    target_idx = int(np.argmin(np.abs(freqs - 5.0)))

    # S=2, T=10, C=1, W=1, F=101.
    fourier = np.full((2, 10, 1, 1, n_freqs), 1.0 + 0j, dtype=np.complex64)
    fourier[0, :, :, :, target_idx] = np.sqrt(10.0)
    fourier[1, :, :, :, target_idx] = np.sqrt(20.0)

    ana = _ssvep_analysis_with_fourier(5.0, 10.0, fourier, freqs)
    mean, sem_val = ana.snr_at_target()

    assert mean == pytest.approx(15.0)
    assert sem_val == pytest.approx(5.0)
    # If SEM were across trials (20 samples) it would be much smaller.
    assert sem_val != pytest.approx(1.17669, rel=1e-3)
