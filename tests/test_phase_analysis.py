"""Tests for Spectral.phase_at — the best-electrode phasor extractor."""
import unittest
from unittest.mock import MagicMock

import numpy as np

from analysis import Spectral
from interfaces import Recording


def _spectral_with_fourier(fourier_5d, freqs_array, sample_rate=100.0):
    """Build a Spectral with pre-computed FFT data, bypassing the raw-data pipeline."""
    rec = MagicMock(spec=Recording)
    rec.sample_rate.return_value = sample_rate
    rec.name.return_value = "mock"
    spec = Spectral.__new__(Spectral)
    spec.rec = rec
    spec.window_duration_s = 1.0
    spec.n_neighbors = 3
    spec.n_skip = 1
    spec._window_size = int(sample_rate * 1.0)
    spec._fourier_cache = fourier_5d.astype(np.complex64)
    # Override frequencies to match the test's bin locations.
    spec.frequencies = lambda: freqs_array.astype(np.float64)
    return spec


class TestPhaseAnalysis(unittest.TestCase):
    def test_as_phase_selection_and_shape(self):
        n_subjects, n_trials, n_channels, n_windows, n_freqs = 1, 2, 2, 1, 15
        freqs_array = np.linspace(1, 15, n_freqs)
        target_freq = 10.0
        target_idx = int(np.argmin(np.abs(freqs_array - target_freq)))

        # (S, T, C, W, F)
        fourier = np.full(
            (n_subjects, n_trials, n_channels, n_windows, n_freqs),
            0.1 + 0j, dtype=np.complex64,
        )
        # Channel 0: weak; Channel 1: strong → phase_at picks channel 1.
        for trial in range(n_trials):
            fourier[0, trial, 0, 0, target_idx] = 1.0 + 0j
            fourier[0, trial, 1, 0, target_idx] = 10.0 + 0j

        spec = _spectral_with_fourier(fourier, freqs_array)
        phases, cycle_dur = spec.phase_at(target_freq)

        self.assertEqual(phases.shape, (n_subjects, n_trials))
        np.testing.assert_allclose(
            phases, np.full((n_subjects, n_trials), 10.0 + 0j), atol=1e-5,
        )
        self.assertAlmostEqual(cycle_dur, 0.1)

    def test_as_phase_multiple_subjects(self):
        n_subjects, n_trials, n_channels, n_windows, n_freqs = 2, 1, 2, 1, 15
        freqs_array = np.linspace(1, 15, n_freqs)
        target_freq = 10.0
        target_idx = int(np.argmin(np.abs(freqs_array - target_freq)))

        fourier = np.full(
            (n_subjects, n_trials, n_channels, n_windows, n_freqs),
            0.1 + 0j, dtype=np.complex64,
        )
        # S0: max at C0; S1: max at C1.
        fourier[0, 0, 0, 0, target_idx] = 10.0 + 0j
        fourier[0, 0, 1, 0, target_idx] = 1.0 + 0j
        fourier[1, 0, 0, 0, target_idx] = 1.0 + 0j
        fourier[1, 0, 1, 0, target_idx] = 20.0 + 0j

        spec = _spectral_with_fourier(fourier, freqs_array)
        phases, _ = spec.phase_at(target_freq)

        self.assertEqual(phases[0, 0], 10.0 + 0j)
        self.assertEqual(phases[1, 0], 20.0 + 0j)


if __name__ == '__main__':
    unittest.main()
