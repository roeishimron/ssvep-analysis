"""Tests for SSVEPAnalysis.processing_time_per_trial — phase ambiguity resolution."""
import unittest
from unittest.mock import MagicMock

import numpy as np

from analysis import SSVEPAnalysis
from core_types import ConditionProperties
from interfaces import SSVEPRecording


def _ssvep_analysis_with_fourier(
    target_f, carrier_f, target_phase_rad, carrier_phase_rad,
):
    """Build a SSVEPAnalysis whose phase_at(...) returns predetermined phasors.

    Bypasses the FFT pipeline to test only the candidate_distances + resolve_to_seconds
    composition that processing_time_per_trial wraps.
    """
    n_freqs = 30
    freqs_array = np.linspace(1, 30, n_freqs).astype(np.float64)
    t_idx = int(np.argmin(np.abs(freqs_array - target_f)))
    c_idx = int(np.argmin(np.abs(freqs_array - carrier_f)))

    # (S=1, T=1, C=1, W=1, F=30) — single phasor at each frequency bin.
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


class TestProcessingTime(unittest.TestCase):
    def test_processing_time_2to1_ratio(self):
        # Target 5Hz (T=200ms), carrier 10Hz (T=100ms). 50ms is the heuristic.

        # Case 1: t_target=0, t_carrier=60ms. diff=140ms. Candidates: 140, 40. Pick 40.
        ana = _ssvep_analysis_with_fourier(5.0, 10.0, 0, 1.2 * np.pi)
        latencies = ana.processing_time_per_trial()
        self.assertAlmostEqual(latencies[0, 0], 0.04, places=5)

        # Case 2: t_target=0, t_carrier=40ms. diff=160ms. Candidates: 160, 60. Pick 60.
        ana = _ssvep_analysis_with_fourier(5.0, 10.0, 0, 0.8 * np.pi)
        latencies = ana.processing_time_per_trial()
        self.assertAlmostEqual(latencies[0, 0], 0.06, places=5)

    def test_processing_time_3to1_ratio(self):
        # Target 5Hz, carrier 15Hz. n_cycles=3. carrier_t=10ms.
        # Candidates: -10, 56.6, 123.3. Pick 56.6 (closest to 50).
        carrier_phase = (0.01 / (1 / 15.0)) * 2 * np.pi
        ana = _ssvep_analysis_with_fourier(5.0, 15.0, 0, carrier_phase)
        latencies = ana.processing_time_per_trial()
        self.assertAlmostEqual(latencies[0, 0], 1 / 15.0 - 0.01, places=5)

    def test_circularity_near_boundary(self):
        # Target 5Hz, carrier 10Hz. t_target=190ms, t_carrier=40ms. diff=150ms.
        # Candidates: 150, 50. Pick 50.
        ana = _ssvep_analysis_with_fourier(5.0, 10.0, 1.9 * np.pi, 0.8 * np.pi)
        latencies = ana.processing_time_per_trial()
        self.assertAlmostEqual(latencies[0, 0], 0.05, places=5)


if __name__ == '__main__':
    unittest.main()
