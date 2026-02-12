import unittest
import numpy as np
from unittest.mock import MagicMock
from core import ConditionView, ConditionBlob, Subject
from core_types import ConditionProperties

class TestProcessingTime(unittest.TestCase):
    def setUp(self):
        self.n_subjects = 1
        self.n_trials = 1
        self.n_electrodes = 1
        self.n_windows = 1
        self.n_freqs = 30
        self.freqs_array = np.linspace(1, 30, self.n_freqs)
        
    def _create_mock_view(self, target_f, carrier_f, target_phase_rad, carrier_phase_rad):
        # target_f and carrier_f should be in the freqs_array
        t_idx = np.argmin(np.abs(self.freqs_array - target_f))
        c_idx = np.argmin(np.abs(self.freqs_array - carrier_f))
        
        data = np.zeros((self.n_subjects, self.n_trials, self.n_electrodes, self.n_windows, self.n_freqs), dtype=np.complex64)
        
        # Set phases
        data[0, 0, 0, 0, t_idx] = np.exp(1j * target_phase_rad)
        data[0, 0, 0, 0, c_idx] = np.exp(1j * carrier_phase_rad)
        
        mock_blob = MagicMock(spec=ConditionBlob)
        mock_blob.data = data
        mock_blob.subjects = [Subject(0, "S1")]
        mock_blob.props = ConditionProperties(target_frequency=np.float64(target_f), carrier_frequency=np.float64(carrier_f))
        mock_blob.n_electrodes = self.n_electrodes
        mock_blob.n_trials = self.n_trials
        mock_blob.n_subjects = self.n_subjects
        
        view = ConditionView(mock_blob)
        view.frequencies = MagicMock(return_value=self.freqs_array)
        # Mock _get_psd to avoid SNR calculation issues
        view._get_psd = MagicMock(return_value=np.abs(data.mean(axis=-2))**2)
        
        return view

    def test_processing_time_2to1_ratio(self):
        # Target: 5Hz (T=200ms), Carrier: 10Hz (T=100ms)
        # 50ms is the heuristic.
        
        # Case 1: dt1 is 60ms, dt2 is 160ms. Should pick 60ms.
        # t_target = 0 (phase 0)
        # t_carrier = 60ms (phase 0.6 * 2pi for 10Hz? No, 60ms/100ms * 2pi = 1.2pi)
        view = self._create_mock_view(5.0, 10.0, 0, 1.2 * np.pi)
        latencies = view.calculate_processing_time()
        # Expect 0.06
        self.assertAlmostEqual(latencies[0, 0], 0.06, places=5)
        
        # Case 2: dt1 is 140ms, dt2 is 40ms. Should pick 40ms.
        # t_target = 0
        # t_carrier = 40ms -> phase = 0.4 * 2pi = 0.8pi
        view = self._create_mock_view(5.0, 10.0, 0, 0.8 * np.pi)
        latencies = view.calculate_processing_time()
        self.assertAlmostEqual(latencies[0, 0], 0.04, places=5)

    def test_processing_time_3to1_ratio(self):
        # Target: 5Hz (T=200ms), Carrier: 15Hz (T=66.6ms)
        # n_cycles = 3.
        # Candidates: dt1, dt1 + 66.6, dt1 + 133.3
        
        # Suppose target_t = 0
        # carrier_t = 10ms. Candidates: 10, 76.6, 143.3
        # 50ms - 10 = 40.
        # 76.6 - 50 = 26.6.
        # 143.3 - 50 = 93.3.
        # Closest is 76.6ms.
        
        carrier_phase = (0.01 / (1/15.0)) * 2 * np.pi
        view = self._create_mock_view(5.0, 15.0, 0, carrier_phase)
        latencies = view.calculate_processing_time()
        self.assertAlmostEqual(latencies[0, 0], 0.01 + 1/15.0, places=5)

    def test_circularity_near_boundary(self):
        # Target 5Hz (T=200ms), Carrier 10Hz (T=100ms)
        # Suppose t_target = 190ms (phase = 1.9 * pi? No, 190/200 * 2pi = 1.9pi)
        # Suppose t_carrier = 40ms (phase = 0.4 * 2pi = 0.8pi)
        # dt1 = (40 - 190) % 200 = -150 % 200 = 50ms.
        # dt2 = (50 + 100) % 200 = 150ms.
        # Should pick 50ms exactly.
        
        view = self._create_mock_view(5.0, 10.0, 1.9 * np.pi, 0.8 * np.pi)
        latencies = view.calculate_processing_time()
        self.assertAlmostEqual(latencies[0, 0], 0.05, places=5)

if __name__ == '__main__':
    unittest.main()
