import unittest
import numpy as np
from unittest.mock import MagicMock
from core import ConditionView, ConditionBlob, Subject
from core_types import ConditionProperties

class TestPhaseAnalysis(unittest.TestCase):
    def test_as_phase_selection_and_shape(self):
        # Setup dimensions
        n_subjects = 1
        n_trials = 2
        n_electrodes = 2
        n_windows = 1
        n_freqs = 15 # Increased to satisfy into_SNR kernel size (needs >= 9)
        
        # Frequencies: 1..15 Hz
        freqs_array = np.linspace(1, 15, n_freqs)
        target_freq = 10.0
        target_idx = np.argmin(np.abs(freqs_array - target_freq))
        
        # Create data: (S, T, E, W, F)
        data = np.full((n_subjects, n_trials, n_electrodes, n_windows, n_freqs), 0.1 + 0j, dtype=np.complex64)
        
        # Electrode 0: Noise (low amplitude)
        # Electrode 1: Signal (high amplitude) at target freq
        
        # Trial 0
        data[0, 0, 0, 0, target_idx] = 1.0 + 0j # E0: Weak
        data[0, 0, 1, 0, target_idx] = 10.0 + 0j # E1: Strong
        
        # Trial 1
        data[0, 1, 0, 0, target_idx] = 1.0 + 0j # E0: Weak
        data[0, 1, 1, 0, target_idx] = 10.0 + 0j # E1: Strong
        
        # Mock Blob and View
        mock_blob = MagicMock(spec=ConditionBlob)
        mock_blob.data = data
        mock_blob.n_subjects = n_subjects
        mock_blob.n_trials = n_trials
        mock_blob.n_electrodes = n_electrodes
        mock_blob.n_windows = n_windows
        mock_blob.n_frequencies = n_freqs
        mock_blob.subjects = [Subject(0, "S1")]
        mock_blob.props = ConditionProperties(target_frequency=target_freq, carrier_frequency=0.0)
        
        # Create View
        view = ConditionView(mock_blob)
        view.frequencies = MagicMock(return_value=freqs_array)
        
        # Execute as_phase
        phases, cycle_dur = view.as_phase(target_freq)
        
        # Check shapes
        self.assertEqual(phases.shape, (n_subjects, n_trials))
        
        # Check values
        # Expected: Both trials should be 10.0
        expected_phase = 10.0 + 0j
        
        np.testing.assert_allclose(phases, np.full((n_subjects, n_trials), expected_phase), atol=1e-5)
        self.assertAlmostEqual(cycle_dur, 0.1)

    def test_as_phase_multiple_subjects(self):
        # Test 2 subjects to check indexing logic
        n_subjects = 2
        n_trials = 1
        n_electrodes = 2 # E0, E1
        n_windows = 1
        n_freqs = 15
        freqs_array = np.linspace(1, 15, n_freqs)
        target_freq = 10.0
        target_idx = np.argmin(np.abs(freqs_array - target_freq))
        
        data = np.full((n_subjects, n_trials, n_electrodes, n_windows, n_freqs), 0.1 + 0j, dtype=np.complex64)
        
        # S0: Max at E0
        data[0, 0, 0, 0, target_idx] = 10.0 + 0j
        data[0, 0, 1, 0, target_idx] = 1.0 + 0j
        
        # S1: Max at E1
        data[1, 0, 0, 0, target_idx] = 1.0 + 0j
        data[1, 0, 1, 0, target_idx] = 20.0 + 0j
        
        mock_blob = MagicMock(spec=ConditionBlob)
        mock_blob.data = data
        mock_blob.subjects = [Subject(0, "S0"), Subject(1, "S1")]
        mock_blob.n_electrodes = n_electrodes
        
        view = ConditionView(mock_blob)
        view.frequencies = MagicMock(return_value=freqs_array)
        
        phases, _ = view.as_phase(target_freq)
        
        # S0 should have 10.0 (from E0)
        # S1 should have 20.0 (from E1)
        self.assertEqual(phases[0, 0], 10.0 + 0j)
        self.assertEqual(phases[1, 0], 20.0 + 0j)

if __name__ == '__main__':
    unittest.main()
