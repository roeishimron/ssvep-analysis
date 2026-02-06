import unittest
import numpy as np
import mne
import sys
from core import Study, ConditionProperties, ConditionBlob, Subject, ConditionView

class TestConditionBlob(unittest.TestCase):
    def setUp(self):
        # (S, T, E, W, F)
        self.data = np.random.rand(2, 3, 3, 5, 50).astype(np.complex64)
        self.props = ConditionProperties(np.float64(10.0), np.float64(20.0))
        self.info = mne.create_info(ch_names=["O1", "O2", "P3"], sfreq=300, ch_types="eeg")
        self.subjects = [Subject(0, "S1"), Subject(1, "S2")]

    def test_happy_path(self):
        blob = ConditionBlob(self.data, self.props, self.info, self.subjects)
        self.assertEqual(blob.n_subjects, 2)
        self.assertEqual(blob.n_trials, 3)
        self.assertEqual(blob.n_electrodes, 3)
        self.assertEqual(blob.n_windows, 5)
        self.assertEqual(blob.n_frequencies, 50)

class TestStudy(unittest.TestCase):
    def setUp(self):
        self.info = mne.create_info(ch_names=["O1", "O2", "P3"], sfreq=300, ch_types="eeg")
        self.props1 = ConditionProperties(np.float64(10.0), np.float64(20.0))
        self.props2 = ConditionProperties(np.float64(15.0), np.float64(30.0))
        
    def test_enforce_min_trials_success(self):
        # min_trials=2. S1: 3 trials, S2: 2 trials.
        # Should succeed and truncate S1 to 2.
        stream = [
            ("S1", self.props1, self.info, np.random.rand(3, 3, 5, 50).astype(np.complex64)),
            ("S2", self.props1, self.info, np.random.rand(2, 3, 5, 50).astype(np.complex64)),
        ]
        study = Study(iter(stream), min_trials=2)
        blob = study._blobs[self.props1]
        self.assertEqual(blob.n_trials, 2)

    def test_subject_discovery_and_alignment(self):
        stream = [
            ("S2", self.props1, self.info, np.random.rand(3, 3, 5, 50).astype(np.complex64)),
            ("S1", self.props1, self.info, np.random.rand(3, 3, 5, 50).astype(np.complex64)),
        ]
        study = Study(iter(stream), min_trials=3)
        subjects = list(study.subjects())
        # Should be sorted alphabetically: S1, S2
        self.assertEqual(subjects[0].name, "S1")
        self.assertEqual(subjects[1].name, "S2")

    def test_psd_only_averages_windows(self):
        # Setup: 2 Subjects, 3 Trials, 2 Electrodes, 5 Windows, 10 Frequencies
        # Shape: (2, 3, 2, 5, 10)
        data = np.random.rand(2, 3, 2, 5, 10).astype(np.complex64)
        info = mne.create_info(ch_names=["O1", "O2"], sfreq=300, ch_types="eeg")
        subjects = [Subject(0, "S1"), Subject(1, "S2")]
        blob = ConditionBlob(data, self.props1, info, subjects)
        
        view = ConditionView(blob, subject_idx=0)
        psd = view._get_psd()
        # Should be (S, T, E, F) -> (1, 3, 2, 10)
        # Averaged ONLY over axis 3 (windows).
        self.assertEqual(psd.shape, (1, 3, 2, 10))

    def test_get_condition_reshaping(self):
        stream = [
            ("S1", self.props1, self.info, np.random.rand(3, 3, 5, 50).astype(np.complex64)),
            ("S2", self.props1, self.info, np.random.rand(3, 3, 5, 50).astype(np.complex64)),
        ]
        study = Study(iter(stream), min_trials=3)
        view = study.get_condition(self.props1)
        # (S, T, E, W, F) -> (1, S*T, E, W, F)
        # (2, 3, 3, 5, 50) -> (1, 6, 3, 5, 50)
        self.assertEqual(view.data.shape, (1, 6, 3, 5, 50))

    def test_snr_averaging_logic(self):
        data = np.random.rand(2, 3, 2, 5, 10).astype(np.complex64)
        info = mne.create_info(ch_names=["O1", "O2"], sfreq=300, ch_types="eeg")
        subjects = [Subject(0, "S1"), Subject(1, "S2")]
        blob = ConditionBlob(data, self.props1, info, subjects)
        view = ConditionView(blob)
        
        snr = view.as_snr()
        # Should be (E, F) -> (2, 10)
        # Averaged over S and T.
        self.assertEqual(snr.shape, (2, 10))

if __name__ == '__main__':
    unittest.main()
