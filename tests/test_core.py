import unittest
import numpy as np
import mne
from core import ConditionProperties, ConditionBlob

class TestConditionProperties(unittest.TestCase):
    def test_happy_path(self):
        props = ConditionProperties(target_frequency=10.0, carrier_frequency=60.0)
        self.assertEqual(props.target_frequency, 10.0)
        self.assertEqual(props.carrier_frequency, 60.0)
        
        # Test equality
        props2 = ConditionProperties(target_frequency=10.0, carrier_frequency=60.0)
        self.assertEqual(props, props2)
        
        # Test hashability (use as dict key)
        d = {props: "test"}
        self.assertEqual(d[props2], "test")

    def test_immutability(self):
        props = ConditionProperties(target_frequency=10.0, carrier_frequency=60.0)
        with self.assertRaises(AttributeError):
            props.target_frequency = 12.0 # type: ignore

    def test_missing_fields(self):
        with self.assertRaises(TypeError):
            ConditionProperties(target_frequency=10.0) # type: ignore

class TestConditionBlob(unittest.TestCase):
    def setUp(self):
        self.props = ConditionProperties(target_frequency=10.0, carrier_frequency=60.0)
        self.info = mne.create_info(ch_names=['O1', 'O2'], sfreq=300.0, ch_types='eeg')
        self.subjects = [Subject(0, 'S1'), Subject(1, 'S2')]
        # (Subject, Trial, Electrode, Window, Frequency)
        self.data = np.zeros((2, 5, 2, 10, 100), dtype=np.complex128)

    def test_happy_path(self):
        blob = ConditionBlob(self.data, self.props, self.info, self.subjects)
        self.assertEqual(blob.data.shape, (2, 5, 2, 10, 100))
        self.assertEqual(blob.data.dtype, np.complex64)
        self.assertEqual(blob.n_subjects, 2)
        self.assertEqual(blob.subjects[0].name, 'S1')

    def test_mismatched_subjects(self):
        with self.assertRaises(ValueError):
            ConditionBlob(self.data, self.props, self.info, [self.subjects[0]])

    def test_wrong_dimensions(self):
        wrong_data = np.zeros((2, 5, 2, 10))
        with self.assertRaises(ValueError):
            ConditionBlob(wrong_data, self.props, self.info, self.subjects)

from core import Study, Subject, ConditionView

class TestStudy(unittest.TestCase):
    def setUp(self):
        self.props1 = ConditionProperties(10.0, 60.0)
        self.props2 = ConditionProperties(15.0, 60.0)
        self.info = mne.create_info(ch_names=['O1', 'O2', 'P3'], sfreq=300.0, ch_types='eeg')
        
        # Stream data: (name, props, info, data)
        # data is 4D: (Trial, Electrode, Window, Frequency)
        data1 = np.ones((3, 3, 5, 50), dtype=np.complex64)
        data2 = np.ones((2, 3, 5, 50), dtype=np.complex64) * 2
        
        self.stream = [
            ("S1", self.props1, self.info, data1),
            ("S2", self.props1, self.info, data1),
            ("S1", self.props2, self.info, data2),
        ]
        
        self.study = Study(iter(self.stream))

    def test_subjects_discovery(self):
        subjects = list(self.study.subjects())
        self.assertEqual(len(subjects), 2)
        names = [s.name for s in subjects]
        self.assertIn("S1", names)
        self.assertIn("S2", names)

    def test_filter_subjects(self):
        # S1 participated in both, S2 only in props1
        subjs1 = list(self.study.filter_subjects({self.props1}))
        self.assertEqual(len(subjs1), 2)
        
        subjs2 = list(self.study.filter_subjects({self.props1, self.props2}))
        self.assertEqual(len(subjs2), 1)
        self.assertEqual(subjs2[0].name, "S1")

    def test_get_condition_aggregate(self):
        view = self.study.get_condition(self.props1)
        # props1 has S1 and S2. 
        # Original blob data: (2 subjects, 3 trials, 3 electrodes, 5 windows, 50 frequencies)
        # Aggregate logic: (1 subject, 2 trials, 3 electrodes, 5 windows, 50 frequencies)
        self.assertEqual(view.data.shape, (1, 2, 3, 5, 50))

    def test_subject_access(self):
        s1 = next(s for s in self.study.subjects() if s.name == "S1")
        view = s1[self.props1]
        # S1 has (1, 3, 3, 5, 50) because we use slice indexing for consistency
        self.assertEqual(view.data.shape, (1, 3, 3, 5, 50))

    def test_restrict_electrodes(self):
        view = self.study.get_condition(self.props1)
        restricted = view.restrict_electrodes(['O1', 'O2'])
        self.assertEqual(restricted.data.shape, (1, 2, 2, 5, 50))
        
    def test_snr_calculation(self):
        view = self.study.get_condition(self.props1)
        snr = view.as_snr()
        # (Electrode, Frequency) -> (3, 50)
        self.assertEqual(snr.shape, (3, 50))

    def test_data_alignment_after_sorting(self):
        # Create a stream with subjects out of alphabetical order
        # data for each subject will be distinct constants to verify alignment
        data_b = np.ones((1, 3, 5, 50), dtype=np.complex64) * 2  # Subject B
        data_a = np.ones((1, 3, 5, 50), dtype=np.complex64) * 1  # Subject A
        
        props = ConditionProperties(20.0, 60.0)
        stream = [
            ("B", props, self.info, data_b),
            ("A", props, self.info, data_a),
        ]
        
        study = Study(iter(stream))
        
        # In the resulting blob, 'A' should be at index 0, 'B' at index 1
        blob = study._blobs[props]
        self.assertEqual(blob.subjects[0].name, "A")
        self.assertEqual(blob.subjects[1].name, "B")
        
        # Verify data values
        # Index 0 should have data from A (value 1)
        # Index 1 should have data from B (value 2)
        self.assertTrue(np.all(blob.data[0] == 1))
        self.assertTrue(np.all(blob.data[1] == 2))

    def test_trial_consolidation(self):
        # Two entries for same subject in same condition
        data1 = np.ones((1, 3, 5, 50), dtype=np.complex64)
        data2 = np.ones((2, 3, 5, 50), dtype=np.complex64)
        
        props = ConditionProperties(30.0, 60.0)
        stream = [
            ("S1", props, self.info, data1),
            ("S1", props, self.info, data2),
        ]
        
        study = Study(iter(stream))
        blob = study._blobs[props]
        
        # Should have 1 subject and 1+2=3 trials
        self.assertEqual(blob.n_subjects, 1)
        self.assertEqual(blob.n_trials, 3)

if __name__ == '__main__':
    unittest.main()
