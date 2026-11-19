import unittest
from unittest.mock import patch

import mne
import numpy as np

from analysis import Spectral
from core import ConditionBlob, ConditionView, Study, Subject
from core_types import ConditionProperties


def _info(ch_names=("O1", "O2", "P3"), sfreq=100.0):
    return mne.create_info(ch_names=list(ch_names), sfreq=sfreq, ch_types="eeg")


class TestConditionProperties(unittest.TestCase):
    def test_immutability(self):
        props = ConditionProperties(np.float64(10.0), np.float64(20.0))
        with self.assertRaises(AttributeError):
            props.target_frequency = 15.0

    def test_missing_fields(self):
        with self.assertRaises(TypeError):
            ConditionProperties(target_frequency=10.0)


class TestConditionBlob(unittest.TestCase):
    def setUp(self):
        # (S, T, C, Time)
        self.raw = np.random.rand(2, 3, 3, 200).astype(np.float64)
        self.props = ConditionProperties(np.float64(10.0), np.float64(20.0))
        self.info = _info()
        self.subjects = [Subject(0, "S1"), Subject(1, "S2")]

    def test_happy_path(self):
        blob = ConditionBlob(self.raw, 100.0, self.props, self.info, self.subjects)
        self.assertEqual(blob.n_subjects, 2)
        self.assertEqual(blob.n_trials, 3)
        self.assertEqual(blob.n_channels, 3)
        self.assertEqual(blob.n_time, 200)
        self.assertEqual(blob.raw_data.dtype, np.float64)

    def test_should_fail_non_4d(self):
        with self.assertRaises(ValueError):
            ConditionBlob(np.random.rand(2, 3, 3), 100.0, self.props, self.info, self.subjects)

    def test_should_fail_mismatched_subjects(self):
        with self.assertRaises(ValueError):
            ConditionBlob(self.raw, 100.0, self.props, self.info, self.subjects[:1])


class TestStudy(unittest.TestCase):
    def setUp(self):
        self.info = _info()
        self.props1 = ConditionProperties(np.float64(10.0), np.float64(20.0))
        self.props2 = ConditionProperties(np.float64(15.0), np.float64(30.0))

    def _stream_entry(self, name, props, n_trials=3, sample_rate=100.0):
        # Each entry: (name, key, info, sample_rate, raw_data) with raw_data shape (1, T, C, Time)
        data = np.random.rand(1, n_trials, 3, 200).astype(np.float64)
        return (name, props, self.info, sample_rate, data)

    def test_enforce_min_trials_success(self):
        # min_trials=2. S1: 3 trials, S2: 2 trials. Should succeed and truncate S1 to 2.
        stream = [
            self._stream_entry("S1", self.props1, n_trials=3),
            self._stream_entry("S2", self.props1, n_trials=2),
        ]
        study = Study(iter(stream), min_trials=2)
        blob = study._blobs[self.props1]
        self.assertEqual(blob.n_trials, 2)

    def test_subject_discovery_and_alignment(self):
        stream = [
            self._stream_entry("S2", self.props1),
            self._stream_entry("S1", self.props1),
        ]
        study = Study(iter(stream), min_trials=3)
        subjects = list(study.subjects().values())
        # Should be sorted alphabetically: S1, S2
        self.assertEqual(subjects[0].name, "S1")
        self.assertEqual(subjects[1].name, "S2")

    def test_get_condition_reshaping(self):
        stream = [
            self._stream_entry("S1", self.props1, n_trials=3),
            self._stream_entry("S2", self.props1, n_trials=3),
        ]
        study = Study(iter(stream), min_trials=3)
        view = study.aggregate(self.props1)
        # (S, T, C, Time) -> (1, S*T, C, Time) → (2, 3, 3, 200) -> (1, 6, 3, 200)
        self.assertEqual(view.raw_data().shape, (1, 6, 3, 200))

    def test_partial_participation(self):
        stream = [
            self._stream_entry("S1", self.props1),
            self._stream_entry("S2", self.props2),
        ]
        study = Study(iter(stream), min_trials=3)
        s1 = study.subjects()["S1"]
        s2 = study.subjects()["S2"]

        self.assertIn(self.props1, s1._views)
        self.assertNotIn(self.props2, s1._views)
        self.assertIn(self.props2, s2._views)
        self.assertNotIn(self.props1, s2._views)

    def test_empty_stream(self):
        study = Study(iter([]))
        self.assertEqual(len(study.subjects()), 0)
        self.assertEqual(len(study._blobs), 0)

    def test_non_existent_condition(self):
        study = Study(iter([]))
        with self.assertRaises(KeyError):
            study.aggregate(self.props1)

    def test_channels_are_aligned_by_name(self):
        # Same channels in a different order: re-indexed, not stacked by position.
        data1 = np.random.rand(1, 3, 3, 200)
        data2 = np.random.rand(1, 3, 3, 200)
        shuffled = _info(("P3", "O1", "O2"))
        study = Study(iter([("S1", self.props1, self.info, 100.0, data1),
                            ("S2", self.props1, shuffled, 100.0, data2)]))
        blob = study._blobs[self.props1]
        np.testing.assert_array_equal(blob.raw_data[1, :, 0], data2[0, :, 1])   # O1
        np.testing.assert_array_equal(blob.raw_data[1, :, 2], data2[0, :, 0])   # P3

    def test_different_channel_sets_are_rejected(self):
        other = _info(("O1", "O2", "Cz"))
        stream = [("S1", self.props1, self.info, 100.0, np.random.rand(1, 3, 3, 200)),
                  ("S2", self.props1, other, 100.0, np.random.rand(1, 3, 3, 200))]
        with self.assertRaises(ValueError):
            Study(iter(stream))

    def test_bad_channels_from_info(self):
        info_bad = _info()
        info_bad["bads"] = ["O2"]
        stream = [("S1", self.props1, self.info, 100.0, np.random.rand(1, 3, 3, 200)),
                  ("S2", self.props1, info_bad, 100.0, np.random.rand(1, 3, 3, 200))]
        study = Study(iter(stream))
        np.testing.assert_array_equal(
            study.conditions()[self.props1].bad_channels(),
            [[False, False, False], [False, True, False]],
        )
        s2 = study.subjects()["S2"][self.props1].take_channels(["O2", "P3"])
        np.testing.assert_array_equal(s2.bad_channels(), [[True, False]])
        self.assertEqual(study.conditions()[self.props1].mne_info()["bads"], [])
        np.testing.assert_array_equal(study.aggregate(self.props1).bad_channels(), [[False, True, False]])

    def test_truncation_is_reported(self):
        stream = [self._stream_entry("S1", self.props1, n_trials=4)]
        with patch("builtins.print") as mock_print:
            study = Study(iter(stream), min_trials=3)
        self.assertEqual(study._blobs[self.props1].n_trials, 3)
        self.assertIn("S1 (4)", " ".join(str(c.args[0]) for c in mock_print.call_args_list))


class TestSpectralShapes(unittest.TestCase):
    """The FFT lives in analysis.Spectral now, not on ConditionView."""

    def test_psd_only_averages_windows(self):
        # 2 subjects, 3 trials, 2 channels. Window 1.0s at sample_rate=100 → 100-sample window.
        # 200 time samples → 2 non-overlapping windows.
        raw = np.random.rand(2, 3, 2, 200).astype(np.float64)
        info = mne.create_info(ch_names=["O1", "O2"], sfreq=100.0, ch_types="eeg")
        subjects = [Subject(0, "S1"), Subject(1, "S2")]
        props = ConditionProperties(np.float64(10.0), np.float64(20.0))
        blob = ConditionBlob(raw, 100.0, props, info, subjects)

        view = ConditionView(blob, subject_idx=0)
        psd = Spectral(view, window_duration_s=1.0).power_per_trial()
        # (S, T, C, F): rfft of 100-sample window → 51 freq bins.
        self.assertEqual(psd.shape, (1, 3, 2, 51))

    def test_snr_topomap_shape(self):
        raw = np.random.rand(2, 3, 2, 200).astype(np.float64)
        info = mne.create_info(ch_names=["O1", "O2"], sfreq=100.0, ch_types="eeg")
        subjects = [Subject(0, "S1"), Subject(1, "S2")]
        props = ConditionProperties(np.float64(10.0), np.float64(20.0))
        blob = ConditionBlob(raw, 100.0, props, info, subjects)
        view = ConditionView(blob)

        snr = Spectral(view, window_duration_s=1.0).snr_topomap()
        # (C, F): averaged over S and T → (2, 51).
        self.assertEqual(snr.shape, (2, 51))


if __name__ == "__main__":
    unittest.main()
