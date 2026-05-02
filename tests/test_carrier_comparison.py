import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from analyze_spectrum import CarrierComparisonAnalysis
from core_types import ConditionProperties


class TestCarrierComparisonAnalysis(unittest.TestCase):
    def setUp(self):
        self.mock_study = MagicMock()
        self.carriers = [10.0, 15.0]
        self.electrodes = ["T5"]
        # Target hardcoded to 5Hz in the implementation.
        self.props1 = ConditionProperties(
            target_frequency=np.float64(5.0), carrier_frequency=np.float64(10.0),
        )
        self.props2 = ConditionProperties(
            target_frequency=np.float64(5.0), carrier_frequency=np.float64(15.0),
        )
        self.mock_study.conditions.return_value = {self.props1: None, self.props2: None}

    def test_happy_path(self):
        # Two subjects participate in both conditions.
        mock_s1 = MagicMock()
        mock_s1.name = "S1"
        mock_s2 = MagicMock()
        mock_s2.name = "S2"
        self.mock_study.filter_subjects.return_value = [mock_s1, mock_s2]

        # subject[props].take_channels(...) returns a sentinel view per (subject, props).
        def make_subject_getitem(by_props):
            def getter(p):
                v = MagicMock(name=f"view_for_{p.carrier_frequency}")
                v.take_channels.return_value = ("S1" if by_props is _S1 else "S2", p)
                return v
            return getter

        _S1 = object()
        _S2 = object()
        mock_s1.__getitem__.side_effect = make_subject_getitem(_S1)
        mock_s2.__getitem__.side_effect = make_subject_getitem(_S2)

        # Patch SSVEPAnalysis so snr_at_target returns the desired (mean, sem) per view.
        snr_table = {
            ("S1", self.props1): (2.0, 0.1),
            ("S1", self.props2): (3.0, 0.2),
            ("S2", self.props1): (1.5, 0.05),
            ("S2", self.props2): (2.5, 0.15),
        }

        def fake_ssvep(view):
            ana = MagicMock()
            ana.snr_at_target.return_value = snr_table[view]
            return ana

        with patch("analyze_spectrum.SSVEPAnalysis", side_effect=fake_ssvep):
            analysis = CarrierComparisonAnalysis(self.mock_study, self.carriers, self.electrodes)
            data = list(analysis._get_comparison_data())

        self.assertEqual(len(data), 2)
        self.assertEqual(data[0], ("S1", [(2.0, 0.1), (3.0, 0.2)]))
        self.assertEqual(data[1], ("S2", [(1.5, 0.05), (2.5, 0.15)]))

    def test_no_common_subjects(self):
        self.mock_study.filter_subjects.return_value = []
        analysis = CarrierComparisonAnalysis(self.mock_study, self.carriers, self.electrodes)
        data = list(analysis._get_comparison_data())
        self.assertEqual(len(data), 0)

    def test_snr_slopes(self):
        analysis = CarrierComparisonAnalysis(self.mock_study, self.carriers, self.electrodes)

        # Slope = (m1 - m0) / (carrier1 - carrier0) = ... / 5.0.
        test_data = [
            ("S1", [(2.0, 0.1), (3.0, 0.2)]),   # slope 0.2
            ("S2", [(1.5, 0.05), (2.5, 0.15)]), # slope 0.2
            ("S3", [(5.0, 0.1), (4.0, 0.1)]),   # slope -0.2
        ]
        slopes = list(analysis.slopes(iter(test_data)))

        self.assertEqual(len(slopes), 3)
        self.assertEqual(slopes[0][0], "S1")
        self.assertAlmostEqual(slopes[0][1], 0.2)
        self.assertEqual(slopes[1][0], "S2")
        self.assertAlmostEqual(slopes[1][1], 0.2)
        self.assertEqual(slopes[2][0], "S3")
        self.assertAlmostEqual(slopes[2][1], -0.2)


if __name__ == '__main__':
    unittest.main()
