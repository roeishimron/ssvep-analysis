import unittest
from unittest.mock import MagicMock
import numpy as np
from analyze_spectrum import CarrierComparisonAnalysis
from core_types import ConditionProperties

class TestCarrierComparisonAnalysis(unittest.TestCase):
    def setUp(self):
        self.mock_study = MagicMock()
        self.carriers = [10.0, 15.0]
        self.electrodes = ["T5"]
        # Target freq is hardcoded to 5.0 in the implementation for comparison
        self.props1 = ConditionProperties(target_frequency=np.float64(5.0), carrier_frequency=np.float64(10.0))
        self.props2 = ConditionProperties(target_frequency=np.float64(5.0), carrier_frequency=np.float64(15.0))
        
        self.mock_study.conditions.return_value = [self.props1, self.props2]
        
    def test_happy_path(self):
        # Subjects S1 and S2 participate in both
        mock_s1 = MagicMock()
        mock_s1.name = "S1"
        mock_s2 = MagicMock()
        mock_s2.name = "S2"
        
        self.mock_study.filter_subjects.return_value = [mock_s1, mock_s2]
        
        def s1_getitem(p):
            v = MagicMock()
            if p == self.props1: 
                v.snr_at_target.return_value = (2.0, 0.1)
            else: 
                v.snr_at_target.return_value = (3.0, 0.2)
            v.restrict_electrodes.return_value = v
            return v
        mock_s1.__getitem__.side_effect = s1_getitem
        
        def s2_getitem(p):
            v = MagicMock()
            if p == self.props1: 
                v.snr_at_target.return_value = (1.5, 0.05)
            else: 
                v.snr_at_target.return_value = (2.5, 0.15)
            v.restrict_electrodes.return_value = v
            return v
        mock_s2.__getitem__.side_effect = s2_getitem

        analysis = CarrierComparisonAnalysis(self.mock_study, self.carriers, self.electrodes)
        data = analysis._get_comparison_data()
        
        self.assertEqual(len(data), 2)
        self.assertEqual(data[0], ("S1", [2.0, 3.0], [0.1, 0.2]))
        self.assertEqual(data[1], ("S2", [1.5, 2.5], [0.05, 0.15]))

    def test_no_common_subjects(self):
        self.mock_study.filter_subjects.return_value = []
        analysis = CarrierComparisonAnalysis(self.mock_study, self.carriers, self.electrodes)
        data = analysis._get_comparison_data()
        self.assertEqual(len(data), 0)

if __name__ == '__main__':
    unittest.main()
