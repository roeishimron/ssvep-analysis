import unittest
from unittest.mock import patch, MagicMock
import numpy as np
import os
from loader import StudyLoader
from core import ConditionProperties

class TestStudyLoader(unittest.TestCase):
    def setUp(self):
        self.loader = StudyLoader(recording_frequency=100.0, window_duration_s=1.0) # Simplified for testing

    def test_parse_folder_name(self):
        props, duration = self.loader._parse_folder_name("10hz_2ob_60s")
        self.assertEqual(props.carrier_frequency, 10.0)
        self.assertEqual(props.target_frequency, 5.0)
        self.assertEqual(duration, 60.0)
        
        props, duration = self.loader._parse_folder_name("10hz_2ob")
        self.assertEqual(duration, 60.0) # Default

    def test_process_data(self):
        # (Trial=2, Electrode=3, Time=300) -> 3 windows of 100 samples
        data = np.random.randn(2, 3, 300)
        processed = self.loader._process_data(data)
        
        # Window size is 100. 300/100 = 3 windows.
        # rfft of 100 samples is 100//2 + 1 = 51 frequencies.
        self.assertEqual(processed.shape, (2, 3, 3, 51))
        self.assertEqual(processed.dtype, np.complex64)

    @patch('mne.io.read_raw_edf')
    @patch('mne.find_events')
    @patch('mne.Epochs')
    @patch('loader.os.listdir')
    @patch('loader.os.path.exists')
    @patch('loader.os.path.isdir')
    def test_load_happy_path(self, mock_isdir, mock_exists, mock_listdir, mock_epochs, mock_find_events, mock_read_raw_edf):
        # Mock FS
        mock_exists.return_value = True
        mock_listdir.side_effect = [['10hz_2ob_60s'], ['subj1.edf']]
        mock_isdir.return_value = True
        
        # Mock MNE behaviors
        mock_raw = MagicMock()
        mock_raw.ch_names = ['Trigger', 'O1', 'O2']
        mock_raw.last_samp = 1000
        mock_read_raw_edf.return_value = mock_raw
        
        mock_find_events.return_value = np.array([[100, 0, 1], [500, 0, 1]])
        
        mock_epochs_inst = MagicMock()
        # (Trial=2, Electrode=2, Time=200) -> 2 windows of 100 samples
        mock_epochs_inst.get_data.return_value = np.random.randn(2, 2, 200)
        mock_epochs.return_value = mock_epochs_inst
        
        results = list(self.loader.load("/fake/path"))
        
        self.assertEqual(len(results), 1)
        name, props, info, data = results[0]
        self.assertEqual(name, "subj1")
        self.assertEqual(props.carrier_frequency, 10.0)
        # 2 trials, 2 electrodes, 2 windows, 51 frequencies
        self.assertEqual(data.shape, (2, 2, 2, 51))

    def test_load_non_existent_dir(self):
        with self.assertRaises(FileNotFoundError):
            list(self.loader.load("/non/existent/path"))

    @patch('loader.os.path.exists')
    @patch('loader.os.listdir')
    def test_load_no_files(self, mock_listdir, mock_exists):
        mock_exists.return_value = True
        mock_listdir.return_value = [] # Empty root
        results = list(self.loader.load("/fake/path"))
        self.assertEqual(len(results), 0)

if __name__ == '__main__':
    unittest.main()