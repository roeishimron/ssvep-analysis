import os
import re
import numpy as np
import mne
from typing import Iterator, Tuple, List
from scipy.signal.windows import kaiser
from core import ConditionProperties, SubjectData

class StudyLoader:
    def __init__(self, recording_frequency: float = 300.0, window_duration_s: float = 2.0):
        self.recording_frequency = recording_frequency
        self.window_size = int(recording_frequency * window_duration_s)

    def _parse_folder_name(self, folder_name: str) -> ConditionProperties:
        # Expected pattern: 10hz_2ob_60s
        match = re.match(r"(\d+)hz_(\d+)ob", folder_name)
        if not match:
            raise ValueError(f"Folder name {folder_name} does not match expected pattern <freq>hz_<ob>ob")
        
        carrier_freq = float(match.group(1))
        modulation = float(match.group(2))
        target_freq = carrier_freq / modulation
        
        return ConditionProperties(target_frequency=np.float64(target_freq), carrier_frequency=np.float64(carrier_freq))

    def _process_data(self, data: np.ndarray) -> SubjectData:
        # data shape: (Trial, Electrode, Time)
        # We want: (Trial, Electrode, Window, Frequency)
        
        # Sliding window view (non-overlapping)
        # data: (T, E, Time) -> segments: (T, E, WindowIdx, WindowSize)
        segments = np.lib.stride_tricks.sliding_window_view(data, self.window_size, axis=-1)
        segments = segments[..., ::self.window_size, :]
        
        # Demean
        segments = segments - np.mean(segments, axis=-1, keepdims=True)
        
        # Windowing and RFFT
        window = kaiser(self.window_size, 4)
        fourier_components = np.fft.rfft(segments * window, axis=-1)
        
        return fourier_components.astype(np.complex64)

    def _load_edf(self, file_path: str) -> Tuple[np.ndarray, mne.Info]:
        raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)
        
        raw.rename_channels(lambda s: s.replace(
        "EEG ", "").replace("-Pz", ""), False)

        raw.drop_channels(['Ax', 'Ay', 'Az'])
        raw.drop_channels(['Event', 'CM'])
        raw.drop_channels([c for c in raw.ch_names if ":" in c])
        raw.set_montage(montage='standard_1020')

        # detect events and edit
        events = mne.find_events(raw, stim_channel="Trigger", mask=8)
        raw.drop_channels(["Trigger"])

        # Set common average reference
        raw.set_eeg_reference()

        # Handle too close events:
        diffs = np.diff(events[:, 0], append=raw.last_samp)
        valids = np.argwhere(diffs > 1000).flatten()
        events = events[valids]
        
        # Construct epochs
        epochs = mne.Epochs(
            raw,
            picks='data',
            events=events,
            tmin=2.5, # Constant for 3 seconds delay minus 0.5 sec
            tmax=60, # TODO: Derive from file
            baseline=None,
        )

        return epochs.get_data(units="mV"), raw.info

    def load(self, root_dir: str) -> Iterator[Tuple[str, ConditionProperties, mne.Info, SubjectData]]:
        if not os.path.exists(root_dir):
            raise FileNotFoundError(f"Root directory {root_dir} does not exist")

        for folder_name in os.listdir(root_dir):
            folder_path = os.path.join(root_dir, folder_name)
            if not os.path.isdir(folder_path):
                continue
                
            try:
                props = self._parse_folder_name(folder_name)
            except ValueError:
                continue # Skip folders that don't match pattern

            for file_name in os.listdir(folder_path):
                if not file_name.endswith(".edf"):
                    continue
                
                file_path = os.path.join(folder_path, file_name)
                subject_name = os.path.splitext(file_name)[0].split("_raw")[0]
                
                try:
                    data, info = self._load_edf(file_path)
                    processed_data = self._process_data(data)
                    yield subject_name, props, info, processed_data
                except Exception as e:
                    print(f"Error loading {file_path}: {e}")
                    continue