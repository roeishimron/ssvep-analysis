import os
import re
from typing import Iterator, Tuple

import mne
import numpy as np

from core_types import ConditionProperties, RawStudyData


class StudyLoader:
    def __init__(self, recording_frequency: float = 300.0, window_duration_s: float = 2.0):
        self.recording_frequency = recording_frequency
        self.window_duration_s = window_duration_s

    def _parse_folder_name(self, folder_name: str) -> Tuple[ConditionProperties, float]:
        # Expected pattern: 10hz_2ob_60s
        match = re.match(r"(\d+)hz_(\d+)ob_(\d+)s", folder_name)
        if not match:
            # Try without duration as fallback
            match = re.match(r"(\d+)hz_(\d+)ob", folder_name)
            if not match:
                raise ValueError(f"Folder name {folder_name} does not match expected pattern <freq>hz_<ob>ob[_<dur>s]")
            duration = 60.0  # Default
        else:
            duration = float(match.group(3))

        carrier_freq = float(match.group(1))
        modulation = float(match.group(2))
        target_freq = carrier_freq / modulation

        return (
            ConditionProperties(
                target_frequency=np.float64(target_freq),
                carrier_frequency=np.float64(carrier_freq),
            ),
            duration,
        )

    def _load_edf(self, file_path: str, duration: float) -> Tuple[RawStudyData, mne.Info]:
        raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)

        raw.rename_channels(lambda s: s.replace(
            "EEG ", "").replace("-Pz", ""), False)

        raw.drop_channels(['Ax', 'Ay', 'Az'])
        raw.drop_channels(['Event', 'CM'])
        raw.drop_channels([c for c in raw.ch_names if ":" in c])
        raw.set_montage(montage='standard_1020')

        events = mne.find_events(raw, stim_channel="Trigger", mask=8)
        raw.drop_channels(["Trigger"])

        raw.set_eeg_reference()

        diffs = np.diff(events[:, 0], append=raw.last_samp)
        valids = np.argwhere(diffs > 1000).flatten()
        events = events[valids]

        epochs = mne.Epochs(
            raw,
            picks='data',
            events=events,
            tmin=2.5,  # Constant for 3 seconds delay minus 0.5 sec
            tmax=duration,
            baseline=None,
        )

        # epochs.get_data: (Trial, Channel, Time). Promote to (1, T, C, Time)
        # so it lines up with the (S, T, C, Time) RawStudyData contract.
        per_subject = epochs.get_data(units="mV").astype(np.float64)
        return per_subject[np.newaxis, ...], raw.info

    def load_folder(
        self, folder_path: str, duration: float = 60.0,
    ) -> Iterator[Tuple[str, mne.Info, float, RawStudyData]]:
        """Read all .edf files in one folder, yielding (subject_name, info, sample_rate, raw_data).

        Folder-name-agnostic — used both by `load` (which parses paradigm-specific
        folder names for ConditionProperties) and by experiments.compose_folders
        (where the caller supplies the condition label directly).
        """
        for file_name in os.listdir(folder_path):
            if not file_name.endswith(".edf"):
                continue

            file_path = os.path.join(folder_path, file_name)
            subject_name = os.path.splitext(file_name)[0].split("_raw")[0]

            try:
                data, info = self._load_edf(file_path, duration)
                yield subject_name, info, float(info["sfreq"]), data
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
                continue

    def load(
        self, root_dir: str,
    ) -> Iterator[Tuple[str, ConditionProperties, mne.Info, float, RawStudyData]]:
        if not os.path.exists(root_dir):
            raise FileNotFoundError(f"Root directory {root_dir} does not exist")

        for folder_name in os.listdir(root_dir):
            folder_path = os.path.join(root_dir, folder_name)
            if not os.path.isdir(folder_path):
                continue

            try:
                props, duration = self._parse_folder_name(folder_name)
            except ValueError:
                continue  # Skip folders that don't match the SSVEP-paradigm pattern

            for subject_name, info, sample_rate, data in self.load_folder(folder_path, duration):
                yield subject_name, props, info, sample_rate, data
