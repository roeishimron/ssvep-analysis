"""Experiment constructors and composition primitives.

Three pieces:

1. `hebrew_vs_mirror` — example wiring for the SSVEP carrier/target paradigm.
   The folder names encode `<carrier>hz_<ob>ob_<dur>s`, parsed by StudyLoader.

2. `dot_experiments` — example wiring for the single-frequency attention
   paradigm. Each folder is one attention condition; the folder identity
   becomes an `AttentionFrequency` key.

3. `compose_folders` — the implementor that takes per-folder streams from a
   folder-name-agnostic loader and re-keys them under user-supplied labels.
   Generic in K so the typed key (e.g. AttentionFrequency, str, an enum) is
   preserved through the stream rather than widened to Hashable.
"""

import os
from typing import Callable, Hashable, Iterable, Iterator, Tuple, TypeVar

import mne
import numpy as np

from core import Study
from core_types import AttentionFrequency, ConditionProperties, RawStudyData
from interfaces import Experiment, MNERecording, MetadataParser, SSVEPRecording
from loader import StudyLoader


K = TypeVar("K", bound=Hashable)
Condition = TypeVar("Condition", bound=Hashable)


def hebrew_vs_mirror(
    root_dir: str,
) -> Experiment[ConditionProperties, SSVEPRecording]:
    """Construct the Experiment for the existing experiments_hebrew_vs_mirror data."""
    return Study(StudyLoader().load(root_dir))


def dot_experiments(
    root_dir: str,
) -> Experiment[AttentionFrequency, MNERecording[AttentionFrequency]]:
    """Construct the Experiment for the dot-experiments single-frequency attention paradigm.

    Reads `<root_dir>/attention_10hz/` and `<root_dir>/attention_15hz/` as the
    two conditions. Each folder is one attention condition; subjects with
    matching filenames in both folders are unified under one Subject in the
    resulting registry.
    """
    loader = StudyLoader()
    named = (
        (AttentionFrequency(np.float64(10.0)), os.path.join(root_dir, "attention_10hz")),
        (AttentionFrequency(np.float64(15.0)), os.path.join(root_dir, "attention_15hz")),
    )
    return Study(compose_folders(named, loader.load_folder), min_trials=3)


def compose_folders(
    named_folders: Iterable[Tuple[K, str]],
    load_folder: Callable[[str], Iterator[Tuple[str, mne.Info, float, RawStudyData]]],
) -> Iterator[Tuple[str, K, mne.Info, float, RawStudyData]]:
    """Combine several folder-streams into one labeled stream consumable by Study.

    Each `(label, folder_path)` pair is loaded via `load_folder`; every emitted
    subject is tagged with the user-supplied label. The label IS the condition
    key — it can be any Hashable; K preserves its precise type so downstream
    `Study[K]` keeps the typed key rather than widening to Hashable.
    """
    for label, folder in named_folders:
        for subject_name, info, sample_rate, data in load_folder(folder):
            yield subject_name, label, info, sample_rate, data


def aggregate_by_metadata(
    raw_stream: Iterable[Tuple[str, mne.Info, float, RawStudyData]],
    metadata_for: Callable[[str], str],
    parser: MetadataParser[Condition],
) -> Iterator[Tuple[str, Condition, mne.Info, float, RawStudyData]]:
    """Apply per-condition aggregation declared by `parser` to each subject's trial data.

    Use this when the upstream stream already produces a rectangular
    (1, T, C, T_trial) per subject and you only need the parser to split
    by condition. For paradigms with per-trial-variable durations use
    `load_segmented` instead — it reads each EDF with per-trial slicing.
    """
    for subject_name, info, sample_rate, trial_data in raw_stream:
        metadata = metadata_for(subject_name)
        for condition_key, condition_data in parser.parse(metadata, trial_data):
            yield subject_name, condition_key, info, sample_rate, condition_data


def load_segmented(
    folder_path: str,
    metadata_for: Callable[[str], str],
    parser: MetadataParser[Condition],
    *,
    loader: "StudyLoader | None" = None,
) -> Iterator[Tuple[str, Condition, mne.Info, float, RawStudyData]]:
    """End-to-end loader for metadata-segmented EDFs with per-trial-variable durations.

    For each .edf in `folder_path`:
      - Looks up the subject's metadata via `metadata_for(subject_name)`.
      - Asks `parser.trial_durations_s(metadata)` for that subject's per-trial
        durations (in seconds, relative to each trigger).
      - Slices the continuous EDF per-trigger via `loader._load_edf_per_trial`,
        zero-padding shorter trials inside the subject's array.
      - Hands the (1, T, C, T_max_for_subject) array to `parser.parse`, which
        yields rectangular per-condition recordings.

    Each subject's T_max differs (depending on their layout), but per-condition
    `T_K` is uniform across subjects when the paradigm is consistent — so the
    downstream `Study` registry stacks subjects rectangularly per condition.
    """
    loader = loader or StudyLoader()
    for file_name in os.listdir(folder_path):
        if not file_name.endswith(".edf"):
            continue

        file_path = os.path.join(folder_path, file_name)
        subject_name = os.path.splitext(file_name)[0].split("_raw")[0]

        try:
            metadata = metadata_for(subject_name)
            durations = parser.trial_durations_s(metadata)
            data, info = loader._load_edf_per_trial(file_path, durations)
            sample_rate = float(info["sfreq"])
            for condition_key, condition_data in parser.parse(metadata, data):
                yield subject_name, condition_key, info, sample_rate, condition_data
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            continue
