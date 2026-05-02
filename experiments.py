"""Experiment constructors and composition primitives.

Two pieces:

1. `hebrew_vs_mirror` — example wiring for the current SSVEP paradigm.
   Demonstrates how a paradigm becomes an Experiment[K, V] in one line.

2. `compose_folders` — the implementor that takes per-folder streams
   from a folder-name-agnostic loader and re-keys them under user-supplied
   labels. Lets a paradigm where each folder represents one condition (no
   in-folder structure to parse) plug into the same Study/Experiment
   machinery as the SSVEP paradigm.
"""

from typing import Callable, Hashable, Iterable, Iterator, Tuple

import mne

from core import Study
from core_types import ConditionProperties, RawStudyData
from interfaces import Experiment, SSVEPRecording
from loader import StudyLoader


def hebrew_vs_mirror(root_dir: str) -> Experiment[ConditionProperties, SSVEPRecording]:
    """Construct the Experiment for the existing experiments_hebrew_vs_mirror data."""
    return Study(StudyLoader().load(root_dir))


def compose_folders(
    named_folders: Iterable[Tuple[Hashable, str]],
    load_folder: Callable[[str], Iterator[Tuple[str, mne.Info, float, RawStudyData]]],
) -> Iterator[Tuple[str, Hashable, mne.Info, float, RawStudyData]]:
    """Combine several folder-streams into one labeled stream consumable by Study.

    Each `(label, folder_path)` pair is loaded via `load_folder`; every emitted
    subject is tagged with the user-supplied label. The label IS the condition
    key — it can be any Hashable.

    Example:
        loader = StudyLoader()
        stream = compose_folders(
            (("attention_10hz", "dot-experiments/attention_10hz"),
             ("attention_15hz", "dot-experiments/attention_15hz")),
            loader.load_folder,
        )
        exp = Study(stream)   # Experiment[Hashable, ConditionView]

    Note: when the label is not a `ConditionProperties`, the resulting
    recordings will not carry meaningful target/carrier frequencies — analyses
    that depend on those (e.g. `SSVEPAnalysis.snr_at_target`) won't apply.
    """
    for label, folder in named_folders:
        for subject_name, info, sample_rate, data in load_folder(folder):
            yield subject_name, label, info, sample_rate, data
