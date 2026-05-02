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
from interfaces import Experiment, MNERecording, SSVEPRecording
from loader import StudyLoader


K = TypeVar("K", bound=Hashable)


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
