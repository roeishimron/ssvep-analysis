import sys
from typing import Dict, Generic, Hashable, Iterator, List, Tuple, TypeVar

import mne
import numpy as np

from core_types import (
    Array1D_i64, ConditionProperties, RawStudyData,
)
from interfaces import (
    Experiment, MNERecording, Recording, SubjectHandle,
)


K = TypeVar("K", bound=Hashable)


class ConditionBlob(Generic[K]):
    """Heavy-duty container for a single condition's raw time-domain data.

    Storage shape: (Subject, Trial, Channel, Time-samples). Generic in the
    condition-key/metadata type K (e.g. ConditionProperties for SSVEP,
    AttentionFrequency for the dots paradigm).
    """

    def __init__(
        self,
        raw_data: RawStudyData,
        sample_rate: float,
        props: K,
        raw_info: mne.Info,
        subjects: List["Subject[K]"],
    ) -> None:
        if raw_data.ndim != 4:
            raise ValueError(
                f"raw_data must be 4D (Subject, Trial, Channel, Time), got {raw_data.ndim}D"
            )
        if raw_data.shape[0] != len(subjects):
            raise ValueError(
                f"Subject dimension size ({raw_data.shape[0]}) must match "
                f"number of subjects ({len(subjects)})"
            )

        self.raw_data: RawStudyData = raw_data.astype(np.float64)
        self.sample_rate = sample_rate
        self.props: K = props
        self.raw_info = raw_info
        self.subjects = subjects

    @property
    def n_subjects(self) -> int:
        return self.raw_data.shape[0]

    @property
    def n_trials(self) -> int:
        return self.raw_data.shape[1]

    @property
    def n_channels(self) -> int:
        return self.raw_data.shape[2]

    @property
    def n_time(self) -> int:
        return self.raw_data.shape[3]


class ConditionView(Generic[K]):
    """A configured "lens" into a ConditionBlob.

    Implements Recording[K] and MNERecording[K]. Carries no analytical
    methods — those live in analysis.Spectral / SSVEPAnalysis. SSVEP-specific
    metadata (target/carrier frequency) is reached via `view.props()`, which
    returns the K-typed condition key from the underlying blob.
    """

    def __init__(
        self,
        blob: ConditionBlob[K],
        subject_idx: int | None = None,
        channel_indices: Array1D_i64 | None = None,
        name: str | None = None,
    ) -> None:
        self.blob = blob
        self.subject_idx = subject_idx
        if channel_indices is None:
            self.channel_indices: Array1D_i64 = np.arange(
                blob.n_channels, dtype=np.int64,
            )
        else:
            self.channel_indices = channel_indices

        if name:
            self._name = name
        elif subject_idx is not None:
            self._name = blob.subjects[subject_idx].name
        elif len(blob.subjects) == 1:
            self._name = blob.subjects[0].name
        else:
            self._name = f"Group({len(blob.subjects)})"

    # --- Recording[K] ---
    def name(self) -> str:
        return self._name

    def raw_data(self) -> RawStudyData:
        if self.subject_idx is None:
            return self.blob.raw_data[:, :, self.channel_indices, :]
        return self.blob.raw_data[
            self.subject_idx : self.subject_idx + 1, :, self.channel_indices, :,
        ]

    def sample_rate(self) -> float:
        return self.blob.sample_rate

    def channel_names(self) -> List[str]:
        all_names = self.blob.raw_info.ch_names
        return [all_names[i] for i in self.channel_indices]

    def take_channels(self, names: List[str]) -> "ConditionView[K]":
        all_names = self.blob.raw_info.ch_names
        wanted = [all_names.index(n) for n in names if n in all_names]
        new_indices: Array1D_i64 = np.intersect1d(
            self.channel_indices, np.array(wanted, dtype=np.int64),
        )
        return ConditionView(self.blob, self.subject_idx, new_indices, name=self._name)

    def props(self) -> K:
        return self.blob.props

    # --- MNERecording[K] ---
    def mne_info(self) -> mne.Info:
        # Restrict the info to the active channel subset so topomaps line up.
        if len(self.channel_indices) == len(self.blob.raw_info.ch_names):
            return self.blob.raw_info
        keep = self.channel_names()
        return mne.pick_info(
            self.blob.raw_info,
            sel=[self.blob.raw_info.ch_names.index(n) for n in keep],
            copy=True,
        )


class Subject(Generic[K], SubjectHandle[K, ConditionView[K]]):
    """A lightweight handle representing a participant in a study.

    Identity is defined by a unique integer id. Carries no data — the
    per-condition views live in self._views and are populated by Study at
    load time. Generic in the condition-key type K.
    """

    def __init__(self, subject_id: int, name: str) -> None:
        self.id = subject_id
        self.name = name
        self._views: Dict[K, ConditionView[K]] = {}

    def conditions(self) -> Dict[K, ConditionView[K]]:
        return self._views

    def __getitem__(self, key: K) -> ConditionView[K]:
        view = self._views.get(key)
        if view is None:
            raise KeyError(
                f"Subject {self.name} (id={self.id}) did not participate in condition {key}"
            )
        return view

    def _register_view(self, key: K, view: ConditionView[K]) -> None:
        self._views[key] = view

    def __repr__(self) -> str:
        return f"Subject({self.name}, id={self.id})"

    def __hash__(self) -> int:
        return hash(self.id)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Subject):
            return False
        return self.id == other.id


class Study(Generic[K], Experiment[K, ConditionView[K]]):
    """A paradigm-agnostic experiment registry, generic in the condition key K.

    Inherits Experiment[K, ConditionView[K]]. Data is pre-stacked into
    ConditionBlobs at construction; conditions() and aggregate() are
    overridden for runtime efficiency. Subjects appear in alphabetical order;
    conditions appear in insertion order (i.e. the order they were first seen
    in the input stream).

    The K type pins which condition-key NamedTuple this study uses
    (ConditionProperties for SSVEP, AttentionFrequency for dots, etc).
    Analyses that require a particular K (e.g. SSVEPAnalysis requires
    K=ConditionProperties) will be type-checked at the call site.
    """

    def __init__(
        self,
        data_stream: Iterator[Tuple[str, K, mne.Info, float, RawStudyData]],
        min_trials: int = 3,
    ) -> None:
        self._subjects_by_name: Dict[str, Subject[K]] = {}
        self._next_id = 0

        # Buffer raw time-domain data per (condition_key, subject_name).
        buffer: Dict[K, Dict[str, List[np.ndarray]]] = {}
        infos: Dict[K, mne.Info] = {}
        rates: Dict[K, float] = {}

        for name, props, info, sample_rate, data in data_stream:
            if name not in self._subjects_by_name:
                self._subjects_by_name[name] = Subject(self._next_id, name)
                self._next_id += 1

            if props not in buffer:
                buffer[props] = {}
                infos[props] = info
                rates[props] = sample_rate

            buffer[props].setdefault(name, []).append(data)

        self._blobs: Dict[K, ConditionBlob[K]] = {}
        self._group_views: Dict[K, ConditionView[K]] = {}

        for props, subject_data_map in buffer.items():
            sorted_names = sorted(subject_data_map.keys())
            subject_handles = [self._subjects_by_name[n] for n in sorted_names]

            subject_trial_data: List[np.ndarray] = []
            for name in sorted_names:
                # Concatenate this subject's recordings along the trial axis.
                # Each recording has shape (1, T_i, C, Time); strip the leading
                # subject axis before concatenating along trials.
                per_recording = [d[0] for d in subject_data_map[name]]
                combined = np.concatenate(per_recording, axis=0)
                n_trials = combined.shape[0]

                if n_trials < min_trials:
                    print(
                        f"\nERROR: Subject '{name}' has only {n_trials} trials for condition {props}, "
                        f"which is less than the minimum required ({min_trials})."
                    )
                    sys.exit(1)

                subject_trial_data.append(combined[:min_trials])

            blob_data: RawStudyData = np.stack(subject_trial_data, axis=0).astype(np.float64)
            blob = ConditionBlob(blob_data, rates[props], props, infos[props], subject_handles)
            self._blobs[props] = blob

            self._group_views[props] = ConditionView(blob)

            for i, subj in enumerate(subject_handles):
                view = ConditionView(blob, subject_idx=i)
                subj._register_view(props, view)

        # Subjects sorted alphabetically by name. Conditions keep insertion
        # order from the input stream (Python dict default).
        self._subjects_by_name = {
            n: self._subjects_by_name[n] for n in sorted(self._subjects_by_name.keys())
        }

    # --- Experiment[K, ConditionView[K]] ---
    def subjects(self) -> Dict[str, Subject[K]]:
        return self._subjects_by_name

    def conditions(self) -> Dict[K, ConditionView[K]]:
        return self._group_views

    def filter_subjects(
        self, requirements,
    ) -> Iterator[Subject[K]]:
        req_set = set(requirements)
        for subj in self._subjects_by_name.values():
            if req_set.issubset(subj.conditions().keys()):
                yield subj

    def aggregate(self, key: K) -> ConditionView[K]:
        blob = self._blobs.get(key)
        if blob is None:
            raise KeyError(f"Condition {key} not found in study")

        s, t, c, time = blob.raw_data.shape
        flat: RawStudyData = blob.raw_data.reshape(1, s * t, c, time).astype(np.float64)
        new_blob: ConditionBlob[K] = ConditionBlob(
            flat, blob.sample_rate, blob.props, blob.raw_info,
            [Subject(-1, "Aggregate")],
        )
        label = f"Aggregate({blob.props})"
        return ConditionView(new_blob, name=label)
