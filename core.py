import sys
from typing import Dict, Iterator, List, Tuple

import mne
import numpy as np

from core_types import (
    Array1D_i64, ConditionProperties, RawStudyData,
)
from interfaces import (
    Experiment, MNERecording, SSVEPRecording, SubjectHandle,
)


class ConditionBlob:
    """Heavy-duty container for a single condition's raw time-domain data.

    Storage shape: (Subject, Trial, Channel, Time-samples).
    """

    def __init__(
        self,
        raw_data: RawStudyData,
        sample_rate: float,
        props: ConditionProperties,
        raw_info: mne.Info,
        subjects: List["Subject"],
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
        self.props = props
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


class ConditionView:
    """A configured "lens" into a ConditionBlob.

    Implements Recording, SSVEPRecording, MNERecording (and therefore
    TopomapSSVEPRecording). Carries no analytical methods — those live in
    analysis.SSVEPAnalysis (which takes any SSVEPRecording).
    """

    def __init__(
        self,
        blob: ConditionBlob,
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

    # --- Recording ---
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

    def take_channels(self, names: List[str]) -> "ConditionView":
        all_names = self.blob.raw_info.ch_names
        wanted = [all_names.index(n) for n in names if n in all_names]
        new_indices: Array1D_i64 = np.intersect1d(
            self.channel_indices, np.array(wanted, dtype=np.int64),
        )
        return ConditionView(self.blob, self.subject_idx, new_indices, name=self._name)

    # --- SSVEPRecording ---
    def target_frequency(self) -> float:
        return float(self.blob.props.target_frequency)

    def carrier_frequency(self) -> float:
        return float(self.blob.props.carrier_frequency)

    # --- MNERecording ---
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


class Subject(SubjectHandle[ConditionProperties, ConditionView]):
    """A lightweight handle representing a participant in an SSVEP study.

    Identity is defined by a unique integer id. Carries no data — the per-condition
    views live in self._views and are populated by Study at load time. Pinned to
    ConditionProperties keys, matching Study's enforcement.
    """

    def __init__(self, subject_id: int, name: str) -> None:
        self.id = subject_id
        self.name = name
        self._views: Dict[ConditionProperties, ConditionView] = {}

    def conditions(self) -> Dict[ConditionProperties, ConditionView]:
        return self._views

    def __getitem__(self, key: ConditionProperties) -> ConditionView:
        view = self._views.get(key)
        if view is None:
            raise KeyError(
                f"Subject {self.name} (id={self.id}) did not participate in condition {key}"
            )
        return view

    def _register_view(self, key: ConditionProperties, view: ConditionView) -> None:
        self._views[key] = view

    def __repr__(self) -> str:
        return f"Subject({self.name}, id={self.id})"

    def __hash__(self) -> int:
        return hash(self.id)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Subject):
            return False
        return self.id == other.id


class Study(Experiment[ConditionProperties, ConditionView]):
    """SSVEP-paradigm experiment registry.

    Inherits Experiment[ConditionView]. Stream is typed as ConditionProperties-keyed
    — that's the type-level enforcement that every blob in this Study has real
    SSVEP target/carrier frequencies. Non-SSVEP paradigms (e.g. plain string
    labels from `compose_folders`) need a different Experiment implementor.

    Overrides conditions() and aggregate() for runtime efficiency: data is
    pre-stacked into ConditionBlobs at construction and the views are reused.
    """

    def __init__(
        self,
        data_stream: Iterator[Tuple[str, ConditionProperties, mne.Info, float, RawStudyData]],
        min_trials: int = 3,
    ) -> None:
        self._subjects_by_name: Dict[str, Subject] = {}
        self._next_id = 0

        # Buffer raw time-domain data per (condition_key, subject_name).
        buffer: Dict[ConditionProperties, Dict[str, List[np.ndarray]]] = {}
        infos: Dict[ConditionProperties, mne.Info] = {}
        rates: Dict[ConditionProperties, float] = {}

        for name, props, info, sample_rate, data in data_stream:
            if name not in self._subjects_by_name:
                self._subjects_by_name[name] = Subject(self._next_id, name)
                self._next_id += 1

            if props not in buffer:
                buffer[props] = {}
                infos[props] = info
                rates[props] = sample_rate

            buffer[props].setdefault(name, []).append(data)

        self._blobs: Dict[ConditionProperties, ConditionBlob] = {}
        self._group_views: Dict[ConditionProperties, ConditionView] = {}

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

        # Deterministic iteration order:
        # - subjects sorted alphabetically by name
        # - conditions sorted by (target_frequency, carrier_frequency)
        self._subjects_by_name = {
            n: self._subjects_by_name[n] for n in sorted(self._subjects_by_name.keys())
        }

        ordered_props = sorted(
            self._group_views.keys(),
            key=lambda p: (float(p.target_frequency), float(p.carrier_frequency)),
        )
        self._group_views = {p: self._group_views[p] for p in ordered_props}
        self._blobs = {p: self._blobs[p] for p in ordered_props}

    # --- Experiment ---
    def subjects(self) -> Dict[str, Subject]:
        return self._subjects_by_name

    def conditions(self) -> Dict[ConditionProperties, ConditionView]:
        return self._group_views

    def filter_subjects(
        self, requirements,
    ) -> Iterator[Subject]:
        req_set = set(requirements)
        for subj in self._subjects_by_name.values():
            if req_set.issubset(subj.conditions().keys()):
                yield subj

    def aggregate(self, key: ConditionProperties) -> ConditionView:
        blob = self._blobs.get(key)
        if blob is None:
            raise KeyError(f"Condition {key} not found in study")

        s, t, c, time = blob.raw_data.shape
        flat: RawStudyData = blob.raw_data.reshape(1, s * t, c, time).astype(np.float64)
        new_blob = ConditionBlob(
            flat, blob.sample_rate, blob.props, blob.raw_info,
            [Subject(-1, "Aggregate")],
        )
        # Stream type guarantees only ConditionProperties keys are stored, so
        # blob.props is always a real ConditionProperties (no placeholder branch).
        label = f"Aggregate({blob.props.target_frequency}Hz)"
        return ConditionView(new_blob, name=label)
