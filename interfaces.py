"""Protocol definitions for multi-paradigm experiment analysis.

Three layers:
- Recording[K] / MNERecording[K]: a single observation's raw-data accessors,
  generic in the condition-key/metadata type K. SSVEPRecording and
  TopomapSSVEPRecording are type aliases for the K=ConditionProperties cases.
- SubjectHandle[K, V] / Experiment[K, V]: registry layer keyed by K.

Implementors only have to supply the few primitives marked as required; every
other method on Experiment is default-implemented on the Protocol body and may
be overridden for runtime efficiency (see Study in core.py).

Generic parameters:
  K — the condition-key/metadata type (bound to Hashable). Each implementor
  pins K to whatever it actually stores (Study uses ConditionProperties or
  AttentionFrequency depending on the paradigm).
  V — the Recording subtype the registry yields. Covariant so a `Dict[K,
  ConditionView[K]]` can satisfy `Mapping[K, Recording[K]]`.
"""

from typing import (
    Hashable, Iterable, Iterator, List, Mapping, Protocol, Set, Tuple, TypeVar, runtime_checkable,
)

import mne

from core_types import ConditionProperties, RawStudyData


# K_co (covariant) is used in the Recording / MNERecording protocols, where K
# only ever appears in return positions (props() -> K, take_channels() ->
# Recording[K]). K (invariant) is used as the dict key in SubjectHandle /
# Experiment, where invariance is required for Mapping-typed APIs.
K_co = TypeVar("K_co", bound=Hashable, covariant=True)
K = TypeVar("K", bound=Hashable)
Condition = TypeVar("Condition", bound=Hashable, covariant=True)


@runtime_checkable
class Recording(Protocol[K_co]):
    """Paradigm-agnostic, MNE-free raw-data accessor for one observation.

    raw_data is shape (Subject, Trial, Channel, Time). Single-subject views
    have S=1; group views stack subjects along axis 0.

    `props()` exposes the condition-key/metadata associated with this
    recording. The key type K is what differentiates paradigms — e.g.
    ConditionProperties for SSVEP, AttentionFrequency for the dots paradigm.
    """

    def name(self) -> str: ...
    def raw_data(self) -> RawStudyData: ...
    def sample_rate(self) -> float: ...
    def channel_names(self) -> List[str]: ...
    def take_channels(self, names: List[str]) -> "Recording[K_co]": ...
    def props(self) -> K_co: ...


@runtime_checkable
class MNERecording(Recording[K_co], Protocol[K_co]):
    """Recording with MNE channel-layout info, needed for topomaps."""

    def mne_info(self) -> mne.Info: ...
    def take_channels(self, names: List[str]) -> "MNERecording[K_co]": ...


# Type aliases for the SSVEP paradigm — pin K to ConditionProperties.
# Analyses that need target/carrier frequencies should use these aliases in
# their signatures; the type system will then reject views whose K is not
# ConditionProperties (e.g. AttentionFrequency from the dots paradigm).
SSVEPRecording = Recording[ConditionProperties]
TopomapSSVEPRecording = MNERecording[ConditionProperties]


V = TypeVar("V", bound=Recording, covariant=True)


class SubjectHandle(Protocol[K, V]):
    """One participant's per-condition recordings, keyed by K.

    Required: name, id, conditions().
    """

    name: str
    id: int

    def conditions(self) -> Mapping[K, V]: ...

    def __getitem__(self, key: K) -> V:
        return self.conditions()[key]


class Experiment(Protocol[K, V]):
    """A registry of subjects and conditions, keyed by K.

    Required: subjects().
    Default-implemented: conditions(), aggregate(key), filter_subjects(reqs).
    Implementors MAY override the defaults for runtime efficiency — see
    core.Study, which pre-builds multi-subject blobs and serves conditions()
    / aggregate() from them.
    """

    def subjects(self) -> Mapping[str, SubjectHandle[K, V]]: ...

    def conditions(self) -> Mapping[K, V]:
        from analysis import combine_subject_recordings

        keys: Set[K] = set()
        for subj in self.subjects().values():
            keys.update(subj.conditions().keys())

        out: dict = {}
        for k in keys:
            recs = [
                subj.conditions()[k]
                for subj in self.subjects().values()
                if k in subj.conditions()
            ]
            out[k] = combine_subject_recordings(recs)
        return out

    def aggregate(self, key: K) -> V:
        from analysis import flatten_subject_axis

        return flatten_subject_axis(self.conditions()[key])  # type: ignore[return-value]

    def filter_subjects(
        self, requirements: Iterable[K],
    ) -> Iterator[SubjectHandle[K, V]]:
        req_set: Set[K] = set(requirements)
        for subj in self.subjects().values():
            if req_set.issubset(subj.conditions().keys()):
                yield subj


@runtime_checkable
class MetadataParser(Protocol[Condition]):
    """Interprets a metadata configuration string and aggregates per-condition data.

    `parse(metadata, trial_data)` receives the metadata file's contents as a
    string (the caller does the I/O) and the post-epoch (1, T, C, T_trial)
    array, and returns one (condition_key, condition_data) pair per condition
    declared in the metadata. Each `condition_data` is shape (1, T, C, T_K) —
    the parser performs whatever per-trial sample selection / aggregation is
    required (contiguous slice, interleaved pickup, reordering); the resulting
    time axis is uniform across trials so the array stays rectangular.

    File format (JSON, CSV, TOML, ...) is the parser's concern — the composer
    never opens the metadata file.
    """

    def parse(
        self, metadata: str, trial_data: RawStudyData,
    ) -> Iterable[Tuple[Condition, RawStudyData]]: ...
