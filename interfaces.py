"""Protocol definitions for multi-paradigm experiment analysis.

Three layers:
- Recording / SSVEPRecording / MNERecording / TopomapSSVEPRecording: a single
  observation's raw-data accessors. Implementors expose data; analysis classes
  in analysis.py do the FFT/SNR/phase work.
- SubjectHandle / Experiment: registry layer keyed by any Hashable.

Implementors only have to supply the few primitives marked as required; every
other method is default-implemented on the Protocol body and may be overridden
for runtime efficiency (see Study in core.py).

Generic parameters:
  K — the condition-key type (bound to Hashable). Each implementor pins K to
  whatever it actually stores (Study uses ConditionProperties; a string-keyed
  Experiment would use str).
  V — the Recording subtype the registry yields. Covariant so a `Dict[K,
  ConditionView]` can satisfy `Mapping[K, SSVEPRecording]`.
"""

from typing import (
    Hashable, Iterable, Iterator, List, Mapping, Protocol, Set, TypeVar, runtime_checkable,
)

import mne

from core_types import RawStudyData


@runtime_checkable
class Recording(Protocol):
    """Paradigm-agnostic, MNE-free raw-data accessor for one observation.

    raw_data is shape (Subject, Trial, Channel, Time). Single-subject views
    have S=1; group views stack subjects along axis 0.
    """

    def name(self) -> str: ...
    def raw_data(self) -> RawStudyData: ...
    def sample_rate(self) -> float: ...
    def channel_names(self) -> List[str]: ...
    def take_channels(self, names: List[str]) -> "Recording": ...


@runtime_checkable
class SSVEPRecording(Recording, Protocol):
    """Recording that knows its SSVEP target and carrier frequencies."""

    def target_frequency(self) -> float: ...
    def carrier_frequency(self) -> float: ...
    def take_channels(self, names: List[str]) -> "SSVEPRecording": ...


@runtime_checkable
class MNERecording(Recording, Protocol):
    """Recording with MNE channel-layout info, needed for topomaps."""

    def mne_info(self) -> mne.Info: ...
    def take_channels(self, names: List[str]) -> "MNERecording": ...


@runtime_checkable
class TopomapSSVEPRecording(SSVEPRecording, MNERecording, Protocol):
    """Combined contract required by plot_snrs (SSVEP + MNE topomap layout)."""

    def take_channels(self, names: List[str]) -> "TopomapSSVEPRecording": ...


K = TypeVar("K", bound=Hashable)
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
