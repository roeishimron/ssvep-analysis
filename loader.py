import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Dict, Hashable, Iterable, Iterator, List, Optional, Sequence, Tuple

import mne
import numpy as np
from scipy.signal import butter, sosfiltfilt

from core_types import ConditionProperties, RawStudyData


# --- Channel conventions of the DSI-24 recordings -----------------------------
# The files are saved against Pz, which is present as an all-zero channel. Pz is
# kept as a (zero) scalp channel in the average reference. The earclips are not
# scalp sites and never enter the reference.
RECORDING_REFERENCE = "Pz"
EARCLIPS = ("A1", "A2")

# A new session starts when two files are further apart than this, or when the
# condition is already present in the current session.
SESSION_GAP_S = 3600.0


@dataclass
class Session:
    """One recording session: the files of one participant on one visit.

    Subjects are identified by session (EDF header time), not by file name:
    file names are participants' names and are spelled inconsistently across
    condition folders. `id` is a pseudonym (P01, P02, ...) in chronological order.
    """
    id: str
    start: datetime
    end: datetime
    files: Dict[Hashable, str] = field(default_factory=dict)


def group_sessions(
    entries: Iterable[Tuple[datetime, Hashable, str]],
    max_gap_s: float = SESSION_GAP_S,
) -> List[Session]:
    """Group (meas_date, condition_key, path) entries into chronological sessions.

    A file opens a new session if it is more than `max_gap_s` after the previous
    file of the current session, or if the current session already has a file
    for its condition.
    """
    sessions: List[Session] = []
    for when, key, path in sorted(entries, key=lambda e: (e[0], str(e[1]), e[2])):
        last = sessions[-1] if sessions else None
        if (last is None or (when - last.end).total_seconds() > max_gap_s
                or key in last.files):
            last = Session(id="", start=when, end=when)
            sessions.append(last)
        last.files[key] = path
        last.end = when
    for i, s in enumerate(sessions):
        s.id = f"P{i + 1:02d}"
    return sessions


def summarize_sessions(sessions: Sequence[Session]) -> str:
    """Session counts by the set of conditions present. Pseudonymous IDs only."""
    def carrier(p: Hashable) -> float:
        return float(getattr(p, "carrier_frequency", 0.0))

    patterns: Dict[str, int] = {}
    for s in sessions:
        key = "+".join(f"{carrier(p):g}" for p in sorted(s.files, key=carrier))
        patterns[key] = patterns.get(key, 0) + 1
    n_files = sum(len(s.files) for s in sessions)
    per_pattern = ", ".join(
        f"{k} Hz: {v}" for k, v in sorted(patterns.items(), key=lambda kv: -kv[1])
    )
    return f"{len(sessions)} sessions, {n_files} files; by conditions present: {per_pattern}"


def screen_channels(
    trials: np.ndarray, ch_names: Sequence[str], sample_rate: float,
    flat_uv: float = 0.5, n_mad: float = 3.0, min_mad: float = 0.25,
) -> np.ndarray:
    """Flag bad channels on (Trial, Channel, Time) data in microvolts against Pz.

    A channel is bad if it is flat (1 Hz high-passed SD < `flat_uv` in its worst
    trial) or if its log SD exceeds the median of the scalp channels by more than
    `n_mad` robust SDs (MAD * 1.4826, at least `min_mad`). The recording reference
    is never flagged. Returns a boolean mask over channels.
    """
    names = list(ch_names)
    sos = butter(2, 1.0, btype="high", fs=sample_rate, output="sos")
    sd = sosfiltfilt(sos, trials, axis=-1).std(-1).max(0)          # worst trial per channel
    log_sd = np.log(np.maximum(sd, 1e-6))
    scalp = [i for i, n in enumerate(names)
             if n not in EARCLIPS and n != RECORDING_REFERENCE and sd[i] > flat_uv]
    if not scalp:
        return np.ones(len(names), dtype=bool)
    med = np.median(log_sd[scalp])
    mad = max(1.4826 * np.median(np.abs(log_sd[scalp] - med)), min_mad)
    bad = (sd < flat_uv) | (log_sd > med + n_mad * mad)
    if RECORDING_REFERENCE in names:
        bad[names.index(RECORDING_REFERENCE)] = False
    return bad


def reference_channels(ch_names: Sequence[str], bad: np.ndarray) -> List[int]:
    """Indices entering the average reference: good scalp channels, Pz included, earclips excluded."""
    return [i for i, n in enumerate(ch_names) if n not in EARCLIPS and not bad[i]]


class StudyLoader:
    """Reads DSI-24 EDFs into (1, Trial, Channel, Time) arrays in microvolts.

    epoch_s: (tmin, tmax) relative to the trigger event, tmax inclusive (MNE
        convention). The default (2.5, 60.0) is the legacy epoch used by the
        2 s-window power-SNR and latency code. SPECTRUM_EPOCH_S is the 56 s
        window used by the full-length spectrum (analysis.FrequencyTagging): it
        starts 4.5 s after the trigger event, i.e. about 1.5 s into the stream.
    reference: "screened" (default) - bad channels are screened on
        SCREEN_WINDOW_S of the kept trials, then the data are re-referenced to
        the average of the good scalp channels (Pz as a zero channel, earclips
        excluded). Bad channels are listed in info["bads"].
        "average" - the legacy average of all channels, no screening in the
        reference (bads are still listed).
    n_trials: keep the first n complete trials per file (counts are reported);
        None keeps all.
    """

    SPECTRUM_EPOCH_S = (4.5, 4.5 + 56.0 - 1 / 300.0)
    LEGACY_EPOCH_S = (2.5, 60.0)
    SCREEN_WINDOW_S = (4.5, 60.5)

    def __init__(
        self,
        recording_frequency: float = 300.0,
        window_duration_s: float = 2.0,
        *,
        epoch_s: Tuple[float, float] = LEGACY_EPOCH_S,
        reference: str = "screened",
        n_trials: Optional[int] = 3,
        log: Callable[[str], None] = print,
    ):
        if reference not in ("screened", "average"):
            raise ValueError(f"unknown reference {reference!r}")
        self.recording_frequency = recording_frequency
        self.window_duration_s = window_duration_s
        self.epoch_s = epoch_s
        self.reference = reference
        self.n_trials = n_trials
        self.log = log
        self.sessions: List[Session] = []
        # One record per loaded file: label, trigger counts, bad channels.
        self.records: List[dict] = []

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

    def _read_raw(self, file_path: str) -> Tuple[mne.io.BaseRaw, np.ndarray, int]:
        """Read one EDF, returning (raw against Pz, valid_events, n_bursts). No re-referencing.

        Trigger filtering: keep events whose gap to the next event exceeds 1000
        samples, i.e. the last event of each trigger burst. n_bursts counts every
        burst of changes on the trigger line, so a burst the event rule misses
        (e.g. one too close to the end of the file) still shows in the report.
        """
        raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)

        raw.rename_channels(lambda s: s.replace(
            "EEG ", "").replace("-Pz", ""), False)

        try:
            raw.drop_channels(['Ax', 'Ay', 'Az'])
        except ValueError:
            raw.drop_channels(['X1:', 'X2:', 'X3:'])

        raw.drop_channels(['Event', 'CM'])
        raw.drop_channels([c for c in raw.ch_names if ":" in c])
        raw.set_montage(montage='standard_1020')

        events = mne.find_events(raw, stim_channel="Trigger", mask=8, verbose=False)
        changes = np.flatnonzero(np.diff(raw.get_data(picks=["Trigger"])[0]) != 0)
        n_bursts = int(len(changes) and 1 + np.sum(np.diff(changes) > 1000))
        raw.drop_channels(["Trigger"])

        diffs = np.diff(events[:, 0], append=raw.last_samp)
        valids = np.argwhere(diffs > 1000).flatten()
        return raw, events[valids], n_bursts

    def _read_raw_with_events(self, file_path: str) -> Tuple[mne.io.BaseRaw, np.ndarray]:
        """`_read_raw` plus the legacy average reference over all channels.

        Used by `_load_edf_per_trial` (segmented paradigms), whose variable trial
        lengths do not define a screening window.
        """
        raw, events, _ = self._read_raw(file_path)
        raw.set_eeg_reference(verbose=False)
        return raw, events

    def _load_edf(
        self, file_path: str, duration: float, label: str = "",
    ) -> Tuple[RawStudyData, mne.Info]:
        """Epoch one file. Returns ((1, T, C, Time) in microvolts, info with info['bads']).

        `duration` (from the folder name) is kept for API compatibility; the
        epoch is set by `self.epoch_s`. Only complete trials are kept: the
        recording must extend past both the epoch and the screening window.
        """
        del duration
        raw, events, n_bursts = self._read_raw(file_path)
        fs = float(raw.info["sfreq"])
        names = list(raw.ch_names)
        x = raw.get_data(units="uV")                                   # (C, N) against Pz
        onsets = events[:, 0] - raw.first_samp

        ep0, ep1 = int(round(self.epoch_s[0] * fs)), int(round(self.epoch_s[1] * fs)) + 1
        sc0, sc1 = (int(round(t * fs)) for t in self.SCREEN_WINDOW_S)
        start, stop = min(ep0, sc0), max(ep1, sc1)
        complete = [int(e) for e in onsets if e + start >= 0 and e + stop <= x.shape[1]]
        used = complete if self.n_trials is None else complete[:self.n_trials]
        if not used:
            raise ValueError(f"no complete trial ({len(onsets)} trigger events)")

        bad = screen_channels(np.stack([x[:, e + sc0:e + sc1] for e in used]), names, fs)
        if self.reference == "screened":
            x = x - x[reference_channels(names, bad)].mean(0, keepdims=True)
        else:
            x = x - x.mean(0, keepdims=True)

        data = np.stack([x[:, e + ep0:e + ep1] for e in used]).astype(np.float64)
        info = raw.info.copy()
        info["bads"] = [names[i] for i in np.flatnonzero(bad)]

        self.records.append(dict(label=label, n_bursts=n_bursts, n_events=len(onsets),
                                 n_complete=len(complete), n_used=len(used),
                                 bads=list(info["bads"])))
        note = "" if n_bursts == len(used) else "  <- not every trigger burst used"
        self.log(f"  {label}: {n_bursts} trigger bursts, {len(onsets)} stream onsets detected, "
                 f"{len(complete)} complete, using the first {len(used)}; "
                 f"bad channels: {info['bads'] or 'none'}{note}")
        return data[np.newaxis, ...], info

    def _load_edf_per_trial(
        self, file_path: str, trial_durations_s: List[float],
    ) -> Tuple[RawStudyData, mne.Info]:
        """Slice the continuous EDF per trigger using metadata-declared durations.

        Each trial gets its own `mne.Epochs` call so tmax can vary per
        trigger. Within a subject's array, shorter trials are zero-padded
        to the longest trial's sample length so the (T, C, T_max) shape
        stays rectangular; the parser aggregates only over each trial's
        valid (un-padded) range.
        """
        raw, events = self._read_raw_with_events(file_path)

        n_trials = len(trial_durations_s)
        if len(events) != n_trials:
            raise ValueError(
                f"{len(events)} valid triggers, metadata declares {n_trials} trials"
            )

        durations = [2.5 + d for d in trial_durations_s]
        per_trial: List[np.ndarray] = []
        for event, duration in zip(events, durations):
            epoch = mne.Epochs(
                raw,
                picks='data',
                events=np.array([event]),
                tmin=2.5,  # Constant for 3 seconds delay minus 0.5 sec
                tmax=duration,
                baseline=None,
                verbose=False,
            )
            arr = np.asarray(epoch.get_data(units="uV"), dtype=np.float64)
            per_trial.append(arr[0])  # (1, C, T_i) -> (C, T_i)

        max_samples = max(arr.shape[-1] for arr in per_trial)
        n_channels = per_trial[0].shape[0]
        output = np.zeros((n_trials, n_channels, max_samples), dtype=np.float64)
        for i, arr in enumerate(per_trial):
            output[i, :, :arr.shape[-1]] = arr

        return output[np.newaxis, ...], raw.info

    @staticmethod
    def _scrub(message: str, file_path: str) -> str:
        """Remove the file path and name (participants' names) from a message."""
        base = os.path.basename(file_path)
        for s in (file_path, base, os.path.splitext(base)[0]):
            message = message.replace(s, "<file>")
        return message

    def load_folder(
        self, folder_path: str, duration: float = 60.0,
    ) -> Iterator[Tuple[str, mne.Info, float, RawStudyData]]:
        """Read all .edf files in one folder, yielding (subject_name, info, sample_rate, raw_data).

        Folder-name-agnostic: used by experiments.compose_folders, where the
        caller supplies the condition label. Subjects are named by file stem
        here; `load` names them by recording session instead.
        """
        for file_name in sorted(os.listdir(folder_path)):
            if not file_name.endswith(".edf"):
                continue

            file_path = os.path.join(folder_path, file_name)
            subject_name = os.path.splitext(file_name)[0].split("_raw")[0]

            try:
                data, info = self._load_edf(file_path, duration, label=subject_name)
                yield subject_name, info, float(info["sfreq"]), data
            except Exception as e:
                print(f"Error loading a file in {os.path.basename(folder_path)}: "
                      f"{type(e).__name__}: {self._scrub(str(e), file_path)}")
                continue

    def list_sessions(self, root_dir: str) -> List[Session]:
        """Group the EDFs of all condition folders under `root_dir` into sessions.

        Only .edf files directly inside a condition folder count; subfolders
        (e.g. `bads`) are ignored. Reads headers only.
        """
        entries = []
        for folder_name in sorted(os.listdir(root_dir)):
            folder_path = os.path.join(root_dir, folder_name)
            if not os.path.isdir(folder_path):
                continue
            try:
                props, _ = self._parse_folder_name(folder_name)
            except ValueError:
                continue
            for file_name in sorted(os.listdir(folder_path)):
                path = os.path.join(folder_path, file_name)
                if not file_name.endswith(".edf") or not os.path.isfile(path):
                    continue
                raw = mne.io.read_raw_edf(path, preload=False, verbose=False)
                entries.append((raw.info["meas_date"], props, path))
        return group_sessions(entries)

    def load(
        self, root_dir: str,
    ) -> Iterator[Tuple[str, ConditionProperties, mne.Info, float, RawStudyData]]:
        """Yield (session_id, condition, info, sample_rate, data) for every file.

        Subjects are recording sessions with pseudonymous IDs (see Session);
        file names are never used as identities or printed.
        """
        if not os.path.exists(root_dir):
            raise FileNotFoundError(f"Root directory {root_dir} does not exist")

        self.sessions = self.list_sessions(root_dir)
        self.log(summarize_sessions(self.sessions))
        for session in self.sessions:
            for props in sorted(session.files, key=lambda p: p.carrier_frequency):
                file_path = session.files[props]
                label = f"{session.id} {props.carrier_frequency:g} Hz"
                try:
                    data, info = self._load_edf(file_path, 60.0, label=label)
                except Exception as e:
                    print(f"Error loading {label}: {type(e).__name__}: "
                          f"{self._scrub(str(e), file_path)}")
                    continue
                yield session.id, props, info, float(info["sfreq"]), data
