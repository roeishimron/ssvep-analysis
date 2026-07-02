"""Analysis classes and protocol-plumbing helpers.

Two analysis classes:
- Spectral(rec): FFT-cached spectral analysis on any Recording.
- SSVEPAnalysis(rec): adds target/carrier convenience + neural-processing-time
  (phase ambiguity resolution) for SSVEPRecording inputs.

Also exports:
- RawRecording: concrete dataclass-style Recording implementor used by the
  default Experiment.conditions() / aggregate() paths.
- combine_subject_recordings, flatten_subject_axis: helpers used by those
  defaults.
"""

from typing import Generic, Hashable, List, Sequence, Tuple, TypeVar

import numpy as np
from scipy.signal.windows import kaiser
from scipy.stats import circmean, circstd, sem

from core_types import (
    Array1D_c64, Array1D_f64, ConditionPhases, RawStudyData,
    StudyData, StudyPower, SubjectPower,
)
from interfaces import Recording, SSVEPRecording


K = TypeVar("K", bound=Hashable)


class RawRecording(Generic[K]):
    """Concrete Recording[K] backed by an in-memory ndarray.

    Used by the default Experiment.conditions() / aggregate() paths to wrap
    stacked / reshaped raw data without depending on a paradigm-specific
    storage class. Implementors with their own typed views (e.g. ConditionView)
    can return their own type from take_channels.
    """

    def __init__(
        self,
        name: str,
        raw_data: RawStudyData,
        sample_rate: float,
        channel_names: List[str],
        props: K,
    ) -> None:
        self._name = name
        self._raw_data = raw_data
        self._sample_rate = sample_rate
        self._channel_names = list(channel_names)
        self._props = props

    def name(self) -> str:
        return self._name

    def raw_data(self) -> RawStudyData:
        return self._raw_data

    def sample_rate(self) -> float:
        return self._sample_rate

    def channel_names(self) -> List[str]:
        return self._channel_names

    def take_channels(self, names: List[str]) -> "RawRecording[K]":
        wanted = set(names)
        indices = [i for i, n in enumerate(self._channel_names) if n in wanted]
        return RawRecording(
            name=self._name,
            raw_data=self._raw_data[:, :, indices, :],
            sample_rate=self._sample_rate,
            channel_names=[self._channel_names[i] for i in indices],
            props=self._props,
        )

    def props(self) -> K:
        return self._props


def combine_subject_recordings(
    recordings: Sequence[Recording[K]],
) -> RawRecording[K]:
    """Stack per-subject recordings (each with S=1) along the subject axis.

    Truncates to the smallest common trial count so the stacked array is
    rectangular — matches the min_trials behavior in core.Study. Propagates
    props from the first recording (all per-subject recordings for one
    condition share the same props by construction).
    """
    if not recordings:
        raise ValueError("combine_subject_recordings: empty list")
    min_trials = min(r.raw_data().shape[1] for r in recordings)
    stacked = np.stack(
        [r.raw_data()[0, :min_trials] for r in recordings], axis=0,
    ).astype(np.float64)
    first = recordings[0]
    return RawRecording(
        name=f"Group({len(recordings)})",
        raw_data=stacked,
        sample_rate=first.sample_rate(),
        channel_names=first.channel_names(),
        props=first.props(),
    )


def flatten_subject_axis(rec: Recording[K]) -> RawRecording[K]:
    """Reshape (S, T, C, Time) → (1, S*T, C, Time). Subjects become trials."""
    data = rec.raw_data()
    s, t, c, time = data.shape
    flat = data.reshape(1, s * t, c, time).astype(np.float64)
    return RawRecording(
        name=f"Aggregate({rec.name()})",
        raw_data=flat,
        sample_rate=rec.sample_rate(),
        channel_names=rec.channel_names(),
        props=rec.props(),
    )


class Spectral:
    """Spectral analysis bound to one Recording.

    Caches the Fourier transform once on construction; every derivative
    re-uses it. Construct once per (recording, window setting).
    """

    def __init__(
        self,
        rec: Recording,
        *,
        window_duration_s: float = 2.0,
        n_neighbors: int = 3,
        n_skip: int = 1,
    ) -> None:
        self.rec = rec
        self.window_duration_s = window_duration_s
        self.n_neighbors = n_neighbors
        self.n_skip = n_skip
        self._window_size = int(window_duration_s * rec.sample_rate())
        self._fourier_cache: StudyData = self._compute_fourier()

    def _compute_fourier(self) -> StudyData:
        """Non-overlapping sliding-window kaiser-tapered rfft.

        Output (S, T, C, W, F) complex64. Was: StudyLoader._process_data.
        """
        data = self.rec.raw_data()  # (S, T, C, Time) float
        ws = self._window_size

        segments = np.lib.stride_tricks.sliding_window_view(data, ws, axis=-1)
        segments = segments[..., ::ws, :]
        segments = segments - np.mean(segments, axis=-1, keepdims=True)

        window = kaiser(ws, 4)
        return np.fft.rfft(segments * window, axis=-1).astype(np.complex64)

    @staticmethod
    def _into_snr(
        psd: np.ndarray, n_neighbors: int = 3, n_skip: int = 1,
    ) -> np.ndarray:
        kernel = np.concatenate((
            np.ones(n_neighbors),
            np.zeros(2 * n_skip + 1),
            np.ones(n_neighbors),
        ))
        kernel /= kernel.sum()

        mean_noise = np.apply_along_axis(
            lambda x: np.convolve(x, kernel, mode="valid"),
            axis=-1, arr=psd,
        )
        edge = n_neighbors + n_skip
        pad = [(0, 0)] * (mean_noise.ndim - 1) + [(edge, edge)]
        mean_noise = np.pad(mean_noise, pad_width=pad, constant_values=np.inf)
        # Edge bins are padded with inf so they yield SNR=0 by design. Interior
        # bins with all-zero noise neighborhoods (only happens on synthetic
        # zero-padded test inputs — real EEG PSD is never exactly zero) flow
        # through as nan/inf the same way; downstream argmax / snr_at_frequency
        # never queries those bins, so silence the warning rather than guarding.
        with np.errstate(divide="ignore", invalid="ignore"):
            return psd / mean_noise

    def frequencies(self) -> Array1D_f64:
        return np.fft.rfftfreq(
            self._window_size, d=1 / self.rec.sample_rate(),
        ).astype(np.float64)

    def power_per_trial(self) -> StudyPower:
        """Coherent averaging over windows, then squared magnitude. (S, T, C, F)."""
        return (np.abs(self._fourier_cache.mean(axis=-2)) ** 2).astype(np.float64)

    def snr_per_trial(self) -> StudyPower:
        return self._into_snr(
            self.power_per_trial(), self.n_neighbors, self.n_skip,
        )

    def power_spectrum(self) -> Tuple[Array1D_f64, Array1D_f64]:
        """(mean, sem) across S/T/C with proper per-subject SEM when S>1."""
        return self._mean_sem(self.power_per_trial())

    def snr_spectrum(self) -> Tuple[Array1D_f64, Array1D_f64]:
        return self._mean_sem(self.snr_per_trial())

    def snr_topomap(self) -> SubjectPower:
        """SNR averaged over Subject and Trial, channel dim retained: (C, F)."""
        return np.average(self.snr_per_trial(), axis=(0, 1))

    def snr_at(self, frequency: float) -> Tuple[float, float]:
        idx = self._closest_index(frequency)
        means, sems = self.snr_spectrum()
        return float(means[idx]), float(sems[idx])

    def power_at(self, frequency: float) -> Tuple[float, float]:
        idx = self._closest_index(frequency)
        means, sems = self.power_spectrum()
        return float(means[idx]), float(sems[idx])

    def phase_at(self, frequency: float) -> Tuple[ConditionPhases, float]:
        """Per-subject best-channel phasors at frequency. Returns (phasors, cycle_s)."""
        idx = self._closest_index(frequency)
        snr_target = self.snr_per_trial()[..., idx]      # (S, T, C)
        snr_avg_trial = snr_target.mean(axis=1)           # (S, C)
        best_channel = np.argmax(snr_avg_trial, axis=-1)  # (S,)

        if snr_avg_trial[0, best_channel] < 2:
            print("Analyzing improper trial, SNR is less then 2")

        fourier = self._fourier_cache  # (S, T, C, W, F)
        s_idx = np.arange(fourier.shape[0])
        # Index (S, T, W, F) by picking each subject's best channel:
        over_best = fourier[s_idx, :, best_channel]       # (S, T, W, F)
        signal = over_best[..., idx]                      # (S, T, W)
        return np.mean(signal, axis=-1).astype(np.complex64), 1.0 / float(frequency)

    # --- helpers ---
    def _closest_index(self, frequency: float) -> int:
        return int(np.argmin(np.abs(self.frequencies() - frequency)))

    @staticmethod
    def _mean_sem(arr: StudyPower) -> Tuple[Array1D_f64, Array1D_f64]:
        """(mean, sem) over the (S, T, C) axes for a (S, T, C, F) array.

        When S>1, the SEM is across subjects (first average T and C per
        subject, then SEM across the resulting per-subject vectors).
        """
        if arr.shape[0] > 1:
            subject_means = arr.mean(axis=(1, 2))   # (S, F)
            return (
                np.mean(subject_means, axis=0).astype(np.float64),
                sem(subject_means, axis=0).astype(np.float64),
            )
        flat = arr.reshape(-1, arr.shape[-1])
        return (
            np.average(flat, axis=0).astype(np.float64),
            sem(flat, axis=0).astype(np.float64),
        )


class SSVEPAnalysis(Spectral):
    """SSVEP-flavored analysis: target/carrier convenience + processing-time."""

    rec: SSVEPRecording

    def __init__(
        self,
        rec: SSVEPRecording,
        *,
        expected_latency_s: float = 0.05,
        window_duration_s: float = 2.0,
        n_neighbors: int = 3,
        n_skip: int = 1,
    ) -> None:
        super().__init__(
            rec,
            window_duration_s=window_duration_s,
            n_neighbors=n_neighbors,
            n_skip=n_skip,
        )
        self.expected_latency_s = expected_latency_s

    def _target_frequency(self) -> float:
        return float(self.rec.props().target_frequency)

    def _carrier_frequency(self) -> float:
        return float(self.rec.props().carrier_frequency)

    def snr_at_target(self) -> Tuple[float, float]:
        return self.snr_at(self._target_frequency())

    def snr_at_carrier(self) -> Tuple[float, float]:
        return self.snr_at(self._carrier_frequency())

    def power_at_target(self) -> Tuple[float, float]:
        return self.power_at(self._target_frequency())

    def power_at_carrier(self) -> Tuple[float, float]:
        return self.power_at(self._carrier_frequency())

    def processing_time_per_trial(self) -> Array1D_f64:
        """Per-trial latency in seconds, shape (S, T)."""
        c_target, T_target = self.phase_at(self._target_frequency())
        c_carrier, T_carrier = self.phase_at(self._carrier_frequency())
        distances = self._candidate_distances(
            c_target, c_carrier, T_target, T_carrier,
        )
        return self._resolve_to_seconds(
            distances, T_target, expected_s=self.expected_latency_s,
        )

    def processing_time_summary(
        self, r_min: float = 0.3,
    ) -> Tuple[Array1D_f64, Array1D_f64]:
        """Per-subject (mean_ms, sd_ms) circular summary.

        mean_ms: circular-mean latency (ms) after resolving the candidate-region
                 ambiguity once on the trial-aggregated phase.
        sd_ms:   circular sd of carrier-phase jitter, in ms.
                 r_min currently only kept for API parity; mask is intentionally
                 not applied here (see core.py history).
        """
        del r_min  # parity with old API; mask intentionally not applied
        f_carrier = self._carrier_frequency()

        # Data is up-to trial level (window is already averaged), T_* is the time in seconds.
        c_target, T_target = self.phase_at(self._target_frequency())
        c_carrier, T_carrier = self.phase_at(f_carrier)

        distances = self._candidate_distances(
            c_target, c_carrier, T_target, T_carrier,
        )

        chosen_distance_per_trial = self._choose_best_distances(
            distances, T_target, expected_s=self.expected_latency_s,
        )

        mean_distances = np.mean(chosen_distance_per_trial, axis=-1)
        
        sd_distances = circstd(np.angle(chosen_distance_per_trial), axis=-1)  / 2 / np.pi * T_target * 1000

        return (np.angle(mean_distances) * T_target / (2*np.pi) * 1000).astype(np.float64), sd_distances.astype(np.float64)

    @staticmethod
    def _candidate_distances(
        c_target, c_carrier, T_target: float, T_carrier: float,
    ) -> ConditionPhases:
        """Linear part: candidate complex unit-distances between phasors.

        Returns an array with a new last axis of length M = T_target / T_carrier.
        Each candidate's angle, scaled by T_target/(2π), is a possible latency.
        Units are according to the Target
        """
        assert T_target // T_carrier == T_target / T_carrier
        inflation_ratio = T_target // T_carrier
        norm_target = c_target / np.abs(c_target)
        norm_carrier = c_carrier / np.abs(c_carrier)
        adjusted_carrier = norm_carrier ** (1 / inflation_ratio)
        regions = np.exp(np.arange(inflation_ratio) * 2 * np.pi * 1j / inflation_ratio)
        possible_carriers = adjusted_carrier[..., np.newaxis] * regions
        return (norm_target[..., np.newaxis] / possible_carriers).astype(np.complex64)

    @staticmethod
    def _choose_best_distances(
        distances, T_target: float, expected_s: float = 0.05,
    ) -> Array1D_f64:
        """Non-linear (lossy) part: argmin selection by literature anchor."""
        assert T_target > expected_s
        expected_distance = np.exp((expected_s / T_target * 2 * np.pi) * 1j)
        best_idx = np.argmin(
            np.abs(np.angle(distances / expected_distance)),
            axis=-1, keepdims=True,
        )
        best_dt = np.take_along_axis(distances, best_idx, axis=-1).squeeze(-1)
        return best_dt
