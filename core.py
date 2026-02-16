import sys
from typing import List, Dict, Iterator, Set, Tuple, overload
import numpy as np
import mne
from scipy.stats import sem
from power_specra_analyzable import PowerSpectcraAnalyzable
from core_types import (
    ConditionProperties, StudyData, SubjectData, StudyPower, 
    SubjectPower, Array1D_f64, Array1D_i64, ConditionPhases
)

@overload
def into_SNR(psd: StudyPower, n_neighbors: int = 3, n_skip: int = 1) -> StudyPower: ...

@overload
def into_SNR(psd: SubjectPower, n_neighbors: int = 3, n_skip: int = 1) -> SubjectPower: ...

def into_SNR(psd: np.ndarray, n_neighbors: int = 3, n_skip: int = 1) -> np.ndarray:
    """
    Calculates SNR from power spectrum using neighbor-based noise estimation.
    """
    kernel = np.concatenate((np.ones(n_neighbors), np.zeros(2*n_skip+1), np.ones(n_neighbors)))
    kernel /= kernel.sum()
    
    mean_noise = np.apply_along_axis(
        lambda x: np.convolve(x, kernel, mode="valid"), axis=-1, arr=psd
    )
    
    edge_width = n_neighbors + n_skip
    pad_width = [(0, 0)] * (mean_noise.ndim - 1) + [(edge_width, edge_width)]
    mean_noise = np.pad(mean_noise, pad_width=pad_width, constant_values=np.inf)
    
    return psd / mean_noise

class ConditionBlob:
    """
    Heavy-duty container for a single experimental condition's data.
    """
    def __init__(
        self,
        data: StudyData,
        props: ConditionProperties,
        raw_info: mne.Info,
        subjects: List['Subject']
    ):
        if data.ndim != 5:
            raise ValueError(f"Data must be 5D (Subject, Trial, Electrode, Window, Frequency), got {data.ndim}D")
        
        if data.shape[0] != len(subjects):
            raise ValueError(f"Subject dimension size ({data.shape[0]}) must match number of subjects ({len(subjects)})")
            
        self.data: StudyData = data.astype(np.complex64)
        self.props = props
        self.raw_info = raw_info
        self.subjects = subjects

    @property
    def n_subjects(self) -> int:
        return self.data.shape[0]

    @property
    def n_trials(self) -> int:
        return self.data.shape[1]

    @property
    def n_electrodes(self) -> int:
        return self.data.shape[2]

    @property
    def n_windows(self) -> int:
        return self.data.shape[3]

    @property
    def n_frequencies(self) -> int:
        return self.data.shape[4]

class ConditionView(PowerSpectcraAnalyzable):
    """
    A configured "lens" into a ConditionBlob.
    Implements PowerSpectcraAnalyzable.
    """
    def __init__(
        self,
        blob: ConditionBlob,
        subject_idx: int | None = None,
        electrode_indices: Array1D_i64 | None = None,
        name: str | None = None
    ):
        self.blob = blob
        self.subject_idx = subject_idx
        if electrode_indices is None:
            self.electrode_indices: Array1D_i64 = np.arange(blob.n_electrodes, dtype=np.int64)
        else:
            self.electrode_indices: Array1D_i64 = electrode_indices
        
        if name:
            self._name = name
        elif subject_idx is not None:
            self._name = blob.subjects[subject_idx].name
        else:
            # All subjects
            if len(blob.subjects) == 1:
                self._name = blob.subjects[0].name
            else:
                self._name = f"Group({len(blob.subjects)})"

    @property
    def data(self) -> StudyData:
        # Always returns 5D: (Subject, Trial, Electrode, Window, Frequency)
        if self.subject_idx is None:
            return self.blob.data[:, :, self.electrode_indices, :, :]
        # indexing with [idx:idx+1] keeps the dimension
        return self.blob.data[self.subject_idx : self.subject_idx + 1, :, self.electrode_indices, :, :]

    def _get_psd(self) -> StudyPower:
        """
        Calculates PSD using coherent averaging only over windows:
        ALWAYS first average the fourier components over windows and only then take their absolute value^2.
        Returns: (Subject, Trial, Electrode, Frequency)
        """
        # data: (S, T, E, W, F)
        # Average over Windows axis (-2) ONLY
        avg_window = np.mean(self.data, axis=-2)
        return np.abs(avg_window)**2

    def restrict_electrodes(self, names: List[str]) -> 'ConditionView':
        all_names = self.blob.raw_info.ch_names
        indices = np.array([all_names.index(n) for n in names if n in all_names], dtype=np.int64)
        new_indices: Array1D_i64 = np.intersect1d(self.electrode_indices, indices)
        return ConditionView(self.blob, self.subject_idx, new_indices, name=self._name)

    def as_power_spectrum(self) -> Tuple[Array1D_f64, Array1D_f64]:
        # Returns (average, sem) across Subject, Trial and Electrode dimensions
        ps: StudyPower = self._get_psd() 
        
        if ps.shape[0] > 1:
            # SEM across subjects: average over trials/electrodes per subject, then SEM over subjects
            subject_means = np.mean(ps, axis=(1, 2)) # (S, F)
            return np.mean(subject_means, axis=0), sem(subject_means, axis=0)
        else:
            # Flatten all non-frequency dimensions: (S, T, E, F) -> (-1, F)
            flat_ps = ps.reshape(-1, ps.shape[-1])
            return np.average(flat_ps, axis=0), sem(flat_ps, axis=0)

    def as_snr(self) -> SubjectPower:
        # returns the average snr across subjects and trials, but keeps Electrode dimension (spatial)
        ps: StudyPower = self._get_psd()
        snrs: StudyPower = into_SNR(ps)
        # Average over Subject (0) and Trial (1) dimensions
        return np.average(snrs, axis=(0, 1))

    def as_snr_average(self) -> Tuple[Array1D_f64, Array1D_f64]:
        # returns (average, sem) at each frequency across Subjects, Trials and Electrodes
        ps: StudyPower = self._get_psd()
        snrs: StudyPower = into_SNR(ps)
        
        if snrs.shape[0] > 1:
            # SEM across subjects: average over trials/electrodes per subject, then SEM over subjects
            subject_means = np.mean(snrs, axis=(1, 2)) # (S, F)
            print(f"taking the sem of {subject_means.shape[0]} subjects")
            return np.mean(subject_means, axis=0), sem(subject_means, axis=0)
        else:
            flat_snrs = snrs.reshape(-1, snrs.shape[-1])
            return np.average(flat_snrs, axis=0), sem(flat_snrs, axis=0)
        
    def as_phase(self, frequency: np.float64) -> Tuple[ConditionPhases, np.float64]:
        """Returns the phases of the subjects at each trial coupeled with the cycle duration (seconds)"""
        frequency_index = np.argmin(np.abs(self.frequencies() - frequency))
        
        # Calculate SNR on the full PSD first to ensure correct neighbor convolution
        psd_all = self._get_psd() # (S, T, E, F)
        snrs_all = into_SNR(psd_all) # (S, T, E, F)
        
        # Select target frequency SNR and average over trials
        snrs_target = snrs_all[..., frequency_index] # (S, T, E)
        snrs_avg_trial = snrs_target.mean(axis=1) # (S, E)
        
        target_electrode_per_subject = np.argmax(snrs_avg_trial, axis=-1) # (S,)
        
        over_maximum_electrode = self.data[np.arange(self.data.shape[0]),:, target_electrode_per_subject]
        
        # Select frequency -> (S, T, W)
        signal_components = over_maximum_electrode[..., frequency_index]
        
        # Average over Windows -> (S, T)
        return np.mean(signal_components, axis=-1), 1/frequency

    def name(self) -> str:
        return self._name

    def target_frequency(self) -> np.float64:
        return self.blob.props.target_frequency
        

    def carrier_frequency(self) -> np.float64:
        return self.blob.props.carrier_frequency

    def calculate_processing_time(self) -> np.ndarray:
        """
        Calculates the time difference (latency) between the carrier frequency response 
        and the target frequency response.
        
        The method accounts for the phase ambiguity arising from different cycle durations.
        If the carrier frequency is M times the target frequency, there are M possible 
        time differences within one target cycle. The method selects the one closest 
        to 50ms (0.05s), based on literature for typical neural latencies.
        
        Returns:
            np.ndarray: A 2D array of shape (Subject, Trial) containing the calculated 
                        time differences in seconds.
        """
        f_target = self.target_frequency()
        f_carrier = self.carrier_frequency()
        
        # Get phases and cycle durations
        c_target, T_target = self.as_phase(f_target)
        c_carrier, T_carrier = self.as_phase(f_carrier)

        assert T_target // T_carrier == T_target / T_carrier
        inflation_ratio = T_target // T_carrier
        normalized_target, normalized_carrier = c_target / np.abs(c_target), c_carrier / np.abs(c_carrier)
        adjusted_carrier = normalized_carrier ** (1/inflation_ratio) # its phase should be lowered down
        regions = np.exp(np.arange(inflation_ratio)*2*np.pi*1j/inflation_ratio)
        possible_carriers = adjusted_carrier[..., np.newaxis] * regions
        distances = normalized_target[..., np.newaxis] / possible_carriers

        assert T_target > 0.05
        expected_distance = np.exp((0.05/T_target*2*np.pi)*1j)

        best_distances = np.argmin(np.abs(np.angle(distances / expected_distance)), -1)
        
        best_dts = np.take_along_axis(distances, best_distances[..., np.newaxis], axis=-1).squeeze(axis=-1)
        
        return np.angle(best_dts)/2/np.pi * T_target

    def frequencies(self) -> Array1D_f64:
        sfreq = self.blob.raw_info['sfreq']
        n_points = self.data.shape[-1]
        window_size = (n_points - 1) * 2
        return np.fft.rfftfreq(window_size, d=1/sfreq)

class Subject:
    """
    A lightweight handle representing a participant.
    Identity is defined by a unique ID.
    Knows its own views (blobs).
    """
    def __init__(self, subject_id: int, name: str):
        self._id = subject_id
        self._name = name
        self._views: Dict[ConditionProperties, 'ConditionView'] = {}

    @property
    def id(self) -> int:
        return self._id

    @property
    def name(self) -> str:
        return self._name

    def __getitem__(self, props: ConditionProperties) -> 'ConditionView':
        view = self._views.get(props)
        if view is None:
            raise KeyError(f"Subject {self.name} (id={self.id}) did not participate in condition {props}")
        return view

    def _register_view(self, props: ConditionProperties, view: 'ConditionView'):
        self._views[props] = view

    def conditions(self) -> Iterator[Tuple[ConditionProperties, 'ConditionView']]:
        yield from self._views.items()

    def __repr__(self) -> str:
        return f"Subject({self._name}, id={self._id})"

    def __hash__(self) -> int:
        return hash(self._id)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Subject):
            return False
        return self._id == other._id

class Study:
    """
    Manages the lifecycle of the data blobs and identifies subjects across conditions.
    Enforces consistent trial counts within conditions.
    """
    def __init__(self, data_stream: Iterator[Tuple[str, ConditionProperties, mne.Info, SubjectData]], min_trials: int = 3):
        self._subjects_by_name: Dict[str, Subject] = {}
        self._next_id = 0
        
        # Buffer to collect data: props -> name -> List[data]
        buffer: Dict[ConditionProperties, Dict[str, List[np.ndarray]]] = {}
        infos: Dict[ConditionProperties, mne.Info] = {}

        for name, props, info, data in data_stream:
            if name not in self._subjects_by_name:
                self._subjects_by_name[name] = Subject(self._next_id, name)
                self._next_id += 1
            
            if props not in buffer:
                buffer[props] = {}
                infos[props] = info
            
            if name not in buffer[props]:
                buffer[props][name] = []
            buffer[props][name].append(data)

        self._blobs: Dict[ConditionProperties, ConditionBlob] = {}
        for props, subject_data_map in buffer.items():
            # Sort by name to ensure deterministic alignment in the blob
            sorted_names = sorted(subject_data_map.keys())
            subject_handles = [self._subjects_by_name[name] for name in sorted_names]
            
            # Check trial consistency and enforce min_trials
            subject_trial_data: List[np.ndarray] = []
            
            for name in sorted_names:
                # Concatenate all files for this subject in this condition
                combined_subj_data = np.concatenate(subject_data_map[name], axis=0)
                n_trials = combined_subj_data.shape[0]
                
                if n_trials < min_trials:
                    print(f"\nERROR: Subject '{name}' has only {n_trials} trials for condition {props}, which is less than the minimum required ({min_trials}).")
                    sys.exit(1)
                
                # Keep only the first min_trials to ensure consistent dimensions
                subject_trial_data.append(combined_subj_data[:min_trials])

            # Stack subjects: (Subject, Trial, Electrode, Window, Frequency)
            blob_data: StudyData = np.stack(subject_trial_data, axis=0).astype(np.complex64)
            
            blob = ConditionBlob(blob_data, props, infos[props], subject_handles)
            self._blobs[props] = blob
            
            # Register views in subjects
            for i, subj in enumerate(subject_handles):
                view = ConditionView(blob, subject_idx=i)
                subj._register_view(props, view)

    def subjects(self) -> Iterator[Subject]:
        # Return subjects sorted by name for consistency
        for name in sorted(self._subjects_by_name.keys()):
            yield self._subjects_by_name[name]

    def conditions(self) -> Iterator[ConditionProperties]:
        # Return conditions sorted by target frequency, then carrier frequency
        for props in sorted(self._blobs.keys(), key=lambda p: (p.target_frequency, p.carrier_frequency)):
            yield props

    def filter_subjects(self, requirements: Set[ConditionProperties]) -> Iterator[Subject]:
        for subject in self.subjects():
            participated_in_all = True
            for req in requirements:
                if req not in subject._views:
                    participated_in_all = False
                    break
            if participated_in_all:
                yield subject

    def get_group_view(self, props: ConditionProperties) -> ConditionView:
        blob = self._blobs.get(props)
        if blob is None:
            raise KeyError(f"Condition {props} not found in study")
        return ConditionView(blob)

    def get_condition(self, props: ConditionProperties) -> ConditionView:
        blob = self._blobs.get(props)
        if blob is None:
            raise KeyError(f"Condition {props} not found in study")
            
        # Reshape to flatten Subject dimension into Trial dimension
        # blob.data: (S, T, E, W, F)
        s, t, e, w, f = blob.data.shape
        
        # Reshape to (1, S*T, E, W, F)
        aggregate_data: StudyData = blob.data.reshape(1, s*t, e, w, f).astype(np.complex64)
        
        new_blob = ConditionBlob(
            aggregate_data, 
            blob.props, 
            blob.raw_info, 
            [Subject(-1, "Aggregate")]
        )
        return ConditionView(new_blob, name=f"Aggregate({props.target_frequency}Hz)")
