import sys
from typing import NamedTuple, List, Dict, Iterator, Set, Tuple, Any, TypeVar, overload
import numpy as np
import mne
from scipy.stats import sem

# Defined interpretable shape types
# S=Subject, T=Trial, E=Electrode, W=Window, F=Frequency
STEWF = Tuple[int, int, int, int, int]
TEWF = Tuple[int, int, int, int]
STEF = Tuple[int, int, int, int]
TEF = Tuple[int, int, int]
SEF = Tuple[int, int, int]
EF = Tuple[int, int]

# Semantic Type Definitions for NumPy Arrays using standard ndarray subscription
# Format: [DType]_[Dimensions]

# Complex64 Arrays (Fourier Components)
StudyData = np.ndarray[STEWF, np.dtype[np.complex64]] 
SubjectData = np.ndarray[TEWF, np.dtype[np.complex64]]   

# Float64 Arrays (Power, SNR, Frequencies)
StudyPower = np.ndarray[STEF, np.dtype[np.float64]]           # (S, T, E, F)
SubjectPower = np.ndarray[EF, np.dtype[np.float64]]           # (E, F)

# Indexing / Identification Arrays
Array1D_f64 = np.ndarray[Tuple[int], np.dtype[np.float64]]             
Array1D_i64 = np.ndarray[Tuple[int], np.dtype[np.int64]]        

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

class ConditionProperties(NamedTuple):
    """
    Unique identifier for an experimental condition.
    """
    target_frequency: np.float64
    carrier_frequency: np.float64

    def __repr__(self) -> str:
        return f"ConditionProperties(target={self.target_frequency}Hz, carrier={self.carrier_frequency}Hz)"

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

class ConditionView:
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
        flat_snrs = snrs.reshape(-1, snrs.shape[-1])
        return np.average(flat_snrs, axis=0), sem(flat_snrs, axis=0)

    def name(self) -> str:
        return self._name

    def target_frequency(self) -> float:
        return self.blob.props.target_frequency

    def carrier_frequency(self) -> float:
        return self.blob.props.carrier_frequency

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

    def filter_subjects(self, requirements: Set[ConditionProperties]) -> Iterator[Subject]:
        for subject in self.subjects():
            participated_in_all = True
            for req in requirements:
                if req not in subject._views:
                    participated_in_all = False
                    break
            if participated_in_all:
                yield subject

    def get_condition(self, props: ConditionProperties) -> ConditionView:
        blob = self._blobs.get(props)
        if blob is None:
            raise KeyError(f"Condition {props} not found in study")
            
        # Reshape to flatten Subject dimension into Trial dimension
        # blob.data: (S, T, E, W, F)
        s, t, e, w, f = blob.data.shape
        
        # Reshape to (1, S*T, E, W, F)
        # We merge S and T into a single "Trial" dimension
        # User explicitly asked to merge S into T so there's no difference.
        aggregate_data: StudyData = blob.data.reshape(1, s*t, e, w, f).astype(np.complex64)
        
        new_blob = ConditionBlob(
            aggregate_data, 
            blob.props, 
            blob.raw_info, 
            [Subject(-1, "Aggregate")]
        )
        return ConditionView(new_blob, name=f"Aggregate({props.target_frequency}Hz)")