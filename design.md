# SSVEP Analysis Design: Property-Based Study Registry

## 1. Core Data Structures

### A. ConditionProperties (NamedTuple)
The unique identifier for an experimental condition.
- `target_frequency: np.float64`
- `carrier_frequency: np.float64`

### B. ConditionBlob
The primary data container for a single experimental condition.
- **Data**: `StudyData` (NumPy array of `complex64`). 
- **Dimensions**: `(Subject, Trial, Electrode, Window, Frequency)` (Strictly 5D)
- **Metadata**: 
    - `props: ConditionProperties`
    - `raw_info: mne.Info`
    - `subjects: List[Subject]`

## 2. Data Ingestion: `StudyLoader`

Responsible for filesystem traversal and initial signal processing.
- **FS Pattern**: Scans folders like `10hz_2ob_60s` and parses `.edf` files.
- **Processing**: Performs non-overlapping sliding window `rfft` to produce `complex64` components.
- **Output**: Returns an `Iterator[Tuple[str, ConditionProperties, mne.Info, SubjectData]]`.
    - `str`: Subject name.
    - `ConditionProperties`: Parsed from folder context.
    - `SubjectData`: 4D array `(Trial, Electrode, Window, Frequency)`.

## 3. The Registry: `Study`

The `Study` class manages the lifecycle of data and coordinates subject identities.

### Initialization
- `__init__(data_stream: Iterator, min_trials: int = 3)`: 
    - Consumes the stream and groups data by condition and subject.
    - Enforces a minimum trial count per subject; truncates extra trials to maintain consistent dimensions.
    - Assigns unique integer IDs to subjects.
    - Populates each `Subject` object's internal view registry.

### API
- `subjects() -> Iterator[Subject]`: Returns all unique subjects in the study (sorted alphabetically).
- `filter_subjects(requirements: Set[ConditionProperties]) -> Iterator[Subject]`: Returns subjects who participated in the intersection of provided conditions.
- `get_condition(props: ConditionProperties) -> ConditionView`: Returns an aggregate view where the Subject dimension is flattened into the Trial dimension `(1, S*T, E, W, F)`.

## 4. The Access Layer: `Subject` & `ConditionView`

### Subject
A lightweight handle. Identity is based on a unique `id`.
- `name: str`, `id: int`
- **Accessors**:
    - `subject[props]` -> `ConditionView`: Direct access to the subject's data for a condition.
    - `conditions()`: Iterator over all available conditions for this subject.

### ConditionView (Implements `PowerSpectcraAnalyzable`)
A configured "lens" into a `ConditionBlob`.
- **State**:
    - `blob`: Reference to the parent `ConditionBlob`.
    - `subject_idx: int | None`: Index of the specific subject (or `None` for all subjects).
    - `electrode_indices: Array1D_i64`: Subset of active electrodes.
- **Methods**:
    - `data`: Always returns a 5D view `(Subject, Trial, Electrode, Window, Frequency)`.
    - `_get_psd()`: Calculates PSD using coherent averaging **only across windows**. Returns `(S, T, E, F)`.
    - `as_power_spectrum()`: Returns `(mean, sem)` across S, T, and E dimensions.
    - `as_snr()`: Returns average SNR across S and T, keeping the Electrode dimension (spatial view).
    - `as_snr_average()`: Returns `(mean, sem)` SNR at each frequency across all non-frequency dimensions.

## 5. Implementation Notes
- **Strict Typing**: Uses explicit NumPy annotations like `np.ndarray[Shape, DType]` for dimension and datatype enforcement.
- **Coherent Averaging**: Fourier components are averaged in the complex domain **only across the window axis** before power calculation.
- **Trial Alignment**: All subjects within a condition are enforced to have the exact same number of trials at loading time.
- **Deterministic Alignment**: Subjects are sorted alphabetically during blob creation.
