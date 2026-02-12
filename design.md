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

## 6. Analysis & Visualization

### A. Single-Subject/Aggregate Spectrum (`analyze_spectrum`)
Plots the frequency-domain characteristics of a `PowerSpectcraAnalyzable` object.
- **SNR Spectrum**: Visualizes the Signal-to-Noise Ratio across frequencies.
- **Amplitude Spectrum**: Visualizes power in log-scale (dB).
- **Statistics**: Displays Mean ± STD (or SEM) across the aggregated dimensions (e.g., trials/electrodes).

### B. Spatial Distribution (`plot_snrs`)
Visualizes the topographical distribution of SNR.
- **Harmonics**: Automatically calculates and plots SNR topomaps for the target frequency and its harmonics (1st through 8th).
- **Max SNR Tracking**: Identifies and displays the global maximum SNR across all plotted harmonics for scaling.

### C. Condition-Wise Comparison (`plot_snr_comparison`)
Compares performance metrics across all conditions in a `Study`.
- **Metrics**: Plots both SNR and Absolute Power at the target frequency.
- **Group Statistics**: Aggregates data across all subjects in each condition, displaying Mean ± SEM.
- **Baseline**: Includes a "Noise Floor" reference line at SNR=1.

### D. Subject-Wise Comparison (`CarrierComparisonAnalysis`)
Specialized class for tracking individual subject performance across a trajectory of conditions.
- **Goal**: Subject-wise comparison of SNR at carrier frequencies (or target frequencies) across multiple conditions.
- **Data Extraction**:
    - Uses `Study.filter_subjects` to identify the intersection of participants who completed all requested conditions.
    - `_get_comparison_data()`: Returns an iterator of `(Subject Name, List[SNR])` tuples.
- **Analysis**:
    - `snr_slopes(data)`: Calculates the linear regression slope of SNR vs. Carrier Frequency for each subject.
- **Visualization**:
    - `plot()`: Generates a slope plot (multi-line plot) where each line represents a single subject with SNR values labeled at each point.
    - `plot_slopes_distribution()`: Displays a histogram of the calculated SNR slopes across the subject population.
