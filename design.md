# SSVEP Analysis Design: Property-Based Study Registry

## 1. Core Data Structures

### A. ConditionProperties (NamedTuple)
The unique identifier for an experimental condition.
- `target_frequency: float`
- `carrier_frequency: float`
*Note: Being a NamedTuple, this is hashable and serves as the "key" for retrieving data views.*

### B. ConditionBlob
The heavy-duty container for a single experimental condition's data.
- **Data**: `np.ndarray` of `complex64`. 
- **Dimensions**: `(Subject, Trial, Electrode, Window, Frequency)`
- **Metadata**: 
    - `props: ConditionProperties`
    - `raw_info: mne.Info` (Essential for name-to-index electrode mapping)
    - `subject_names: List[str]` (Maps the local `Subject` dimension to global identities)

## 2. Data Ingestion: `StudyLoader`

A specialized utility responsible for traversing the file system and parsing raw data.
- **Responsibility**: Scans experiment folders, parses EDF files via MNE, performs `rfft` to produce `complex64` data.
- **Output**: Returns an `Iterator[Tuple[str, ConditionProperties, mne.Info, np.ndarray]]`.
    - `str`: Subject name.
    - `ConditionProperties`: Parsed from folder/filename.
    - `mne.Info`: Recording metadata.
    - `np.ndarray`: 4D array `(Trial, Electrode, Window, Frequency)`.

## 3. The Management Layer: `Study`

The `Study` class manages the lifecycle of the data blobs and identifies subjects across conditions.

### Initialization
- `__init__(data_stream: Iterator[Tuple[str, ConditionProperties, mne.Info, np.ndarray]])`:
    - Consumes the stream and organizes data into `ConditionBlob`s.
    - Correlates subjects across conditions to build the registry.

### API
- `subjects() -> Iterator[Subject]`: Returns an iterator of all unique subjects.
- `filter_subjects(requirements: Set[ConditionProperties]) -> Iterator[Subject]`: 
    - Returns an iterator of subjects who participated in *every* condition specified.
- `get_condition(props: ConditionProperties) -> ConditionView`:
    - Returns a `ConditionView` representing the aggregate of *all* subjects for that condition.
    - **Logic**: The original `Trial` dimension is averaged out, and the `Subject` dimension is promoted to be the new `Trial` dimension for the view (treating different subjects as independent trials).

## 4. The Access Layer: `Subject` & `ConditionView`

### Subject
A lightweight handle representing a participant.
- `name: str`
- **Accessors**:
    - `subject[props]` -> `ConditionView`: Direct access via `ConditionProperties`.
    - `conditions() -> Iterator[Tuple[ConditionProperties, ConditionView]]`: Iterator over available conditions.

### ConditionView (Implements `PowerSpectcraAnalyzable`)
A configured "lens" into a `ConditionBlob`.
- **State**:
    - Reference to the parent `ConditionBlob`.
    - `subject_slice: slice | int`: Identifies one or all subjects in the blob.
    - `electrode_indices: np.ndarray`: Current active electrodes.
- **Methods**:
    - `restrict_electrodes(names: List[str]) -> ConditionView`: Returns a NEW view with a subset of electrodes.
    - `as_power_spectrum()`: Computes magnitude squared on-the-fly.
    - `as_snr()`: Calculates SNR from the power spectrum view.

## 5. Performance & Memory
- **Precision**: `complex64` preserves phase integrity at half the memory cost of `complex128`.
- **Iterators**: Prevents massive list allocations during discovery and filtering.
- **View-Based**: No data is copied when restricting electrodes or selecting subjects; only indices and slices are updated.
