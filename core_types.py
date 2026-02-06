from typing import NamedTuple, Tuple
import numpy as np

# Defined interpretable shape types
# S=Subject, T=Trial, E=Electrode, W=Window, F=Frequency
STEWF = Tuple[int, int, int, int, int]
TEWF = Tuple[int, int, int, int]
STEF = Tuple[int, int, int, int]
TEF = Tuple[int, int, int]
SEF = Tuple[int, int, int]
EF = Tuple[int, int]

# Semantic Type Definitions for NumPy Arrays
StudyData = np.ndarray[STEWF, np.dtype[np.complex64]] 
SubjectData = np.ndarray[TEWF, np.dtype[np.complex64]]   

StudyPower = np.ndarray[STEF, np.dtype[np.float64]]           # (S, T, E, F)
SubjectPower = np.ndarray[EF, np.dtype[np.float64]]           # (E, F)

Array1D_f64 = np.ndarray[Tuple[int], np.dtype[np.float64]]             
Array1D_i64 = np.ndarray[Tuple[int], np.dtype[np.int64]]        

class ConditionProperties(NamedTuple):
    """
    Unique identifier for an experimental condition.
    """
    target_frequency: np.float64
    carrier_frequency: np.float64

    def __repr__(self) -> str:
        return f"ConditionProperties(target={self.target_frequency}Hz, carrier={self.carrier_frequency}Hz)"
