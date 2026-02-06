import numpy as np
from typing import Protocol, Any, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from core import SubjectPower, Array1D_f64

class PowerSpectcraAnalyzable(Protocol):
    def as_power_spectrum(self) -> Tuple['Array1D_f64', 'Array1D_f64']:
        """returns the average power and the std at each frequency"""
        ...

    def as_snr(self) -> 'SubjectPower':
        """returns the snr across electrodes (across space)"""
        ...

    def as_snr_average(self) -> Tuple['Array1D_f64', 'Array1D_f64']:
        """returns the average snr at each frequency and the std"""
        ...

    def name(self) -> str:
        """returns the name for plots"""
        ...

    def target_frequency(self) -> float:
        """returns the target frequency"""
        ...

    def carrier_frequency(self) -> float:
        """returns the carrier frequency"""
        ...

    def frequencies(self) -> 'Array1D_f64':
        """returns the array of frequencies"""
        ...
