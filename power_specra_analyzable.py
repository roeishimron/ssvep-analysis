import numpy as np
from typing import Protocol, Tuple, runtime_checkable
from core_types import SubjectPower, Array1D_f64


@runtime_checkable
class PowerSpectcraAnalyzable(Protocol):
    def as_power_spectrum(self) -> Tuple[Array1D_f64, Array1D_f64]:
        """returns the average power and the std at each frequency"""
        ...

    def as_snr(self) -> SubjectPower:
        """returns the snr across electrodes (across space)"""
        ...

    def as_snr_average(self) -> Tuple[Array1D_f64, Array1D_f64]:
        """returns the average snr at each frequency and the std"""
        ...

    def name(self) -> str:
        """returns the name for plots"""
        ...

    def target_frequency(self) -> np.float64:
        """returns the target frequency"""
        ...

    def carrier_frequency(self) -> np.float64:
        """returns the carrier frequency"""
        ...

    def frequencies(self) -> Array1D_f64:
        """returns the array of frequencies"""
        ...

    def snr_at_frequency(self, target) -> Tuple[np.float64, np.float64]:
        """returns the (average, sem) snr at target frequency"""
        freqs = self.frequencies()
        idx = np.argmin(np.abs(freqs - target))
        means, sems = self.as_snr_average()
        return means[idx], sems[idx]

    def snr_at_target(self) -> Tuple[np.float64, np.float64]:
        """returns the (average, sem) snr at target frequency"""
        return self.snr_at_frequency(self.target_frequency())

    def snr_at_carrier(self) -> Tuple[np.float64, np.float64]:
        """returns the (average, sem) snr at carrier frequency"""
        return self.snr_at_frequency(self.carrier_frequency())

    def power_at_frequency(self, target) -> Tuple[np.float64, np.float64]:
        """returns the (average, sem) snr at target frequency"""
        freqs = self.frequencies()
        idx = np.argmin(np.abs(freqs - target))
        means, sems = self.as_power_spectrum()
        return means[idx], sems[idx]

    def power_at_target(self) -> Tuple[np.float64, np.float64]:
        """returns the (average, sem) snr at target frequency"""
        return self.power_at_frequency(self.target_frequency())

    def power_at_carrier(self) -> Tuple[np.float64, np.float64]:
        """returns the (average, sem) snr at carrier frequency"""
        return self.power_at_frequency(self.carrier_frequency())
