import numpy as np
from typing import Protocol, Any, Tuple
from scipy.stats import sem

def into_SNR(psd, noise_n_neighbor_freqs=1, noise_skip_neighbor_freqs=1):

    # Construct a kernel that calculates the mean of the neighboring
    # frequencies
    averaging_kernel = np.concatenate(
        (
            np.ones(noise_n_neighbor_freqs),
            np.zeros(2 * noise_skip_neighbor_freqs + 1),
            np.ones(noise_n_neighbor_freqs),
        )
    )
    averaging_kernel /= averaging_kernel.sum()

    # Calculate the mean of the neighboring frequencies by convolving with the
    # averaging kernel.
    # applying along each electrode
    mean_noise = np.apply_along_axis(
        lambda psd_: np.convolve(psd_, averaging_kernel, mode="valid"), axis=-1, arr=psd
    )

    # The mean is not defined on the edges so we will pad it with nas. The
    # padding needs to be done for the last dimension only so we set it to
    # (0, 0) for the other ones.
    edge_width = noise_n_neighbor_freqs + noise_skip_neighbor_freqs
    pad_width = [(0, 0)] * (mean_noise.ndim - 1) + [(edge_width, edge_width)]
    mean_noise = np.pad(mean_noise, pad_width=pad_width,
                        constant_values=np.inf)

    return psd / mean_noise


NOISE_NEIGHBORS = 3
NOISE_SKIP = 1


class PowerSpectcraAnalyzable(Protocol):
    def as_power_spectrum(self) -> Tuple[np.ndarray[Tuple[int], np.dtype[np.float64]],
                                         np.ndarray[Tuple[int], np.dtype[np.float64]]]:
        """returns the average power and the std at each frequency"""
        ...

    def as_snr(self) -> np.ndarray[Tuple[int, int], np.dtype[np.float64]]:
        """returns the snr across electrodes (across space)"""
        ...

    def as_snr_average(self) -> Tuple[np.ndarray[Tuple[int], np.dtype[np.float64]],
                                      np.ndarray[Tuple[int], np.dtype[np.float64]]]:
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

    def frequencies(self) -> np.ndarray[Tuple[int], np.dtype[np.float64]]:
        """returns the array of frequencies"""
        ...


class Condition:
    def __init__(self, amplitudes: np.ndarray[Tuple[int, int, int], np.dtype[np.float64]],
                 frequencies: np.ndarray[Tuple[int], np.dtype[np.float64]],
                 name: str, target_frequency: np.float64, carrier_frequency: np.float64) -> None:
        """Dimentions for amplitudes are [trial, electrode, frequency]"""
        self.amplitudes = amplitudes
        self._frequencies = frequencies
        self._name = name
        self.snrs = into_SNR(self.amplitudes, noise_n_neighbor_freqs=NOISE_NEIGHBORS,
                             noise_skip_neighbor_freqs=NOISE_SKIP)
        self._target_frequency = target_frequency
        self._carrier_frequency = carrier_frequency

    def as_power_spectrum(self) -> Tuple[np.ndarray[Tuple[int], np.dtype[np.float64]],
                                         np.ndarray[Tuple[int], np.dtype[np.float64]]]:
        # reshaping for sem:
        new_shape = (np.prod(self.amplitudes.shape[:-1]), self.amplitudes.shape[-1])
        amps = np.reshape(self.amplitudes, new_shape)
        return np.average(amps, 0), sem(amps)

    def as_snr(self) -> np.ndarray[Tuple[int, int], np.dtype[np.float64]]:
        return np.average(self.snrs, 0)

    def as_snr_average(self) -> Tuple[np.ndarray[Tuple[int], np.dtype[np.float64]],
                                      np.ndarray[Tuple[int], np.dtype[np.float64]]]:
        new_shape = (np.prod(self.snrs.shape[:-1]), self.snrs.shape[-1])
        snrs = np.reshape(self.snrs, new_shape)
        return np.average(snrs, 0), sem(snrs)

    def name(self) -> str:
        return self._name

    def target_frequency(self) -> np.float64:
        return self._target_frequency

    def carrier_frequency(self) -> np.float64:
        return self._carrier_frequency

    def frequencies(self) -> np.ndarray[Tuple[int], np.dtype[np.float64]]:
        return self._frequencies


class AllConditionSubjects(Condition):
    def __init__(self, amplitudes: np.ndarray[Tuple[int, int, int, int], np.dtype[np.float64]],
                 frequencies: np.ndarray[Tuple[int], np.dtype[np.float64]],
                 name: str, target_frequency: np.float64, carrier_frequency: np.float64) -> None:
        """Dimentions for amplitudes are [trial, electrode, frequency]"""
        super().__init__(np.average(amplitudes, 1), frequencies,
                         name, target_frequency, carrier_frequency)
