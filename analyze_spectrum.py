from typing import Any, Tuple
from matplotlib import pyplot as plt
import mne
from power_specra_analyzable import PowerSpectcraAnalyzable
import numpy as np
from core import SubjectPower, Array1D_f64

def analyze_spectrum(subject: PowerSpectcraAnalyzable, fmin: float, fmax: float):

    print("got snr")
    title = f"{subject.name()} (Target: {subject.target_frequency()}Hz, Carrier: {subject.carrier_frequency()}Hz)"
    fig, axes = plt.subplots(2, 1, sharex="all", sharey="none", figsize=(
        8, 6), label=f"{subject.name()}-spectrum")
    fig.suptitle(title)

    freqs: Array1D_f64 = subject.frequencies()
    snr_mean, snr_std = subject.as_snr_average() # Float64_F, Float64_F
    # SNR spectrum

    axes[0].plot(freqs, snr_mean)
    axes[0].fill_between(
        freqs, snr_mean - snr_std, snr_mean + snr_std, color="r", alpha=0.1
    )
    axes[0].set(
        title="SNR spectrum",
        ylabel="SNR [P/N]",
        xlim=[fmin, fmax],
    )

    # Amplitudes spectrum
    amplitudes_mean, amplitudes_std = subject.as_power_spectrum() # Float64_F, Float64_F

    axes[1].plot(freqs, 20 * np.log10(amplitudes_mean))
    axes[1].fill_between(
        freqs, 20 * np.log10(amplitudes_mean - amplitudes_std),
        20 * np.log10(amplitudes_mean + amplitudes_std),
        color="r", alpha=0.1
    )
    axes[1].set(
        title="Amplitudes spectrum",
        xlabel="Frequency [Hz]",
        ylabel="microV (logscale)",
        xlim=[fmin, fmax],
    )


def _into_channel_average(freqs: Array1D_f64,
                          snrs: SubjectPower,
                          target_freq: float) -> Tuple[float, np.ndarray]:
    target_center = int(np.argmin(np.abs(freqs - target_freq)))

    # get average SNR at target Hz for ALL channels
    snrs_stim_hz = snrs[:, target_center]
    return (float(freqs[target_center]), snrs_stim_hz)


def plot_snrs(subject: PowerSpectcraAnalyzable, raw_mne_info: Any):

    TOPO_WIDTH = 4
    TOPO_HEIGHT = 2

    target_freq = subject.target_frequency()
    carrier_freq = subject.carrier_frequency()
    freqs: Array1D_f64 = subject.frequencies()
    snr: SubjectPower = subject.as_snr()

    HARMONEY_FREQS = np.arange(TOPO_WIDTH * TOPO_HEIGHT)+1

    TARGET_FREQS = target_freq*HARMONEY_FREQS

    freqs_with_channel_averages = [
        _into_channel_average(freqs, snr, f) for f in TARGET_FREQS]

    upper_limit = np.max(
        np.array([t[1] for t in freqs_with_channel_averages]).flatten())
    # plot SNR topography
    fig, axs = plt.subplots(TOPO_HEIGHT, TOPO_WIDTH,  sharex="none",
                            sharey="none", label=f"{subject.name()}-topomap")
    fig.suptitle(f"{subject.name()} Topography (Target: {target_freq}Hz, Carrier: {carrier_freq}Hz)\nMax SNR: {upper_limit:.2f}")
    
    for ((freq, channel_average), ax) in zip(freqs_with_channel_averages, axs.flatten()):

        print(f"looking at freq {freq:.2f}")
        ax.set_title(f"SNR at F*{freq/target_freq:.0f} ({freq:.2f})")

        mne.viz.plot_topomap(channel_average, raw_mne_info,
                             vlim=(1, upper_limit), axes=ax, show=False)
