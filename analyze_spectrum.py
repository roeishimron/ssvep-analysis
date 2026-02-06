from typing import Any, Tuple
from matplotlib import pyplot as plt
import mne
from power_specra_analyzable import PowerSpectcraAnalyzable
import numpy as np
from core_types import SubjectPower, Array1D_f64
from core import Study

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

def plot_snr_comparison(study: Study, electrode_names: List[str]):
    labels = []
    snr_means = []
    snr_sems = []
    psd_means = []
    psd_sems = []
    
    for props in study.conditions():
        view = study.get_group_view(props)
        # Restrict to the specific electrode
        elec_view = view.restrict_electrodes(electrode_names)
        
        # Get SNR at target
        snr_m, snr_s = elec_view.snr_at_target()
        snr_means.append(snr_m)
        snr_sems.append(snr_s)
        
        # Get PSD at target
        psd_m, psd_s = elec_view.power_at_target()
        psd_means.append(psd_m)
        psd_sems.append(psd_s)
        
        labels.append(f"{props.carrier_frequency}Hz\n({props.target_frequency:.1f}Hz)")
        
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True, label=f"comparison-{electrode_names}")
    x = np.arange(len(labels))
    
    # SNR Plot
    ax1.bar(x, snr_means, yerr=snr_sems, capsize=5, color='skyblue', edgecolor='navy')
    ax1.set_ylabel("SNR [P/N]")
    ax1.set_title(f"SNR at Target Frequency over {electrode_names}")
    ax1.grid(axis='y', linestyle='--', alpha=0.7)
    ax1.axhline(1, color='red', linestyle='--', alpha=0.5, label="Noise Floor")
    ax1.legend()
    
    # PSD Plot
    ax2.bar(x, psd_means, yerr=psd_sems, capsize=5, color='salmon', edgecolor='darkred')
    ax2.set_ylabel("Power [microV^2]")
    ax2.set_xlabel("Condition (Carrier / Target)")
    ax2.set_title(f"Power at Target Frequency over {electrode_names}")
    ax2.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.xticks(x, labels)
    fig.suptitle(f"Condition Comparison over {electrode_names}\n(Mean ± SEM across subjects)")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])