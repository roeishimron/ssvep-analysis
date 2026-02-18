from typing import Any, Tuple, List, Iterator
from matplotlib import pyplot as plt
import mne
from scipy.stats import pearsonr
from power_specra_analyzable import PowerSpectcraAnalyzable
import numpy as np
from core_types import SubjectPower, Array1D_f64, ConditionProperties
from core import Study

def analyze_spectrum(subject: PowerSpectcraAnalyzable, fmin: float, fmax: float, show_psd: bool):

    print("got snr")
    title = f"Target: {subject.target_frequency()}Hz, Carrier: {subject.carrier_frequency()}Hz"

    fig, axes = plt.subplots(1+show_psd, 1, sharex="all", sharey="none", figsize=(
        8, 6), label=f"{subject.name()}-{subject.carrier_frequency()}-spectrum")
    fig.suptitle(title)
    freqs: Array1D_f64 = subject.frequencies()
    snr_mean, snr_std = subject.as_snr_average() # Float64_F, Float64_F
    # SNR spectrum

    if not show_psd:
        axes = [axes]

    axes[0].plot(freqs, snr_mean)
    axes[0].fill_between(
        freqs, snr_mean - snr_std, snr_mean + snr_std, color="r", alpha=0.1
    )
    axes[0].set(
        title="SNR",
        ylabel="SNR [P/N]",
        xlim=[fmin, fmax],
        ylim=[0, np.max((snr_mean+snr_std)[freqs<=fmax])]
    )

    if show_psd:
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
    TOPO_HEIGHT = 1

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
                            sharey="none", label=f"{subject.name()}-{carrier_freq}-topomap")
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
        
    fig, ax = plt.subplots(1, 1, figsize=(10, 10), sharex=True, label=f"comparison-{electrode_names}")
    x = np.arange(len(labels))
    
    # SNR Plot
    ax.bar(x, snr_means, yerr=snr_sems, capsize=5, color='skyblue', edgecolor='navy')
    ax.set_ylabel("SNR [P/N]")
    ax.set_title(f"SNR at Target Frequency over {electrode_names}")
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.axhline(1, color='red', linestyle='--', alpha=0.5, label="Noise Floor")
    ax.legend()
    
    plt.xticks(x, labels)
    fig.suptitle(f"Condition Comparison over {electrode_names}\n(Mean ± SEM across subjects)")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

class CarrierComparisonAnalysis:
    """
    Analyzes and plots SNR at carrier frequencies for subjects who participated in specific carrier conditions.
    """
    def __init__(self, study: Study, carriers: List[float], electrode_names: List[str]):
        self.study = study
        self.carriers = [np.float64(c) for c in carriers]
        self.electrode_names = electrode_names

    def _get_comparison_data(self) -> Iterator[Tuple[str, List[float]]]:
        """
        Extracts SNR at carrier frequencies for subjects who participated in all requested conditions.
        Returns: Iterator of (subject_name, [snr_carrier1, snr_carrier2, ...])
        """
        # We assume target frequency is 5Hz for these comparisons as per user's hardcoded update
        props_list = [ConditionProperties(np.float64(5), c) for c in self.carriers]
        
        common_subjects = list(self.study.filter_subjects(set(props_list)))
        
        for subject in common_subjects:
            subject_snrs = []
            for props in props_list:
                view = subject[props].restrict_electrodes(self.electrode_names)
                snr, _ = view.snr_at_target()
                subject_snrs.append(float(snr))
            
            yield (subject.name, subject_snrs)

    def _calculate_slope(self, item: Tuple[str, List[float]]) -> Tuple[str, float]:
        name, snrs = item
        if len(self.carriers) < 2:
            return (name, 0.0)
        slope, _ = np.polyfit(self.carriers, snrs, 1)
        return (name, float(slope))

    def slopes(self, data: Iterator[Tuple[str, List[float]]]) -> Iterator[Tuple[str, float]]:
        """
        Calculates the slope of SNR vs Carrier Frequency for each subject.
        """
        return map(self._calculate_slope, data)


    def plot(self):
        """
        Creates a slope plot comparing SNR across carrier frequencies.
        """
        data = list(self._get_comparison_data())
        if not data:
            print(f"No subjects found who participated in all requested conditions: {self.carriers}Hz.")
            return

        fig, ax = plt.subplots(figsize=(10, 6), label=f"carrier-comparison-{'-'.join(map(str, self.carriers))}")
        
        x = np.arange(len(self.carriers))
        for name, snrs in data:
            ax.plot(x, snrs, marker='o', label=name)
            # Add text labels
            for i, snr in enumerate(snrs):
                ax.text(i, snr, f"{snr:.2f}", horizontalalignment='center', verticalalignment='bottom')

        ax.set_xticks(x)
        ax.set_xticklabels([f"{c} Hz" for c in self.carriers])
        ax.set_ylabel("SNR at Carrier Frequency")
        ax.set_title(f"Subject-wise SNR Comparison across Carriers\n(Electrodes: {self.electrode_names})")
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        ax.set_xlim(-0.5, len(self.carriers) - 0.5)
        
        # Place legend outside
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.tight_layout()

def plot_latency_mean_vs_snr_slope(study: Study,
                                       carriers: List[float],
                                       latency_carrier: float,
                                       target_electrodes: List[str],
                                       carrier_electrodes: List[str]):
    """
    Analyzes and plots the relationship between the mean of neural processing time
    and the SNR slope across different carrier frequencies.
    """
    comparison = CarrierComparisonAnalysis(study, carriers, target_electrodes)
    # Re-use logic for identifying common subjects and their slopes
    data = list(comparison._get_comparison_data())
    if not data:
        print("No data for Latency mean vs SNR Slope analysis.")
        return

    slopes_dict = dict(comparison.slopes(iter(data)))
    all_subjects = {s.name: s for s in study.subjects()}

    names = []
    slope_values = []
    mean_values = []

    for name, slope in slopes_dict.items():
        subject = all_subjects[name]
        # Convention: target frequency is 5Hz for these carrier comparisons
        props = ConditionProperties(np.float64(5), np.float64(latency_carrier))

        try:
            view = subject[props].restrict_electrodes(target_electrodes + carrier_electrodes)
            # calculate_processing_time returns (Subject, Trial)
            latencies = view.calculate_processing_time()
            mean = float(np.mean(latencies))

            names.append(name)
            slope_values.append(float(slope))
            mean_values.append(mean)
        except KeyError:
            continue

    if not slope_values:
        print("No matching subjects for Latency mean vs SNR Slope plot.")
        return

    fig, ax = plt.subplots(figsize=(8, 6), label="latency-mean-vs-snr-slope")
    ax.scatter(slope_values, mean_values, color='blue', alpha=0.7)

    for i, name in enumerate(names):
        ax.annotate(name, (slope_values[i], mean_values[i]),
                    textcoords="offset points", xytext=(0, 10), ha='center')

    if len(slope_values) > 1:
        r, p = pearsonr(slope_values, mean_values)
        title = f"Latency mean vs. SNR Slope\n(r={r:.3f}, p={p:.3f})"
    else:
        title = "Latency mean vs. SNR Slope"

    ax.set_title(title)
    ax.set_xlabel("SNR Slope (SNR/Hz)")
    ax.set_ylabel("Latency mean s")
    ax.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
