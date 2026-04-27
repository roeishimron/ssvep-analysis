from typing import Any, Tuple, List, Iterator
from matplotlib import pyplot as plt
import mne
from scipy.stats import pearsonr, sem
from power_specra_analyzable import PowerSpectcraAnalyzable
import numpy as np
from core_types import SubjectPower, Array1D_f64, ConditionProperties
from core import Study

def analyze_spectrum(subject: PowerSpectcraAnalyzable, fmin: float, fmax: float, show_psd: bool):

    print("got snr")
    title = f"Target: {subject.target_frequency()}Hz, Carrier: {subject.carrier_frequency()}Hz"

    freqs: Array1D_f64 = subject.frequencies()
    snr_mean, snr_std = subject.as_snr_average()

    if show_psd:
        # Generate separate amplitudes figure
        amp_fig, amp_ax = plt.subplots(1, 1, figsize=(8, 3),
            label=f"{subject.name()}-{subject.carrier_frequency()}-amplitudes")
        amplitudes_mean, amplitudes_std = subject.as_power_spectrum()
        amp_ax.plot(freqs, 20 * np.log10(amplitudes_mean))
        # Floor the lower edge before log10 so we never feed it 0 or negatives —
        # protects matplot2tikz's pgfplots output from "Dimension too large".
        amp_low = np.maximum(amplitudes_mean - amplitudes_std, 1e-6)
        amp_ax.fill_between(
            freqs, 20 * np.log10(amp_low),
            20 * np.log10(amplitudes_mean + amplitudes_std),
            color="r", alpha=0.1
        )
        amp_ax.set(
            title="Amplitudes spectrum",
            ylabel=r"Amplitude (dB$\mu$V)",
            xlabel='$\\downarrow$ SNR calculation $\\downarrow$',
            xlim=[fmin, fmax],
        )

    # SNR figure (same for all conditions)
    fig, ax = plt.subplots(1, 1, figsize=(8, 3),
        label=f"{subject.name()}-{subject.carrier_frequency()}-spectrum")
    ax.plot(freqs, snr_mean)
    # Clip both edges of the SEM band to the visible y-axis range. With
    # high-variance subjects there are out-of-range frequency bins where
    # snr_mean + snr_std reaches the hundreds; matplot2tikz still writes those
    # raw values into the pgfplots table and xelatex then aborts with
    # "Dimension too large" when transforming the clipped coordinates.
    snr_ymax = np.max((snr_mean + snr_std)[freqs <= fmax])
    ax.fill_between(
        freqs, np.clip(snr_mean - snr_std, 0, snr_ymax),
        np.clip(snr_mean + snr_std, 0, snr_ymax),
        color="r", alpha=0.1
    )
    ax.set(
        title="SNR",
        ylabel="SNR",
        xlabel="Frequency [Hz]",
        xlim=[fmin, fmax],
        ylim=[0, snr_ymax]
    )
    ax.axhline(1, color='red', linestyle='--', alpha=0.5)
    if show_psd:
        ax.annotate('SNR=1', xy=(fmin, 1), xytext=(fmin + 0.1, 1.5),
                    fontsize=11, color='red', alpha=0.7)


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
    # plot SNR topography — figure-level suptitle is omitted on purpose; the
    # LyX caption describes the figure content already.
    fig, axs = plt.subplots(TOPO_HEIGHT, TOPO_WIDTH,  sharex="none",
                            sharey="none", label=f"{subject.name()}-{carrier_freq}-topomap")

    im = None
    for ((freq, channel_average), ax) in zip(freqs_with_channel_averages, axs.flatten()):

        print(f"looking at freq {freq:.2f}")
        ax.set_title(f"{freq:.0f} Hz")

        im, _ = mne.viz.plot_topomap(channel_average, raw_mne_info,
                                     vlim=(1, upper_limit), axes=ax, show=False)

    if im is not None:
        # Anchor the colorbar to the rightmost subplot so it spans only one
        # head's height, not the whole row.
        fig.colorbar(im, ax=axs[-1], shrink=1.0, fraction=0.08, pad=0.05, label="SNR")

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
        
        labels.append(f"{props.carrier_frequency:.0f} Hz")
        
    fig, ax = plt.subplots(1, 1, figsize=(10, 10), sharex=True, label=f"comparison-{electrode_names}")
    x = np.arange(len(labels))
    
    # SNR Plot
    ax.bar(x, snr_means, yerr=snr_sems, capsize=5, color='skyblue', edgecolor='navy')
    ax.set_ylabel("SNR")
    ax.set_title(f"SNR at Target Frequency over {electrode_names}\n"
                 r"(Mean $\pm$ SEM across subjects)")
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.axhline(1, color='red', linestyle='--', alpha=0.5, label="Noise Floor")
    ax.legend()

    plt.xticks(x, labels)
    plt.tight_layout()

class CarrierComparisonAnalysis:
    """
    Analyzes and plots SNR at carrier frequencies for subjects who participated in specific carrier conditions.
    """
    def __init__(self, study: Study, carriers: List[float], electrode_names: List[str]):
        self.study = study
        self.carriers = [np.float64(c) for c in carriers]
        self.electrode_names = electrode_names

    def _get_comparison_data(self) -> Iterator[Tuple[str, List[Tuple[float, float]]]]:
        """
        Extracts SNR at carrier frequencies for subjects who participated in all requested conditions.
        Returns: Iterator of (subject_name, [(snr_mean, snr_sem), ...])
        """
        # We assume target frequency is 5Hz for these comparisons as per user's hardcoded update
        props_list = [ConditionProperties(np.float64(5), c) for c in self.carriers]

        common_subjects = list(self.study.filter_subjects(set(props_list)))

        for subject in common_subjects:
            subject_snrs: List[Tuple[float, float]] = []
            for props in props_list:
                view = subject[props].restrict_electrodes(self.electrode_names)
                snr, snr_sem = view.snr_at_target()
                subject_snrs.append((float(snr), float(snr_sem)))

            yield (subject.name, subject_snrs)

    def _calculate_slope(self, item: Tuple[str, List[Tuple[float, float]]]) -> Tuple[str, float]:
        name, snrs = item
        if len(self.carriers) < 2:
            return (name, 0.0)
        means = [m for m, _ in snrs]
        slope, _ = np.polyfit(self.carriers, means, 1)
        return (name, float(slope))

    def slopes(self, data: Iterator[Tuple[str, List[Tuple[float, float]]]]) -> Iterator[Tuple[str, float]]:
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
            means = [m for m, _ in snrs]
            sems = [s for _, s in snrs]
            ax.errorbar(x, means, yerr=sems, marker='o', capsize=3, label=name.replace("_", " "))
            # Add text labels
            for i, (m, _) in enumerate(snrs):
                ax.text(i, m, f"{m:.2f}", horizontalalignment='center', verticalalignment='bottom')

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
                                       latency_carriers: List[float],
                                       target_electrodes: List[str],
                                       carrier_electrodes: List[str]):
    """
    Analyzes and plots the relationship between the mean of neural processing time
    and the SNR slope across different carrier frequencies.

    Latencies are pooled across all `latency_carriers` per subject, giving more
    samples (and thus a tighter SEM) for the processing-time estimate.
    """
    comparison = CarrierComparisonAnalysis(study, carriers, target_electrodes)
    # Re-use logic for identifying common subjects and their slopes
    data = list(comparison._get_comparison_data())
    if not data:
        print("No data for Latency mean vs SNR Slope analysis.")
        return

    slopes_dict = dict(comparison.slopes(iter(data)))
    all_subjects = {s.name: s for s in study.subjects()}

    # Require participation in every latency-carrier condition as well —
    # consistent with Study.filter_subjects usage across the project.
    latency_requirements = {
        ConditionProperties(np.float64(5), np.float64(c)) for c in latency_carriers
    }
    eligible_names = {s.name for s in study.filter_subjects(latency_requirements)}

    names = []
    slope_values = []
    mean_values = []
    sem_values = []

    for name, slope in slopes_dict.items():
        if name not in eligible_names:
            continue
        subject = all_subjects[name]
        # Convention: target frequency is 5Hz for these carrier comparisons
        per_carrier_latencies: List[np.ndarray] = []
        for lc in latency_carriers:
            props = ConditionProperties(np.float64(5), np.float64(lc))
            view = subject[props].restrict_electrodes(target_electrodes + carrier_electrodes)
            # calculate_processing_time returns (Subject, Trial)
            per_carrier_latencies.append(
                np.asarray(view.calculate_processing_time()).flatten()
            )

        flat = np.concatenate(per_carrier_latencies) * 1000
        mean = float(np.mean(flat))
        lat_sem = float(sem(flat)) if flat.size > 1 else 0.0

        names.append(name)
        slope_values.append(float(slope))
        mean_values.append(mean)
        sem_values.append(lat_sem)

    if not slope_values:
        print("No matching subjects for Latency mean vs SNR Slope plot.")
        return

    fig, ax = plt.subplots(figsize=(8, 6), label="latency-mean-vs-snr-slope")
    ax.errorbar(slope_values, mean_values, yerr=sem_values,
                fmt='o', color='blue', alpha=0.7, capsize=3)

    # Subject names are deliberately omitted from the exported figure so
    # participant identifiers don't end up in the PDF. The `names` list is
    # still computed above for downstream stats / debugging.
    _ = names

    if len(slope_values) > 1:
        r, p = pearsonr(slope_values, mean_values)
        title = f"Latency mean vs. SNR Slope\n(r={r:.3f}, p={p:.3f})"
    else:
        title = "Latency mean vs. SNR Slope"

    ax.set_title(title)
    ax.set_xlabel("SNR Slope (SNR/Hz)")
    ax.set_ylabel("Latency mean (ms)")
    ax.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
