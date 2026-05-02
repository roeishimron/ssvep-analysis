from typing import Callable, Hashable, Iterator, List, Tuple, TypeVar

import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import pearsonr

import mne

from analysis import SSVEPAnalysis, Spectral
from core_types import Array1D_f64, ConditionProperties, SubjectPower
from interfaces import Experiment, Recording, SSVEPRecording, SubjectHandle, TopomapSSVEPRecording


K = TypeVar("K", bound=Hashable)


def analyze_spectrum(
    rec: SSVEPRecording, fmin: float, fmax: float, show_psd: bool,
) -> None:
    analysis = SSVEPAnalysis(rec)
    freqs: Array1D_f64 = analysis.frequencies()
    snr_mean, snr_std = analysis.snr_spectrum()

    if show_psd:
        amp_fig, amp_ax = plt.subplots(
            1, 1, figsize=(8, 3),
            label=f"{rec.name()}-{rec.props().carrier_frequency}-amplitudes",
        )
        amplitudes_mean, amplitudes_std = analysis.power_spectrum()
        amp_ax.plot(freqs, 20 * np.log10(amplitudes_mean))
        # Floor the lower edge before log10 so we never feed it 0 or negatives —
        # protects matplot2tikz's pgfplots output from "Dimension too large".
        amp_low = np.maximum(amplitudes_mean - amplitudes_std, 1e-6)
        amp_ax.fill_between(
            freqs, 20 * np.log10(amp_low),
            20 * np.log10(amplitudes_mean + amplitudes_std),
            color="r", alpha=0.1,
        )
        amp_ax.set(
            title="Amplitudes spectrum",
            ylabel=r"Amplitude (dB$\mu$V)",
            xlabel='$\\downarrow$ SNR calculation $\\downarrow$',
            xlim=[fmin, fmax],
        )

    fig, ax = plt.subplots(
        1, 1, figsize=(8, 3),
        label=f"{rec.name()}-{rec.props().carrier_frequency}-spectrum",
    )
    ax.plot(freqs, snr_mean)
    snr_ymax = np.max((snr_mean + snr_std)[freqs <= fmax])
    ax.fill_between(
        freqs, np.clip(snr_mean - snr_std, 0, snr_ymax),
        np.clip(snr_mean + snr_std, 0, snr_ymax),
        color="r", alpha=0.1,
    )
    ax.set(
        title="SNR",
        ylabel="SNR",
        xlabel="Frequency [Hz]",
        xlim=[fmin, fmax],
        ylim=[0, snr_ymax],
    )
    ax.axhline(1, color='red', linestyle='--', alpha=0.5)
    if show_psd:
        ax.annotate(
            'SNR=1', xy=(fmin, 1), xytext=(fmin + 0.1, 1.5),
            fontsize=11, color='red', alpha=0.7,
        )


def _into_channel_average(
    freqs: Array1D_f64, snrs: SubjectPower, target_freq: float,
) -> Tuple[float, np.ndarray]:
    target_center = int(np.argmin(np.abs(freqs - target_freq)))
    return float(freqs[target_center]), snrs[:, target_center]


def plot_snrs(rec: TopomapSSVEPRecording) -> None:
    TOPO_WIDTH = 4
    TOPO_HEIGHT = 1

    analysis = SSVEPAnalysis(rec)
    target_freq = float(rec.props().target_frequency)
    carrier_freq = float(rec.props().carrier_frequency)
    freqs: Array1D_f64 = analysis.frequencies()
    snr: SubjectPower = analysis.snr_topomap()

    HARMONIC_FREQS = np.arange(TOPO_WIDTH * TOPO_HEIGHT) + 1
    TARGET_FREQS = target_freq * HARMONIC_FREQS

    freqs_with_channel_averages = [
        _into_channel_average(freqs, snr, f) for f in TARGET_FREQS
    ]

    upper_limit = np.max(
        np.array([t[1] for t in freqs_with_channel_averages]).flatten()
    )
    fig, axs = plt.subplots(
        TOPO_HEIGHT, TOPO_WIDTH, sharex="none", sharey="none",
        label=f"{rec.name()}-{carrier_freq}-topomap",
    )

    info = rec.mne_info()
    im = None
    for ((freq, channel_average), ax) in zip(freqs_with_channel_averages, axs.flatten()):
        print(f"looking at freq {freq:.2f}")
        ax.set_title(f"{freq:.0f} Hz")
        im, _ = mne.viz.plot_topomap(
            channel_average, info, vlim=(1, upper_limit), axes=ax, show=False,
        )

    if im is not None:
        fig.colorbar(im, ax=axs[-1], shrink=1.0, fraction=0.08, pad=0.05, label="SNR")


def plot_snr_comparison(
    experiment: Experiment[ConditionProperties, SSVEPRecording], electrode_names: List[str],
) -> None:
    labels: List[str] = []
    snr_means: List[float] = []
    snr_sems: List[float] = []
    psd_means: List[float] = []
    psd_sems: List[float] = []

    for props, group_view in experiment.conditions().items():
        analysis = SSVEPAnalysis(group_view.take_channels(electrode_names))

        snr_m, snr_s = analysis.snr_at_target()
        snr_means.append(snr_m)
        snr_sems.append(snr_s)

        psd_m, psd_s = analysis.power_at_target()
        psd_means.append(psd_m)
        psd_sems.append(psd_s)

        labels.append(f"{props.carrier_frequency:.0f} Hz")

    fig, ax = plt.subplots(
        1, 1, figsize=(10, 10), sharex=True, label=f"comparison-{electrode_names}",
    )
    x = np.arange(len(labels))

    ax.bar(x, snr_means, yerr=snr_sems, capsize=5, color='skyblue', edgecolor='navy')
    ax.set_ylabel("SNR")
    ax.set_title(
        f"SNR at Target Frequency over {electrode_names}\n"
        r"(Mean $\pm$ SEM across subjects)"
    )
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.axhline(1, color='red', linestyle='--', alpha=0.5, label="Noise Floor")
    ax.legend()

    plt.xticks(x, labels)
    plt.tight_layout()


class CarrierComparisonAnalysis:
    """SNR at carrier frequencies for the intersection of subjects across carriers."""

    def __init__(
        self, study: Experiment[ConditionProperties, SSVEPRecording], carriers: List[float], electrode_names: List[str],
    ) -> None:
        self.study = study
        self.carriers = [np.float64(c) for c in carriers]
        self.electrode_names = electrode_names

    def _get_comparison_data(
        self,
    ) -> Iterator[Tuple[str, List[Tuple[float, float]]]]:
        # Target frequency hardcoded at 5Hz, matching prior project convention.
        props_list = [ConditionProperties(np.float64(5), c) for c in self.carriers]

        common_subjects = list(self.study.filter_subjects(set(props_list)))

        for subject in common_subjects:
            subject_snrs: List[Tuple[float, float]] = []
            for props in props_list:
                view = subject[props].take_channels(self.electrode_names)
                snr, snr_sem = SSVEPAnalysis(view).snr_at_target()
                subject_snrs.append((float(snr), float(snr_sem)))
            yield (subject.name, subject_snrs)

    def _calculate_slope(
        self, item: Tuple[str, List[Tuple[float, float]]],
    ) -> Tuple[str, float]:
        name, snrs = item
        if len(self.carriers) < 2:
            return (name, 0.0)
        means = [m for m, _ in snrs]
        slope, _ = np.polyfit(self.carriers, means, 1)
        return (name, float(slope))

    def slopes(
        self, data: Iterator[Tuple[str, List[Tuple[float, float]]]],
    ) -> Iterator[Tuple[str, float]]:
        return map(self._calculate_slope, data)

    def plot(self) -> None:
        data = list(self._get_comparison_data())
        if not data:
            print(f"No subjects found who participated in all requested conditions: {self.carriers}Hz.")
            return

        fig, ax = plt.subplots(
            figsize=(10, 6),
            label=f"carrier-comparison-{'-'.join(map(str, self.carriers))}",
        )

        x = np.arange(len(self.carriers))
        for name, snrs in data:
            means = [m for m, _ in snrs]
            sems = [s for _, s in snrs]
            ax.errorbar(
                x, means, yerr=sems, marker='o', capsize=3,
                label=name.replace("_", " "),
            )
            for i, (m, _) in enumerate(snrs):
                ax.text(
                    i, m, f"{m:.2f}",
                    horizontalalignment='center', verticalalignment='bottom',
                )

        ax.set_xticks(x)
        ax.set_xticklabels([f"{c} Hz" for c in self.carriers])
        ax.set_ylabel("SNR at Carrier Frequency")
        ax.set_title(
            f"Subject-wise SNR Comparison across Carriers\n(Electrodes: {self.electrode_names})"
        )
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        ax.set_xlim(-0.5, len(self.carriers) - 0.5)
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.tight_layout()


def plot_latency_mean_vs_snr_slope(
    study: Experiment[ConditionProperties, SSVEPRecording],
    carriers: List[float],
    latency_carriers: List[float],
    target_electrodes: List[str],
    carrier_electrodes: List[str],
) -> None:
    """Scatter plot: mean processing-time vs. SNR-slope-across-carriers per subject.

    Latencies are pooled across `latency_carriers` per subject for tighter SEM.
    """
    comparison = CarrierComparisonAnalysis(study, carriers, target_electrodes)
    data = list(comparison._get_comparison_data())
    if not data:
        print("No data for Latency mean vs SNR Slope analysis.")
        return

    slopes_dict = dict(comparison.slopes(iter(data)))
    all_subjects = study.subjects()

    latency_requirements = {
        ConditionProperties(np.float64(5), np.float64(c)) for c in latency_carriers
    }
    eligible_names = {s.name for s in study.filter_subjects(latency_requirements)}

    names: List[str] = []
    slope_values: List[float] = []
    mean_values: List[float] = []
    sd_values: List[float] = []

    for name, slope in slopes_dict.items():
        if name not in eligible_names:
            continue
        subject = all_subjects[name]
        per_carrier_means: List[float] = []
        per_carrier_sds: List[float] = []
        for lc in latency_carriers:
            props = ConditionProperties(np.float64(5), np.float64(lc))
            view = subject[props].take_channels(target_electrodes + carrier_electrodes)
            mean_ms, sd_ms = SSVEPAnalysis(view).processing_time_summary()
            per_carrier_means.append(float(mean_ms[0]))
            sd = float(sd_ms[0])
            per_carrier_sds.append(0.0 if np.isnan(sd) else sd)

        mean = float(np.mean(per_carrier_means))
        lat_sd = float(np.sqrt(np.mean(np.square(per_carrier_sds))))

        names.append(name)
        slope_values.append(float(slope))
        mean_values.append(mean)
        sd_values.append(lat_sd)

    if not slope_values:
        print("No matching subjects for Latency mean vs SNR Slope plot.")
        return

    fig, ax = plt.subplots(figsize=(8, 6), label="latency-mean-vs-snr-slope")
    ax.errorbar(
        slope_values, mean_values, yerr=sd_values,
        fmt='o', color='blue', alpha=0.7, capsize=3,
        label='mean ± SD (trial-to-trial)',
    )
    _ = names  # subject names omitted from the exported figure on purpose

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


def plot_snr_spectra_overlay(
    experiment: Experiment[K, Recording[K]],
    fmin: float,
    fmax: float,
    electrodes: List[str],
    label_func: Callable[[K], str] = str,
) -> None:
    """Overlay every condition's group-aggregated SNR spectrum on one axis.

    Paradigm-agnostic — works on any Experiment[K, Recording[K]] (SSVEP,
    dot-experiments, etc). Each condition gets a colored line plus a shaded
    SEM band. `label_func` controls the legend entry for each key.
    """
    fig, ax = plt.subplots(figsize=(10, 6), label="snr-spectra-overlay")

    for key, view in experiment.conditions().items():
        spectral = Spectral(view.take_channels(electrodes))
        freqs = spectral.frequencies()
        snr_mean, snr_sem = spectral.snr_spectrum()
        mask = (freqs >= fmin) & (freqs <= fmax)
        line = ax.plot(freqs[mask], snr_mean[mask], label=label_func(key))[0]
        lower = np.clip(snr_mean[mask] - snr_sem[mask], 0, None)
        upper = snr_mean[mask] + snr_sem[mask]
        ax.fill_between(freqs[mask], lower, upper, alpha=0.2, color=line.get_color())

    ax.axhline(1, color="red", linestyle="--", alpha=0.5, label="noise floor")
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("SNR")
    ax.set_xlim(fmin, fmax)
    ax.set_title(f"SNR spectra by condition (electrodes: {electrodes})")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend()
    plt.tight_layout()
