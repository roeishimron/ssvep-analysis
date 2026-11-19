from typing import Callable, Dict, Hashable, Iterator, List, Tuple, TypeVar

import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import pearsonr

import mne

from analysis import (
    ExistenceResult, FrequencyTagging, SSVEPAnalysis, Spectral, is_carrier_harmonic,
    oddball_harmonics,
)
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
        # Power of the window-averaged spectrum, in (data units)^2 = uV^2.
        power_mean, power_sem = analysis.power_spectrum()
        amp_ax.plot(freqs, 10 * np.log10(power_mean))
        # Floor the lower edge before log10 so we never feed it 0 or negatives —
        # protects matplot2tikz's pgfplots output from "Dimension too large".
        power_low = np.maximum(power_mean - power_sem, 1e-12)
        amp_ax.fill_between(
            freqs, 10 * np.log10(power_low),
            10 * np.log10(power_mean + power_sem),
            color="r", alpha=0.1,
        )
        amp_ax.set(
            title="Power spectrum (2 s windows, coherent average; mean $\\pm$ SEM)",
            ylabel=r"Power (dB re 1 $\mu$V$^2$)",
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
        title="Power SNR (2 s windows, 3 neighbours per side; mean $\\pm$ SEM)",
        ylabel="Power SNR",
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
        # A harmonic of the oddball that is also a multiple of this condition's
        # carrier shows the carrier response, not an oddball response.
        ax.set_title(f"{freq:.0f} Hz" + (" (carrier)" if is_carrier_harmonic(freq, carrier_freq) else ""))
        im, _ = mne.viz.plot_topomap(
            channel_average, info, vlim=(1, upper_limit), axes=ax, show=False,
        )

    if im is not None:
        fig.colorbar(im, ax=axs[-1], shrink=1.0, fraction=0.08, pad=0.05, label="Power SNR (2 s windows)")


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
    ax.set_ylabel("Power SNR at the target frequency (2 s windows)")
    ax.set_title(
        f"Power SNR at Target Frequency over {electrode_names}\n"
        r"(Mean $\pm$ SEM across subjects; legacy measure)"
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
        # The value is the SNR at the 5 Hz target, measured in each carrier condition.
        ax.set_ylabel("Power SNR at 5 Hz (2 s windows)")
        ax.set_title(
            f"Subject-wise SNR Comparison across Carriers\n(Electrodes: {self.electrode_names})"
        )
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        ax.set_xlim(-0.5, len(self.carriers) - 0.5)
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.tight_layout()


def _min_snr_over_carriers(
    subject: SubjectHandle,
    latency_carriers: List[float],
    electrodes: List[str],
    at_carrier: bool,
) -> float:
    """Subject's trial-averaged SNR at the max-response electrode, min over carriers.

    Per carrier, take the trial-averaged SNR under the strongest of `electrodes`
    (not the electrode average); `at_carrier` selects the carrier-frequency SNR
    (for V1/carrier electrodes), otherwise the 5Hz target SNR (target electrodes).
    """
    snrs: List[float] = []
    for lc in latency_carriers:
        props = ConditionProperties(np.float64(5), np.float64(lc))
        analysis = SSVEPAnalysis(subject[props].take_channels(electrodes))
        freq = analysis._carrier_frequency() if at_carrier else analysis._target_frequency()
        per_channel = analysis.snr_topomap()[:, analysis._closest_index(freq)]  # (C,)
        snrs.append(float(np.max(per_channel)))
    return float(np.min(snrs))


def _plot_latency_mean_vs_snr_metric(
    study: Experiment[ConditionProperties, SSVEPRecording],
    carriers: List[float],
    latency_carriers: List[float],
    target_electrodes: List[str],
    carrier_electrodes: List[str],
    metric_by_subject: Dict[str, float],
    metric_label: str,
    figure_label: str,
    min_snr: float = 1.0,
) -> None:
    """Scatter plot of per-subject latency mean against an arbitrary SNR metric.

    `metric_by_subject` maps subject name -> x-value (e.g. SNR slope or SNR at a
    given carrier). Latencies are pooled across `latency_carriers` per subject
    for tighter SEM. Only subjects with all required conditions are plotted, and
    subjects are excluded as low-quality when (averaged over the latency
    carriers) their target-electrode SNR at 5Hz or their carrier-electrode SNR
    at the carrier frequency falls below `min_snr`.
    """
    all_subjects = study.subjects()

    latency_requirements = {
        ConditionProperties(np.float64(5), np.float64(c)) for c in latency_carriers
    }
    carrier_requirements = {
        ConditionProperties(np.float64(5), np.float64(c)) for c in carriers
    }
    eligible_names = {s.name for s in study.filter_subjects(latency_requirements.union(carrier_requirements))}

    metric_values: List[float] = []
    mean_values: List[float] = []
    sem_values: List[float] = []
    dropped = 0

    for name, metric in metric_by_subject.items():
        if name not in eligible_names:
            continue
        subject = all_subjects[name]

        target_snr = _min_snr_over_carriers(subject, latency_carriers, target_electrodes, at_carrier=False)
        carrier_snr = _min_snr_over_carriers(subject, latency_carriers, carrier_electrodes, at_carrier=True)
        if target_snr < min_snr or carrier_snr < min_snr :
            dropped += 1
            continue

        per_carrier_means: List[float] = []
        per_carrier_sds: List[float] = []
        total_trials = 0
        for lc in latency_carriers:
            props = ConditionProperties(np.float64(5), np.float64(lc))
            view = subject[props].take_channels(target_electrodes + carrier_electrodes)
            mean_ms, sd_ms = SSVEPAnalysis(view).processing_time_summary()
            per_carrier_means.append(float(mean_ms[0]))
            sd = float(sd_ms[0])
            per_carrier_sds.append(0.0 if np.isnan(sd) else sd)
            total_trials += view.raw_data().shape[1]

        mean = float(np.mean(per_carrier_means))
        # RMS of the per-carrier trial-to-trial SDs, converted to the SEM of the
        # pooled mean via /sqrt(N_total). Exact equal-weight pooling when trial
        # counts match across carriers; a close approximation otherwise.
        lat_sd = float(np.sqrt(np.mean(np.square(per_carrier_sds))))
        lat_sem = lat_sd / np.sqrt(total_trials) if total_trials else 0.0

        metric_values.append(float(metric))
        mean_values.append(mean)
        sem_values.append(lat_sem)

    if dropped:
        print(f"  Excluded {dropped} subject(s) with mean target or carrier SNR < {min_snr:g}.")

    if not metric_values:
        print(f"No matching subjects for Latency mean vs {metric_label} plot.")
        return

    fig, ax = plt.subplots(figsize=(8, 6), label=figure_label)
    ax.errorbar(
        metric_values, mean_values, yerr=sem_values,
        fmt='o', color='blue', alpha=0.7, capsize=3,
        label='mean ± SEM (trial-to-trial)',
    )

    title = f"Latency mean vs. {metric_label}"
    if len(metric_values) > 1:
        r, p = pearsonr(metric_values, mean_values)
        title += f"\n(r={r:.3f}, p={p:.3f})"

    ax.set_title(title)
    ax.set_xlabel(metric_label)
    ax.set_ylabel("Latency mean (ms)")
    ax.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()


def plot_latency_mean_vs_snr_slope(
    study: Experiment[ConditionProperties, SSVEPRecording],
    carriers: List[float],
    latency_carriers: List[float],
    target_electrodes: List[str],
    carrier_electrodes: List[str],
) -> None:
    """Scatter plot: mean processing-time vs. SNR-slope-across-carriers per subject."""
    comparison = CarrierComparisonAnalysis(study, carriers, target_electrodes)
    data = list(comparison._get_comparison_data())
    if not data:
        print("No data for Latency mean vs SNR Slope analysis.")
        return

    _plot_latency_mean_vs_snr_metric(
        study, carriers, latency_carriers, target_electrodes, carrier_electrodes,
        metric_by_subject=dict(comparison.slopes(iter(data))),
        metric_label="SNR Slope (SNR/Hz)",
        figure_label="latency-mean-vs-snr-slope",
    )


def plot_latency_mean_vs_snr_at_carrier(
    study: Experiment[ConditionProperties, SSVEPRecording],
    snr_carrier: float,
    latency_carriers: List[float],
    target_electrodes: List[str],
    carrier_electrodes: List[str],
) -> None:
    """Scatter plot: mean processing-time vs. SNR at `snr_carrier` Hz per subject.

    Includes every subject with the `snr_carrier` condition (and the latency
    carriers) — unlike the slope plot, no other carriers constrain the population.
    """
    comparison = CarrierComparisonAnalysis(study, [snr_carrier], target_electrodes)
    data = list(comparison._get_comparison_data())
    if not data:
        print("No data for Latency mean vs SNR-at-carrier analysis.")
        return

    _plot_latency_mean_vs_snr_metric(
        study, [snr_carrier], latency_carriers, target_electrodes, carrier_electrodes,
        metric_by_subject={name: snrs[0][0] for name, snrs in data},
        metric_label=f"SNR at {snr_carrier:g} Hz",
        figure_label=f"latency-mean-vs-snr-at-{snr_carrier:g}hz",
    )


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


# --- Full-length spectrum (FrequencyTagging): the default reported measures ---
# Fixed regions of interest, chosen in advance; no per-subject channel selection.
ODDBALL_ROI = ["T5", "T6"]
CARRIER_ROI = ["O1", "O2"]


def _by_carrier(experiment: Experiment[ConditionProperties, SSVEPRecording]):
    return sorted(experiment.conditions().items(), key=lambda kv: float(kv[0].carrier_frequency))


def report_bad_channels(records: List[dict]) -> str:
    """Summary of the loader's per-recording bad-channel screen (StudyLoader.records)."""
    counts: Dict[str, int] = {}
    for r in records:
        for c in r["bads"]:
            counts[c] = counts.get(c, 0) + 1
    n_bad = [len(r["bads"]) for r in records]
    if not records:
        return "No recordings."
    roi = [f"{r['label']}: {[c for c in r['bads'] if c in ODDBALL_ROI + CARRIER_ROI]}"
           for r in records if any(c in ODDBALL_ROI + CARRIER_ROI for c in r["bads"])]
    lines = [
        f"Bad-channel screen: {len(records)} recordings, {sum(n > 0 for n in n_bad)} with at least one "
        f"bad channel (mean {np.mean(n_bad):.2f}, max {max(n_bad)} per recording)",
        "  by channel: " + ", ".join(f"{c} {n}" for c, n in sorted(counts.items(), key=lambda kv: -kv[1])),
        "  recordings with a bad ROI channel: " + ("; ".join(roi) if roi else "none"),
    ]
    trials = [r for r in records if r.get("n_bursts", r["n_used"]) != r["n_used"]]
    lines.append("Trigger bursts not used (bursts / complete / used): " + (
        "; ".join(f"{r['label']}: {r.get('n_bursts')}/{r['n_complete']}/{r['n_used']}" for r in trials)
        if trials else "none"))
    return "\n".join(lines)


def format_existence(label: str, r: ExistenceResult) -> str:
    det = "  ".join(f"z>{th:g}: {n}/{r.n}" for th, n in r.detected.items())
    excl = f" ({r.n_excluded} excluded: ROI all bad)" if r.n_excluded else ""
    return (f"{label:22s} n={r.n}{excl}  BCA {r.mean_bca:.3f} uV (sd {r.sd_bca:.3f}, d={r.cohen_d:.2f})  "
            f"t={r.t:.2f} p={r.p_t:.2g}  Wilcoxon p={r.p_wilcoxon:.2g}  |  {det}  |  "
            f"grand-average z={r.grand_z:.1f} SNR={r.grand_snr:.2f}  noise {r.mean_noise:.3f} uV")


def existence_report(
    experiment: Experiment[ConditionProperties, SSVEPRecording],
    roi: List[str] = ODDBALL_ROI,
    at_carrier: bool = False,
) -> Dict[ConditionProperties, ExistenceResult]:
    """Existence statistics per condition at the oddball (or carrier) frequency over a fixed ROI."""
    out: Dict[ConditionProperties, ExistenceResult] = {}
    for props, view in _by_carrier(experiment):
        freq = float(props.carrier_frequency if at_carrier else props.target_frequency)
        out[props] = FrequencyTagging(view.take_channels(roi)).existence(freq)
        print(format_existence(
            f"{props.carrier_frequency:g} Hz stream, {freq:g} Hz", out[props]))
    return out


def harmonic_report(
    experiment: Experiment[ConditionProperties, SSVEPRecording],
    roi: List[str] = ODDBALL_ROI,
    fmax: float = 35.0,
    z_min: float = 1.64,
) -> Dict[ConditionProperties, List[float]]:
    """Grand-average z at each oddball harmonic; carrier harmonics are labelled, not measured
    as oddball responses. Returns the clean harmonics with grand-average z > z_min per
    condition, and prints the summed BCA over them."""
    significant: Dict[ConditionProperties, List[float]] = {}
    for props, view in _by_carrier(experiment):
        ft = FrequencyTagging(view.take_channels(roi))
        cells, sig = [], []
        for f, clean in oddball_harmonics(float(props.target_frequency),
                                          float(props.carrier_frequency), fmax):
            z = ft.grand_average_at(f)["z"]
            cells.append(f"{f:g}:{z:.1f}" + ("" if clean else "(carrier)"))
            if clean and z > z_min:
                sig.append(f)
        significant[props] = sig
        summed = ft.summed_at(sig)["bca"] if sig else np.array([np.nan])
        print(f"{props.carrier_frequency:g} Hz stream: grand-average z by harmonic  " + "  ".join(cells))
        print(f"   summed BCA over clean harmonics with z>{z_min:g} {sig}: "
              f"mean {np.nanmean(summed):.3f} uV")
    return significant


def plot_bca_topomaps(
    experiment: Experiment[ConditionProperties, TopomapSSVEPRecording],
) -> None:
    """Group-mean BCA per channel at the oddball and at the carrier frequency, per condition.

    Earclips are not shown. Pz is not shown either: it is the recording
    reference, so its value is minus the reference average, not a measured site.
    """
    conds = _by_carrier(experiment)
    fig, axs = plt.subplots(2, len(conds), figsize=(3.6 * len(conds), 6.5),
                            squeeze=False, label="bca-topomaps")
    for j, (props, view) in enumerate(conds):
        names = [n for n in view.channel_names() if n not in ("A1", "A2", "Pz")]
        sub = view.take_channels(names)
        ft = FrequencyTagging(sub)
        for i, freq in enumerate((float(props.target_frequency), float(props.carrier_frequency))):
            bca = np.nanmean(ft.channel_measures_at(freq)["bca"], axis=0)     # (C,)
            ax = axs[i, j]
            vlim = (min(0.0, float(np.nanmin(bca))), float(np.nanmax(bca)))
            im, _ = mne.viz.plot_topomap(bca, sub.mne_info(), axes=ax, show=False,
                                         cmap="viridis", contours=0, vlim=vlim)
            kind = "oddball" if i == 0 else "carrier"
            ax.set_title(f"{props.carrier_frequency:g} Hz stream: {freq:g} Hz ({kind})", fontsize=9)
            fig.colorbar(im, ax=ax, shrink=0.6, label=r"BCA ($\mu$V)")
    fig.suptitle("Group-mean baseline-corrected amplitude (56 s FFT, screened average reference)",
                 fontsize=10)
    fig.tight_layout()


def plot_amplitude_snr_spectra(
    experiment: Experiment[ConditionProperties, SSVEPRecording],
    roi: List[str] = ODDBALL_ROI,
    fmin: float = 1.0,
    fmax: float = 37.0,
) -> None:
    """Amplitude SNR of the grand-average ROI spectrum, one panel per condition.

    Red lines: clean oddball harmonics. Blue lines: carrier and its multiples.
    """
    conds = _by_carrier(experiment)
    fig, axs = plt.subplots(1, len(conds), figsize=(3.8 * len(conds), 3.2), sharey=True,
                            squeeze=False, label="amplitude-snr-spectra")
    for ax, (props, view) in zip(axs[0], conds):
        ft = FrequencyTagging(view.take_channels(roi))
        roi_spec = ft.roi_spectrum()
        grand = roi_spec[~np.isnan(roi_spec[:, 0])].mean(axis=0)
        freqs = ft.frequencies()
        edge = int(np.max(np.abs(ft.offsets)))
        k = np.arange(edge, len(freqs) - edge)
        snr = grand[k] / grand[k[:, None] + ft.offsets].mean(axis=1)
        ax.plot(freqs[k], snr, lw=0.6, color="0.3")
        for f, clean in oddball_harmonics(float(props.target_frequency),
                                          float(props.carrier_frequency), fmax):
            ax.axvline(f, color="tab:red" if clean else "tab:blue", lw=1.0, alpha=0.4, zorder=0)
        ax.set_xlim(fmin, fmax)
        ax.set_title(f"{props.carrier_frequency:g} Hz stream (n={len(roi_spec)})", fontsize=9)
        ax.set_xlabel("Frequency [Hz]")
    axs[0, 0].set_ylabel(f"Amplitude SNR, {'+'.join(roi)} grand average")
    fig.suptitle("Red: clean oddball harmonics. Blue: carrier and its multiples", fontsize=9)
    fig.tight_layout()
