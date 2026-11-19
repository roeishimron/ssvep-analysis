"""Hebrew-vs-mirror frequency-tagging analysis.

Usage: python main.py <experiment_folder> [--legacy] [--no-show]

Default: subjects are recording sessions (pseudonymous IDs P01..), bad channels
are screened before a clean average reference, and the response is measured on
one 56 s FFT per stream (BCA, z and amplitude SNR against 20 neighbouring bins)
over fixed ROIs: T5+T6 for the oddball, O1+O2 for the carrier. Prints the
existence statistics and saves topographies and SNR spectra to figures/.

--legacy additionally runs the earlier figures: the 2 s-window power SNR plots
and the latency figures (on the 2.5-60 s epoch the latency code expects).
"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

from analysis import SSVEPAnalysis
from analyze_spectrum import (
    CARRIER_ROI, ODDBALL_ROI, CarrierComparisonAnalysis, analyze_spectrum,
    existence_report, harmonic_report, plot_amplitude_snr_spectra, plot_bca_topomaps,
    plot_latency_mean_vs_snr_at_carrier, plot_latency_mean_vs_snr_slope,
    plot_snr_comparison, plot_snrs, report_bad_channels,
)
from core import Study
from core_types import ConditionProperties
from figure_latency_variance import (
    build_between_carrier_figure, build_figure as build_sd_figure,
    per_subject_between_carrier_spread, per_subject_sds_by_condition,
)
from loader import StudyLoader


def run_frequency_tagging(root_dir: str) -> None:
    """Default analysis: full-length spectrum, fixed ROIs, existence statistics."""
    loader = StudyLoader(epoch_s=StudyLoader.SPECTRUM_EPOCH_S)
    study = Study(loader.load(root_dir))
    conditions = list(study.conditions().keys())
    if not conditions:
        print("No conditions found in the provided folder.")
        sys.exit(0)

    print()
    print(f"Subjects (sessions): {len(study.subjects())}; with all {len(conditions)} conditions: "
          f"{len(list(study.filter_subjects(set(conditions))))}; per condition: "
          + ", ".join(f"{p.carrier_frequency:g} Hz n={v.raw_data().shape[0]}"
                      for p, v in sorted(study.conditions().items(),
                                         key=lambda kv: kv[0].carrier_frequency)))
    print(report_bad_channels(loader.records))

    print(f"\n== Oddball response over {'+'.join(ODDBALL_ROI)} (56 s FFT, 20 neighbours) ==")
    existence_report(study, ODDBALL_ROI)
    print(f"\n== Carrier response over {'+'.join(CARRIER_ROI)} ==")
    existence_report(study, CARRIER_ROI, at_carrier=True)
    print(f"\n== Oddball harmonics over {'+'.join(ODDBALL_ROI)} ==")
    harmonic_report(study, ODDBALL_ROI)

    plot_bca_topomaps(study)
    plot_amplitude_snr_spectra(study, ODDBALL_ROI)


def run_legacy(root_dir: str) -> None:
    """Earlier figures: 2 s-window power SNR and the latency analyses."""
    study = Study(StudyLoader().load(root_dir))
    conditions = list(study.conditions().keys())

    for props in conditions:
        print(f"\nAnalyzing condition: {props}")
        # Group view (Subject axis kept), so bands are SEM across subjects.
        view = study.conditions()[props]
        fmin = 0.5
        fmax = props.carrier_frequency + 5
        analyze_spectrum(view.take_channels(ODDBALL_ROI), fmin, fmax, props.carrier_frequency == 10)
        plot_snrs(view)

    print("\nPlotting power-SNR comparison across conditions (legacy measure)...")
    plot_snr_comparison(study, ODDBALL_ROI)

    print("\nPlotting subject-wise power SNR at 5 Hz across carriers...")
    comparison = CarrierComparisonAnalysis(study, [10, 15, 20], ODDBALL_ROI)
    comparison.plot()

    LATENCY_CARRIERS = [10.0, 15.0, 20.0]
    print(f"\nAnalyzing Phase Latency pooled across carriers {LATENCY_CARRIERS}Hz...")
    latency_requirements = {
        ConditionProperties(np.float64(5), np.float64(lc)) for lc in LATENCY_CARRIERS
    }
    common_subjects = list(study.filter_subjects(latency_requirements))
    print(f"  {len(common_subjects)} subjects common to all latency carriers")

    pooled_latencies: list[float] = []
    for subject in common_subjects:
        for lc in LATENCY_CARRIERS:
            view = subject[
                ConditionProperties(np.float64(5), np.float64(lc))
            ].take_channels(ODDBALL_ROI + CARRIER_ROI)
            mean_ms, _ = SSVEPAnalysis(view).processing_time_summary()
            pooled_latencies.append(float(mean_ms[0]))

    if pooled_latencies:
        latencies = np.asarray(pooled_latencies)
        plt.figure(figsize=(8, 6), label="phase-latency-pooled")
        plt.hist(latencies, bins=6, edgecolor='black', alpha=0.7)
        mean_lat = np.mean(latencies)
        plt.axvline(float(mean_lat), color='red', linestyle='--', label=f'Mean: {mean_lat:.1f}ms')
        plt.title(f"Distribution of Processing Time (carriers: {LATENCY_CARRIERS}Hz)")
        plt.xlabel("Latency (ms)")
        plt.ylabel("Count (Subjects x Carriers)")
        plt.legend()
        plt.grid(axis='y', alpha=0.3)

    print("\nAnalyzing Latency mean vs. SNR Slope...")
    plot_latency_mean_vs_snr_slope(
        study, [10, 15, 20], LATENCY_CARRIERS, ODDBALL_ROI, CARRIER_ROI
    )

    print("\nAnalyzing Latency mean vs. SNR at 20 Hz...")
    plot_latency_mean_vs_snr_at_carrier(
        study, 20, LATENCY_CARRIERS, ODDBALL_ROI, CARRIER_ROI
    )

    print("\nPlotting per-subject latency circular-sd distribution by condition...")
    by_cond = per_subject_sds_by_condition(study)
    if by_cond:
        build_sd_figure(by_cond)

    print("\nPlotting between-carrier processing-time spread distribution...")
    spreads = per_subject_between_carrier_spread(study, LATENCY_CARRIERS)
    if spreads.size:
        build_between_carrier_figure(spreads, LATENCY_CARRIERS)
    # The MAX[T5,T6] / MAX[O1,O2] SNR histograms (figure_latency_variance) are no
    # longer drawn here: per-subject channel selection biases group numbers.
    # The fixed-ROI measures are in run_frequency_tagging.


def save_figures(out_dir: str = "figures") -> None:
    import matplot2tikz

    os.makedirs(out_dir, exist_ok=True)
    for i in plt.get_fignums():
        fig = plt.figure(i)
        label = fig.get_label() or f"figure_{i}"
        print(f"Saving {label}...")
        # Topomaps (and the new summary figures) are saved as PNG; the legacy
        # topo{carrier}.png names are what the LyX document includes.
        if "topomap" in label:
            for carrier in (10, 15, 20, 30):
                if f"-{carrier}.0-" in label:
                    fig.savefig(f"{out_dir}/topo{carrier}.png", dpi=150, bbox_inches="tight")
                    break
        if label in ("bca-topomaps", "amplitude-snr-spectra"):
            fig.savefig(f"{out_dir}/{label}.png", dpi=150, bbox_inches="tight")
            continue
        try:
            matplot2tikz.clean_figure(fig)
            matplot2tikz.save(f"{out_dir}/{label}.tex", figure=fig)
        except Exception as e:
            print(f"Failed to save {label}: {e}")


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    flags = {a for a in sys.argv[1:] if a.startswith("--")}
    if len(args) < 1:
        print(__doc__)
        sys.exit(1)

    root_dir = args[0]
    if not os.path.isdir(root_dir):
        print(f"Error: {root_dir} is not a directory")
        sys.exit(1)

    print(f"Loading study from {root_dir}...")
    run_frequency_tagging(root_dir)
    if "--legacy" in flags:
        print("\n== Legacy figures (2 s-window power SNR, latency) ==")
        run_legacy(root_dir)

    print("Saving plots...")
    save_figures()
    if "--no-show" not in flags:
        print("\nDisplaying plots...")
        plt.show()


if __name__ == "__main__":
    main()
