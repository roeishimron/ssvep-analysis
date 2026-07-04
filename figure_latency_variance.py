"""
Per-subject circular sd of processing-time, split by condition.

For every (subject, condition) pair, compute the circular sd of trial-to-trial
Plot one histogram per carrier frequency on a shared axis.

Run:
  MPLBACKEND=Agg python figure_latency_sd.py ./experiments_hebrew_vs_mirror
"""
from __future__ import annotations

import os
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from analysis import Spectral, SSVEPAnalysis
from core import Study
from core_types import ConditionProperties
from interfaces import Experiment, SSVEPRecording
from loader import StudyLoader

TARGET_ELECTRODES = ["T5", "T6"]
V1_ELECTRODES = ["O1", "O2"]

CONDITION_COLORS = {
    10.0: "#1976D2",
    15.0: "#FB8C00",
    20.0: "#388E3C",
}


def per_subject_sds_by_condition(
    study: Experiment[ConditionProperties, SSVEPRecording],
) -> dict[ConditionProperties, np.ndarray]:
    """For each condition, return an array of per-subject circular SDs (ms)."""
    by_cond: dict[ConditionProperties, list[float]] = defaultdict(list)
    for subject in study.subjects().values():
        for key, view in subject.conditions().items():
            v = view.take_channels(TARGET_ELECTRODES + V1_ELECTRODES)
            _, var_ms = SSVEPAnalysis(v).processing_time_summary()
            var = float(var_ms[0])
            if np.isnan(var):
                continue
            by_cond[key].append(var)
    return {p: np.asarray(v) for p, v in by_cond.items()}


def per_subject_target_signal_by_condition(
    study: Experiment[ConditionProperties, SSVEPRecording],
    target_frequency: float = 5.0,
) -> dict[ConditionProperties, np.ndarray]:
    """For each condition, per-subject MAX SNR over T5/T6 at the target frequency.

    "Signal" is the trial-averaged SNR at `target_frequency` Hz; MAX picks the
    stronger of the two target electrodes per subject.
    """
    by_cond: dict[ConditionProperties, list[float]] = defaultdict(list)
    for subject in study.subjects().values():
        for key, view in subject.conditions().items():
            spectral = Spectral(view.take_channels(TARGET_ELECTRODES))
            idx = spectral._closest_index(target_frequency)
            per_channel = spectral.snr_topomap()[:, idx]  # (C,) over T5/T6
            by_cond[key].append(float(np.max(per_channel)))
    return {p: np.asarray(v) for p, v in by_cond.items()}


def per_subject_carrier_signal_by_condition(
    study: Experiment[ConditionProperties, SSVEPRecording],
) -> dict[ConditionProperties, np.ndarray]:
    """For each condition, per-subject MAX SNR over O1/O2 at the carrier frequency.

    "Signal" is the trial-averaged SNR at the condition's carrier frequency; MAX
    picks the stronger of the two V1 electrodes per subject.
    """
    by_cond: dict[ConditionProperties, list[float]] = defaultdict(list)
    for subject in study.subjects().values():
        for key, view in subject.conditions().items():
            spectral = Spectral(view.take_channels(V1_ELECTRODES))
            idx = spectral._closest_index(float(key.carrier_frequency))
            per_channel = spectral.snr_topomap()[:, idx]  # (C,) over O1/O2
            by_cond[key].append(float(np.max(per_channel)))
    return {p: np.asarray(v) for p, v in by_cond.items()}


def _build_signal_by_condition_figure(
    by_cond: dict[ConditionProperties, np.ndarray],
    xlabel: str,
    suptitle: str,
    figure_label: str,
) -> plt.Figure:
    """One histogram per carrier of a per-subject scalar signal, shared x-axis."""
    sorted_props = sorted(by_cond.keys(), key=lambda p: p.carrier_frequency)
    n = len(sorted_props)

    all_vals = np.concatenate(list(by_cond.values())) if by_cond else np.array([0.0])
    bins = np.arange(0, np.ceil(float(np.max(all_vals))) + 1, 1.0)

    fig, axes = plt.subplots(
        n, 1, figsize=(8, 2.4 * n), sharex=True, label=figure_label,
    )
    if n == 1:
        axes = [axes]

    for ax, props in zip(axes, sorted_props):
        vals = by_cond[props]
        carrier = float(props.carrier_frequency)
        color = CONDITION_COLORS.get(carrier, "#1E88E5")
        ax.hist(vals, bins=bins, edgecolor="black", color=color, alpha=0.8)
        mean_v = float(np.mean(vals))
        median_v = float(np.median(vals))
        ax.axvline(mean_v, color="#D81B60", linestyle="--", lw=1.4,
                   label=f"mean = {mean_v:.2f}")
        ax.axvline(median_v, color="#2E7D32", linestyle=":", lw=1.4,
                   label=f"median = {median_v:.2f}")
        ax.set_title(
            f"{carrier:.0f} Hz carrier — N = {len(vals)} subjects",
            fontsize=11, weight="bold", loc="left",
        )
        ax.set_ylabel("Subject count")
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)

    axes[-1].set_xlabel(xlabel)
    fig.suptitle(suptitle, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return fig


def build_target_signal_figure(
    by_cond: dict[ConditionProperties, np.ndarray],
    target_frequency: float = 5.0,
) -> plt.Figure:
    """One histogram per carrier of the per-subject MAX[T5,T6] target SNR."""
    return _build_signal_by_condition_figure(
        by_cond,
        xlabel=f"MAX[T5,T6] SNR at {target_frequency:g} Hz",
        suptitle=f"Per-subject target-response signal (MAX[T5,T6] SNR at {target_frequency:g} Hz), by carrier",
        figure_label="target-signal-max-t5t6-by-condition",
    )


def build_carrier_signal_figure(
    by_cond: dict[ConditionProperties, np.ndarray],
) -> plt.Figure:
    """One histogram per carrier of the per-subject MAX[O1,O2] carrier SNR."""
    return _build_signal_by_condition_figure(
        by_cond,
        xlabel="MAX[O1,O2] SNR at carrier frequency",
        suptitle="Per-subject carrier-response signal (MAX[O1,O2] SNR at carrier), by carrier",
        figure_label="carrier-signal-max-o1o2-by-condition",
    )


def per_subject_between_carrier_spread(
    study: Experiment[ConditionProperties, SSVEPRecording],
    carriers: list[float],
) -> np.ndarray:
    """Per-subject SD (ms) of the processing-time mean measured across `carriers`.

    For each subject with all requested carriers, measure the mean processing
    time at each carrier and return the standard deviation across those
    measurements — i.e. how much the latency estimate disagrees between carriers.
    """
    props_list = [ConditionProperties(np.float64(5), np.float64(c)) for c in carriers]
    spreads: list[float] = []
    for subject in study.filter_subjects(set(props_list)):
        per_carrier_means: list[float] = []
        for props in props_list:
            view = subject[props].take_channels(TARGET_ELECTRODES + V1_ELECTRODES)
            mean_ms, _ = SSVEPAnalysis(view).processing_time_summary()
            per_carrier_means.append(float(mean_ms[0]))
        spreads.append(float(np.std(per_carrier_means)))
    return np.asarray(spreads)


def build_between_carrier_figure(
    spreads: np.ndarray,
    carriers: list[float],
) -> plt.Figure:
    """Histogram of the per-subject between-carrier processing-time spread."""
    carrier_label = ", ".join(f"{c:g}" for c in carriers)
    fig, ax = plt.subplots(
        figsize=(8, 5), label="latency-between-carrier-spread",
    )
    bins = np.linspace(0, float(np.max(spreads)) * 1.05 + 1e-9, 12) if spreads.size else 12
    ax.hist(spreads, bins=bins, edgecolor="black", color="#6A1B9A", alpha=0.8)

    if spreads.size:
        mean_v = float(np.mean(spreads))
        median_v = float(np.median(spreads))
        ax.axvline(mean_v, color="#D81B60", linestyle="--", lw=1.4,
                   label=f"mean = {mean_v:.1f} ms")
        ax.axvline(median_v, color="#2E7D32", linestyle=":", lw=1.4,
                   label=f"median = {median_v:.1f} ms")
        ax.legend(fontsize=9)

    ax.set_title(
        f"Between-carrier spread of processing-time estimate — N = {spreads.size} subjects\n"
        f"(SD across {carrier_label} Hz carriers)",
        fontsize=11, weight="bold", loc="left",
    )
    ax.set_xlabel("SD of processing-time mean across carriers (ms)")
    ax.set_ylabel("Subject count")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


def build_figure(
    by_cond: dict[ConditionProperties, np.ndarray],
) -> plt.Figure:
    sorted_props = sorted(by_cond.keys(), key=lambda p: p.carrier_frequency)
    n = len(sorted_props)

    all_sds = np.concatenate(list(by_cond.values())) if by_cond else np.array([0.0])
    bins = np.linspace(0, float(np.max(all_sds)) * 1.05 + 1e-9, 12)

    fig, axes = plt.subplots(
        n, 1, figsize=(8, 2.4 * n), sharex=True,
        label="latency-circular-sd-by-condition",
    )
    if n == 1:
        axes = [axes]

    for ax, props in zip(axes, sorted_props):
        vals = by_cond[props]
        carrier = float(props.carrier_frequency)
        color = CONDITION_COLORS.get(carrier, "#1E88E5")
        ax.hist(vals, bins=bins, edgecolor="black", color=color, alpha=0.8)
        mean_v = float(np.mean(vals))
        median_v = float(np.median(vals))
        ax.axvline(mean_v, color="#D81B60", linestyle="--", lw=1.4,
                   label=f"mean = {mean_v:.1f} ms")
        ax.axvline(median_v, color="#2E7D32", linestyle=":", lw=1.4,
                   label=f"median = {median_v:.1f} ms")
        ax.set_title(
            f"{carrier:.0f} Hz carrier — N = {len(vals)} subjects",
            fontsize=11, weight="bold", loc="left",
        )
        ax.set_ylabel("Subject count")
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)

    axes[-1].set_xlabel("Circular sd of trial-to-trial latency jitter (ms)")
    fig.suptitle("Per-subject latency jitter (circular sd), by condition",
                 weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return fig


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python figure_latency_sd.py <experiment_folder>")
        sys.exit(1)
    root_dir = sys.argv[1]
    if not os.path.isdir(root_dir):
        print(f"Error: {root_dir} is not a directory")
        sys.exit(1)
    print(f"Loading study from {root_dir}...")

    loader = StudyLoader()
    study = Study(loader.load(root_dir))

    by_cond = per_subject_sds_by_condition(study)
    for props in sorted(by_cond.keys(), key=lambda p: p.carrier_frequency):
        v = by_cond[props]
        print(f"  {props}: N={len(v)}  mean={np.mean(v):.2f} ms  "
              f"median={np.median(v):.2f} ms  max={np.max(v):.2f} ms")

    fig = build_figure(by_cond)

    out_dir = Path("figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    png_path = out_dir / "latency-circular-sd-by-condition.png"
    pdf_path = out_dir / "latency-circular-sd-by-condition.pdf"
    fig.savefig(png_path, dpi=180, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    print(f"wrote {png_path}")
    print(f"wrote {pdf_path}")


if __name__ == "__main__":
    main()
