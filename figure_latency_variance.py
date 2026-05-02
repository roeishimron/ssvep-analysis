"""
Per-subject circular SD of processing-time, split by condition.

For every (subject, condition) pair, compute the circular SD of trial-to-trial
carrier-phase jitter and convert to ms via σ_ms = √(−2 ln R)·1000/(2π·f_carrier).
Plot one histogram per carrier frequency on a shared axis.

Run:
  MPLBACKEND=Agg python figure_latency_variance.py ./experiments_hebrew_vs_mirror
"""
from __future__ import annotations

import os
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from analysis import SSVEPAnalysis
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
            _, sd_ms = SSVEPAnalysis(v).processing_time_summary()
            sd = float(sd_ms[0])
            if np.isnan(sd):
                continue
            by_cond[key].append(sd)
    return {p: np.asarray(v) for p, v in by_cond.items()}


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

    axes[-1].set_xlabel("Circular SD of trial-to-trial latency jitter (ms)")
    fig.suptitle("Per-subject latency jitter (circular SD), by condition",
                 weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return fig


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python figure_latency_variance.py <experiment_folder>")
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
