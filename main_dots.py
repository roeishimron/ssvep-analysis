"""POC entry point for the dot-experiments single-frequency attention paradigm.

Demonstrates that the same Study/Recording machinery used for the SSVEP
paradigm works on a non-SSVEP paradigm purely via the typed key:
- the only new type is AttentionFrequency (sibling to ConditionProperties);
- no SSVEP-specific subclasses are needed;
- analyses that need SSVEP target/carrier (e.g. SSVEPAnalysis) would be a
  static type error here — only paradigm-agnostic Spectral applies.

Run:
  python main_dots.py ./dot-experiments
"""

import os
import sys

import matplotlib.pyplot as plt

from analysis import Spectral
from analyze_spectrum import plot_snr_spectra_overlay
from core_types import AttentionFrequency
from experiments import dot_experiments


TARGET_ELECTRODES = ["O1", "O2"]


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python main_dots.py <experiment_folder>")
        print("Example: python main_dots.py ./dot-experiments")
        sys.exit(1)

    root_dir = sys.argv[1]
    if not os.path.isdir(root_dir):
        print(f"Error: {root_dir} is not a directory")
        sys.exit(1)

    print(f"Loading dot-experiments from {root_dir}...")
    exp = dot_experiments(root_dir)

    print(f"Found {len(exp.subjects())} subjects across "
          f"{len(exp.conditions())} conditions.")

    print("\nGroup-aggregated SNR at the attention frequency, per condition:")
    for key, view in exp.conditions().items():
        spectral = Spectral(view.take_channels(TARGET_ELECTRODES))
        snr_mean, snr_sem = spectral.snr_at(float(key.frequency))
        print(f"  {key}: SNR @ {key.frequency}Hz = {snr_mean:.3f} ± {snr_sem:.3f}")

    print("\nPer-subject SNR at the attention frequency:")
    for name, subject in exp.subjects().items():
        per_condition = []
        for key, view in subject.conditions().items():
            spectral = Spectral(view.take_channels(TARGET_ELECTRODES))
            snr_mean, _ = spectral.snr_at(float(key.frequency))
            per_condition.append(f"{key.frequency:.0f}Hz: {snr_mean:.2f}")
        print(f"  {name:<22s}  {'  '.join(per_condition)}")

    print("\nPlotting overlaid SNR spectra by condition...")
    plot_snr_spectra_overlay(
        exp, fmin=1.0, fmax=49.0, electrodes=TARGET_ELECTRODES,
        label_func=lambda k: f"{k.frequency:.0f} Hz",
    )
    plt.show()


if __name__ == "__main__":
    main()
