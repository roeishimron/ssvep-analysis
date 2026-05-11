"""POC entry point for attention experiments.

Two modes, sharing the same analysis pipeline:

  python main_dots.py <experiment_folder>
      Single-frequency dot paradigm: folder names encode the
      AttentionFrequency (folder-per-condition layout).

  python main_dots.py <experiment_folder> <metadata_folder>
      Segmented attention-color paradigm: one JSON per subject in
      <metadata_folder> describes that subject's per-trial layout
      (filename stem must match the subject's name from the EDF, e.g.
      "alice.json" for alice_raw.edf). Conditions are AttentionColor(+/-1).

Below the experiment-construction step, the rest of the script is identical
in both modes — only the per-key frequency-of-interest and label are
parameterised. The common analysis lives in `_run_common` so each branch
hands it a K-narrow `exp`, keeping pyright happy.
"""

import json
import os
import sys
from pathlib import Path
from typing import Callable, Hashable, TypeVar

import matplotlib.pyplot as plt

from analysis import Spectral
from analyze_attention import (
    attentional_snr_modulation, plot_attentional_snr_modulation,
)
from analyze_spectrum import plot_snr_spectra_overlay
from attention_metadata import AttentionColorParser
from core import Study
from experiments import aggregate_by_metadata, dot_experiments
from interfaces import Experiment, Recording
from loader import StudyLoader


TARGET_ELECTRODES = ["Pz", "O1", "O2"]
STIMULUS_FREQUENCY = 10.0   # Hz — used in the AttentionColor mode (the
                            # AttentionColor key carries no frequency, so the
                            # flicker rate is supplied as a script constant).

K = TypeVar("K", bound=Hashable)


def _run_common(
    exp: Experiment[K, Recording[K]],
    freq_of: Callable[[K], float],
    label_of: Callable[[K], str],
) -> None:
    """K-generic analysis: per-condition SNR, per-subject SNR, spectra overlay."""
    print(f"Found {len(exp.subjects())} subjects across "
          f"{len(exp.conditions())} conditions.")

    print("\nGroup-aggregated SNR per condition:")
    for key, view in exp.conditions().items():
        spectral = Spectral(view.take_channels(TARGET_ELECTRODES))
        f = freq_of(key)
        snr_mean, snr_sem = spectral.snr_at(f)
        print(f"  {label_of(key)}: SNR @ {f:g}Hz = {snr_mean:.3f} ± {snr_sem:.3f}")

    print("\nPer-subject SNR:")
    for name, subject in exp.subjects().items():
        per_condition = []
        for key, view in subject.conditions().items():
            spectral = Spectral(view.take_channels(TARGET_ELECTRODES))
            snr_mean, _ = spectral.snr_at(freq_of(key))
            per_condition.append(f"{label_of(key)}: {snr_mean:.2f}")
        print(f"  {name:<22s}  {'  '.join(per_condition)}")

    print("\nPlotting overlaid SNR spectra by condition...")
    plot_snr_spectra_overlay(
        exp, fmin=1.0, fmax=49.0, electrodes=TARGET_ELECTRODES,
        label_func=label_of,
    )


def main() -> None:
    if not 2 <= len(sys.argv) <= 3:
        print("Usage: python main_dots.py <experiment_folder> [<metadata_folder>]")
        sys.exit(1)

    root_dir = sys.argv[1]
    if not os.path.isdir(root_dir):
        print(f"Error: {root_dir} is not a directory")
        sys.exit(1)

    metadata_dir = sys.argv[2] if len(sys.argv) == 3 else None
    if metadata_dir is not None and not os.path.isdir(metadata_dir):
        print(f"Error: {metadata_dir} is not a directory")
        sys.exit(1)

    if metadata_dir is None:
        print(f"Loading dot-experiments from {root_dir}...")
        exp_freq = dot_experiments(root_dir)
        _run_common(
            exp_freq,
            lambda k: float(k.frequency),
            lambda k: f"{k.frequency:.0f} Hz",
        )

        # AttentionFrequency-specific: each subject participated in multiple
        # flicker-rate conditions and we contrast attended vs unattended at
        # each frequency. Not meaningful in the segmented two-color paradigm.
        print("\nAttentional SNR modulation per frequency (attended - unattended):")
        for key, (mean_diff, sem_diff) in attentional_snr_modulation(
            exp_freq, TARGET_ELECTRODES,
        ).items():
            print(f"  {key.frequency:.0f} Hz:  ΔSNR = {mean_diff:+.3f} ± {sem_diff:.3f}")

        print("Plotting attentional SNR modulation...")
        plot_attentional_snr_modulation(exp_freq, TARGET_ELECTRODES)
    else:
        print(f"Loading segmented recordings from {root_dir} with {metadata_dir}...")
        # Read every subject's metadata once. Filename stem (without
        # extension) must equal the subject_name produced by load_folder.
        metadata_by_subject = {
            p.stem: p.read_text() for p in Path(metadata_dir).glob("*.json")
        }
        if not metadata_by_subject:
            print(f"Error: no *.json metadata files in {metadata_dir}")
            sys.exit(1)

        durations = [
            (float(sum(s["steady_state_quantas"] for s in t) + len(t)))*2
            for md in metadata_by_subject.values()
            for t in json.loads(md)
        ]

        def metadata_for(subject_name: str) -> str:
            if subject_name not in metadata_by_subject:
                raise KeyError(
                    f"no metadata file for subject '{subject_name}' in {metadata_dir} "
                    f"(have: {sorted(metadata_by_subject)})"
                )
            return metadata_by_subject[subject_name]

        stream = StudyLoader().load_folder(root_dir, durations=durations)
        exp_color = Study(
            aggregate_by_metadata(stream, metadata_for, AttentionColorParser(sample_rate=300.0)),
            min_trials=3,
        )
        _run_common(
            exp_color,
            lambda _: STIMULUS_FREQUENCY,
            lambda k: f"attend={k.color:+d}",
        )

    plt.show()


if __name__ == "__main__":
    main()
