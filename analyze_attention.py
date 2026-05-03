"""Analyses specific to attention paradigms.

These analyses assume an `Experiment[AttentionFrequency, Recording[AttentionFrequency]]`
where each condition corresponds to attention directed at a single flicker
frequency, and every subject's recording contains all stimulus frequencies
simultaneously (only the attentional set differs across conditions).
"""

from typing import Dict, List, Tuple

import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import sem

from analysis import Spectral
from core_types import AttentionFrequency
from interfaces import Experiment, Recording


def attentional_snr_modulation(
    experiment: Experiment[AttentionFrequency, Recording[AttentionFrequency]],
    electrodes: List[str],
) -> Dict[AttentionFrequency, Tuple[float, float]]:
    """Per-frequency: SNR(attended) - SNR(unattended), averaged over subjects.

    For each attention frequency f present in the experiment:
      - SNR_attended(f, subj)   = SNR at f in the condition where subj attended to f.
      - SNR_unattended(f, subj) = SNR at f averaged over the conditions where
                                  subj attended to anything other than f.
      - diff(f, subj) = SNR_attended(f, subj) - SNR_unattended(f, subj).

    Returns {key: (mean_diff, sem_diff)} where mean and SEM are across the
    subjects who participated in every condition.
    """
    keys = list(experiment.conditions().keys())
    common_subjects = list(experiment.filter_subjects(keys))
    if not common_subjects:
        return {}

    out: Dict[AttentionFrequency, Tuple[float, float]] = {}
    for attended_key in keys:
        f = float(attended_key.frequency)
        diffs: List[float] = []
        for subj in common_subjects:
            attended_view = subj[attended_key].take_channels(electrodes)
            attended_snr, _ = Spectral(attended_view).snr_at(f)

            other_snrs: List[float] = []
            for other_key in keys:
                if other_key == attended_key:
                    continue
                other_view = subj[other_key].take_channels(electrodes)
                snr, _ = Spectral(other_view).snr_at(f)
                other_snrs.append(snr)

            diffs.append(attended_snr - float(np.mean(other_snrs)))

        diffs_arr = np.asarray(diffs)
        out[attended_key] = (
            float(np.mean(diffs_arr)),
            float(sem(diffs_arr)) if len(diffs_arr) > 1 else 0.0,
        )
    return out


def plot_attentional_snr_modulation(
    experiment: Experiment[AttentionFrequency, Recording[AttentionFrequency]],
    electrodes: List[str],
) -> None:
    """Bar plot of `attentional_snr_modulation` with SEM error bars."""
    modulation = attentional_snr_modulation(experiment, electrodes)
    if not modulation:
        print("plot_attentional_snr_modulation: no subjects participated in all conditions.")
        return

    sorted_keys = sorted(modulation.keys(), key=lambda k: float(k.frequency))
    means = [modulation[k][0] for k in sorted_keys]
    sems = [modulation[k][1] for k in sorted_keys]
    n_subjects = len(list(experiment.filter_subjects(list(experiment.conditions().keys()))))

    fig, ax = plt.subplots(figsize=(7, 5), label="attentional-snr-modulation")
    x = np.arange(len(sorted_keys))
    ax.bar(x, means, yerr=sems, capsize=5, color="steelblue", edgecolor="navy")
    ax.axhline(0, color="black", linewidth=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{k.frequency:.0f} Hz" for k in sorted_keys])
    ax.set_xlabel("Attention frequency")
    ax.set_ylabel("SNR(attended) - SNR(unattended)")
    ax.set_title(
        f"Attentional modulation of SNR (electrodes: {electrodes})\n"
        f"Mean ± SEM across {n_subjects} subjects"
    )
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
