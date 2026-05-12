"""Super-thin POC: SNR spectrum from one folder of EDFs.

The folder is treated as a single condition (label = folder basename); every
EDF in it is a subject, loaded with the existing trigger-based epoching.

Run:
  python main_snr.py <folder> [<duration_s>]

`duration_s` is forwarded to `StudyLoader.load_folder` as the per-trial
epoch tmax (default 60.0).
"""

import os
import sys

import matplotlib.pyplot as plt

from analyze_spectrum import plot_snr_spectra_overlay
from core import Study
from experiments import compose_folders
from loader import StudyLoader


TARGET_ELECTRODES = ["Pz", "O1", "O2"]


def main() -> None:
    if not 2 <= len(sys.argv) <= 3:
        print("Usage: python main_snr.py <folder> [<duration_s>]")
        sys.exit(1)

    folder = sys.argv[1]
    if not os.path.isdir(folder):
        print(f"Error: {folder} is not a directory")
        sys.exit(1)

    duration_s = float(sys.argv[2]) if len(sys.argv) == 3 else 60.0

    label = os.path.basename(folder.rstrip("/")) or folder
    loader = StudyLoader()
    study = Study(compose_folders(
        [(label, folder)],
        lambda f: loader.load_folder(f, duration=duration_s),
    ))

    plot_snr_spectra_overlay(study, fmin=1.0, fmax=49.0, electrodes=TARGET_ELECTRODES)
    plt.show()


if __name__ == "__main__":
    main()
