import sys
import os
import matplotlib.pyplot as plt
import numpy as np
from core import Study
from core_types import ConditionProperties
from loader import StudyLoader
from analyze_spectrum import analyze_spectrum, plot_snrs, plot_snr_comparison

def main():
    if len(sys.argv) < 2:
        print("Usage: python main.py <experiment_folder>")
        print("Example: python main.py ./experiments_hebrew_vs_mirror")
        sys.exit(1)

    root_dir = sys.argv[1]
    if not os.path.isdir(root_dir):
        print(f"Error: {root_dir} is not a directory")
        sys.exit(1)

    print(f"Loading study from {root_dir}...")
    
    # 1. Initialize Loader and Study
    loader = StudyLoader()
    study = Study(loader.load(root_dir))

    # 2. Identify all conditions present in the study
    # We can get these from the blobs
    conditions = list(study._blobs.keys())
    if not conditions:
        print("No conditions found in the provided folder.")
        sys.exit(0)

    print(f"Found {len(conditions)} conditions.")

    # 3. For each condition, plot aggregate analysis
    # We'll use a specific electrode subset if available, otherwise all
    # Standard SSVEP electrodes of interest for this project:
    TARGET_ELECTRODES = ["T5"]

    for props in conditions:
        print(f"\nAnalyzing condition: {props}")
        
        # Get the aggregate view (average trials, promote subjects to trials)
        try:
            view = study.get_condition(props)
        except KeyError:
            continue

        # Plot 1: Spectrum (SNR and Amplitudes)
        # Restrict to target electrodes for the spectrum plots to reduce noise
        spectrum_view = view.restrict_electrodes(TARGET_ELECTRODES)
        
        # Determine frequency range for plotting: [0.5, target*2 + 5]
        fmin = 0.5
        fmax = props.carrier_frequency + 5
        
        analyze_spectrum(spectrum_view, fmin, fmax)

        # Plot 2: Topography (Spatial distribution of SNR at harmonics)
        # Topomap needs all electrodes
        plot_snrs(view, view.blob.raw_info)

    # 4. Global Analysis: SNR Comparison across conditions
    print("\nPlotting SNR comparison across conditions for T5...")
    plot_snr_comparison(study, TARGET_ELECTRODES)

    print("\nDisplaying plots...")
    plt.show()

if __name__ == "__main__":
    main()
