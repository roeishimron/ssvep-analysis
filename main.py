import sys
import os
import matplotlib.pyplot as plt
import numpy as np
from core import Study
from core_types import ConditionProperties
from loader import StudyLoader
from analyze_spectrum import (
    analyze_spectrum, plot_snrs, plot_snr_comparison, CarrierComparisonAnalysis,
    plot_latency_mean_vs_snr_slope
)
import matplot2tikz

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
    TARGET_ELECTRODES = ["T5", "T6"]
    V1_ELECTRODES = ["O1", "O2"]

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
        
        analyze_spectrum(spectrum_view, fmin, fmax, props.carrier_frequency == 10)

        # Plot 2: Topography (Spatial distribution of SNR at harmonics)
        # Topomap needs all electrodes
        plot_snrs(view, view.blob.raw_info)

    # 4. Global Analysis: SNR Comparison across conditions
    print("\nPlotting SNR comparison across conditions for T5...")
    plot_snr_comparison(study, TARGET_ELECTRODES)

    # 5. Subject-wise Carrier Comparison (10Hz vs 15Hz)
    print("\nPlotting Subject-wise SNR Comparison (10Hz vs 15Hz carrier) for T5...")
    try:
        comparison = CarrierComparisonAnalysis(study, [10, 15], TARGET_ELECTRODES)
        comparison.plot()
    except ValueError as e:
        print(f"Skipping carrier comparison: {e}")

    # 6. Phase Latency Analysis pooled across carrier groups
    LATENCY_CARRIERS = [10.0, 15.0, 20.0]
    print(f"\nAnalyzing Phase Latency pooled across carriers {LATENCY_CARRIERS}Hz...")

    # Only include subjects who participated in EVERY latency-carrier condition
    latency_requirements = {
        ConditionProperties(np.float64(5), np.float64(lc)) for lc in LATENCY_CARRIERS
    }
    common_subjects = list(study.filter_subjects(latency_requirements))
    print(f"  {len(common_subjects)} subjects common to all latency carriers")

    pooled_latencies = []
    for subject in common_subjects:
        for lc in LATENCY_CARRIERS:
            view = subject[
                ConditionProperties(np.float64(5), np.float64(lc))
            ].restrict_electrodes(TARGET_ELECTRODES + V1_ELECTRODES)
            lat = np.asarray(view.calculate_processing_time()).flatten() * 1000
            pooled_latencies.append(lat)

    if pooled_latencies:
        latencies = np.concatenate(pooled_latencies)
        plt.figure(figsize=(8, 6), label="phase-latency-pooled")
        plt.hist(latencies, bins=6, edgecolor='black', alpha=0.7)
        mean_lat = np.mean(latencies)
        plt.axvline(float(mean_lat), color='red', linestyle='--', label=f'Mean: {mean_lat:.1f}ms')
        plt.title(f"Distribution of Processing Time (carriers: {LATENCY_CARRIERS}Hz)")
        plt.xlabel("Latency (ms)")
        plt.ylabel("Count (Trials x Subjects x Carriers)")
        plt.legend()
        plt.grid(axis='y', alpha=0.3)

    # 7. Latency mean vs. SNR Slope Analysis
    print("\nAnalyzing Latency mean vs. SNR Slope...")
    try:
        # Using 10, 15 Hz carriers for slope, pooled across the same carriers for latency
        plot_latency_mean_vs_snr_slope(
            study, [10, 15, 20], LATENCY_CARRIERS, TARGET_ELECTRODES, V1_ELECTRODES
        )
    except Exception as e:
        print(f"Could not perform Latency mean vs SNR Slope analysis: {e}")

    print("Saving plots...")
    os.makedirs("figures", exist_ok=True)
    for i in plt.get_fignums():
        fig = plt.figure(i)
        label = fig.get_label()
        if not label:
            label = f"figure_{i}"
        print(f"Saving {label}...")
        try:
            # fig.suptitle("")
            matplot2tikz.clean_figure(fig)
            matplot2tikz.save(f"figures/{label}.tex", figure=fig)
        except Exception as e:
            print(f"Failed to save {label}: {e}")

    print("\nDisplaying plots...")
    plt.show()

if __name__ == "__main__":
    main()
