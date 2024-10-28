import matplotlib.pyplot as plt
import numpy as np
import mne

FILENAME = "roei_font_color_arabic"
raw = mne.io.read_raw_edf(
    f"/media/lab-server/roei.shimron/ssvep/experiments/{FILENAME}_raw.edf", preload=True, verbose=False)

print("read data")

raw.rename_channels(lambda s: s.replace("EEG ", "").replace("-Pz", ""), False)

raw.drop_channels(['Ax', 'Ay', 'Az'])
raw.drop_channels(['X3:S1', 'X2:S2', 'X1:S3', 'Event', 'CM'])
raw.set_eeg_reference()

# raw.set_montage(montage='standard_1020')

raw.pick(["T3", "T5", "Trigger"])
# Set common average reference

raw.filter(l_freq=1, h_freq=None, fir_design="firwin",
           verbose=False, n_jobs=-1)


# detect events and edit
events = mne.find_events(raw, stim_channel="Trigger", mask=255)

# Handle too close events:
diffs = np.diff(events[:,0], append=raw.last_samp)
valids = np.argwhere(diffs > 1000).flatten()
events = events[valids]
print(f"found {len(events)} events")

# Construct epochs
tmin, tmax = 5, 58  # in s
epochs = mne.Epochs(
    raw,
    events=events,
    tmin=tmin,
    tmax=tmax,
    baseline=None,
    verbose=False,
)

# Calculate PSD
fmin = 0.5
fmax = 15.0
sfreq = 300

spectrum = epochs.compute_psd(
    "welch",
    n_fft=int(sfreq * (tmax - tmin)),
    fmin=fmin,
    fmax=fmax,
    verbose=False,
    # n_overlap=2048,
    # n_per_seg=int(sfreq)*50,
    n_jobs=-1,
)
psds, freqs = spectrum.get_data(return_freqs=True)

print("got psds")


def snr_spectrum(psd, noise_n_neighbor_freqs=1, noise_skip_neighbor_freqs=1):
    """Compute SNR spectrum from PSD spectrum using convolution.

    Parameters
    ----------
    psd : ndarray, shape ([n_trials, n_channels,] n_frequency_bins)
        Data object containing PSD values. Works with arrays as produced by
        MNE's PSD functions or channel/trial subsets.
    noise_n_neighbor_freqs : int
        Number of neighboring frequencies used to compute noise level.
        increment by one to add one frequency bin ON BOTH SIDES
    noise_skip_neighbor_freqs : int
        set this >=1 if you want to exclude the immediately neighboring
        frequency bins in noise level calculation

    Returns
    -------
    snr : ndarray, shape ([n_trials, n_channels,] n_frequency_bins)
        Array containing SNR for all epochs, channels, frequency bins.
        NaN for frequencies on the edges, that do not have enough neighbors on
        one side to calculate SNR.
    """
    # Construct a kernel that calculates the mean of the neighboring
    # frequencies
    averaging_kernel = np.concatenate(
        (
            np.ones(noise_n_neighbor_freqs),
            np.zeros(2 * noise_skip_neighbor_freqs + 1),
            np.ones(noise_n_neighbor_freqs),
        )
    )
    averaging_kernel /= averaging_kernel.sum()

    # Calculate the mean of the neighboring frequencies by convolving with the
    # averaging kernel.
    mean_noise = np.apply_along_axis(
        lambda psd_: np.convolve(psd_, averaging_kernel, mode="valid"), axis=-1, arr=psd
    )

    # The mean is not defined on the edges so we will pad it with nas. The
    # padding needs to be done for the last dimension only so we set it to
    # (0, 0) for the other ones.
    edge_width = noise_n_neighbor_freqs + noise_skip_neighbor_freqs
    pad_width = [(0, 0)] * (mean_noise.ndim - 1) + [(edge_width, edge_width)]
    mean_noise = np.pad(mean_noise, pad_width=pad_width,
                        constant_values=np.nan)

    return psd / mean_noise


# Average every 3 bins
NOISE_NEIGHBORS = 4
NOISE_SKIP = 2
snrs = snr_spectrum(psds, noise_n_neighbor_freqs=NOISE_NEIGHBORS,
                    noise_skip_neighbor_freqs=NOISE_SKIP)

print("got snr")
fig, axes = plt.subplots(2, 1, sharex="all", sharey="none", figsize=(8, 5))


psds_plot = 10 * np.log10(psds)
psds_mean = psds_plot.mean(axis=(0, 1))
psds_std = psds_plot.std(axis=(0, 1))
axes[0].plot(freqs, psds_mean, color="b")
axes[0].fill_between(
    freqs, psds_mean - psds_std, psds_mean + psds_std, color="b", alpha=0.1
)
axes[0].set(title="PSD spectrum", ylabel="Power Spectral Density [dB]")

# SNR spectrum
snr_mean = snrs.mean(axis=(0, 1))
snr_std = snrs.std(axis=(0, 1))

axes[1].plot(freqs, snr_mean, color="r")
axes[1].fill_between(
    freqs, snr_mean - snr_std, snr_mean + snr_std, color="r", alpha=0.1
)
axes[1].set(
    title="SNR spectrum",
    xlabel="Frequency [Hz]",
    ylabel="SNR",
    xlim=[fmin, fmax],
)
fig.savefig(f"{FILENAME}-graph.png")


# Find corresponding indices using mne.pick_types()

# find index of frequency bin closest to stimulation frequency

TARGET_FREQ = 5.88/5*4
RANGE_OF_TOPO_SEARCH = int(0.01 * sfreq)
target_center = int(np.argmin(np.abs(freqs - TARGET_FREQ)))

range_start = np.max([target_center - RANGE_OF_TOPO_SEARCH, NOISE_NEIGHBORS + NOISE_SKIP])
range_end = target_center + RANGE_OF_TOPO_SEARCH

# adding in the end because the arg is relative to the array
i_bin_target_hz = np.argmax(snr_mean[range_start:range_end]) + range_start


print(f"looking at freq {freqs[i_bin_target_hz]
                         } with value of {snr_mean[i_bin_target_hz]}")


# get average SNR at 1 Hz for ALL channels
snrs_stim_hz = snrs[:, :, i_bin_target_hz]
snrs_stim_hz_chaverage = snrs_stim_hz.mean(axis=0)

# plot SNR topography
fig, ax = plt.subplots(1)
mne.viz.plot_topomap(snrs_stim_hz_chaverage, raw.info, vlim=(1, None), axes=ax)
fig.savefig(f"{FILENAME}-topography.png")

print(f"average SNR (all channels): {snrs_stim_hz_chaverage.mean()}")
