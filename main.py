import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import mne

matplotlib.use('qtagg')

# Load raw data
raw = mne.io.read_raw_bdf("udi-data/Testdata-udi2.bdf", preload=True, verbose=False)
print("read data")

raw.drop_channels(raw.ch_names[64:-1])

# Set montage
raw.set_montage(montage='biosemi64')

# Set common average reference
raw.set_eeg_reference("average", projection=False, verbose=False)
raw.filter(l_freq=0.5, h_freq=None, fir_design="firwin", verbose=False, n_jobs=-1)

# detect events and edit
events = mne.find_events(raw, stim_channel="Status", mask=255)
events = mne.pick_events(events, include=[3])

# Construct epochs
tmin, tmax = 5, 25  # in s
epochs = mne.Epochs(
    raw,
    events=events,
    tmin=tmin,
    tmax=tmax,
    baseline=None,
    verbose=False,
)

#Calculate PSD
fmin = 1
fmax = 16
sfreq = epochs.info["sfreq"]

spectrum = epochs.compute_psd(
    "welch",
    n_fft=int(sfreq * (tmax - tmin)),
    fmin=fmin,
    fmax=fmax,
    verbose=False,
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
    mean_noise = np.pad(mean_noise, pad_width=pad_width, constant_values=np.nan)

    return psd / mean_noise

#Average every 5 bins
snrs = snr_spectrum(psds, noise_n_neighbor_freqs=5, noise_skip_neighbor_freqs=1)

print("got snr")
fig, axes = plt.subplots(2, 1, sharex="all", sharey="none", figsize=(8, 5))
freq_range = range(
    np.where(freqs >= fmin)[0][0], np.where(freqs >= fmax)[0][0]
)

psds_plot = 10 * np.log10(psds)
psds_mean = psds_plot.mean(axis=(0, 1))[freq_range]
psds_std = psds_plot.std(axis=(0, 1))[freq_range]
axes[0].plot(freqs[freq_range], psds_mean, color="b")
axes[0].fill_between(
    freqs[freq_range], psds_mean - psds_std, psds_mean + psds_std, color="b", alpha=0.1
)
axes[0].set(title="PSD spectrum", ylabel="Power Spectral Density [dB]")

# SNR spectrum
snr_mean = snrs.mean(axis=(0, 1))[freq_range]
snr_std = snrs.std(axis=(0, 1))[freq_range]

axes[1].plot(freqs[freq_range], snr_mean, color="r")
axes[1].fill_between(
    freqs[freq_range], snr_mean - snr_std, snr_mean + snr_std, color="r", alpha=0.1
)
axes[1].set(
    title="SNR spectrum",
    xlabel="Frequency [Hz]",
    ylabel="SNR",
    xlim=[fmin, fmax],
)
fig.show()

#topographic
roi_vis = [
    "POz",
    "Oz",
    "O1",
    "O2",
    "PO3",
    "PO4",
    "PO7",
    "PO8"
] 

# Find corresponding indices using mne.pick_types()
picks_roi_vis = mne.pick_types(
    epochs.info, eeg=True, stim=False, exclude="bads", selection=roi_vis
)

STIM_FREQUENCY = 6

# find index of frequency bin closest to stimulation frequency
i_bin_stim_hz = np.argmin(abs(freqs - STIM_FREQUENCY))
snrs_target = snrs[:, :, i_bin_stim_hz][:, picks_roi_vis]

# get average SNR at 1 Hz for ALL channels
snrs_stim_hz = snrs[:, :, i_bin_stim_hz]
snrs_stim_hz_chaverage = snrs_stim_hz.mean(axis=0)

# plot SNR topography
fig, ax = plt.subplots(1)
mne.viz.plot_topomap(snrs_stim_hz_chaverage, epochs.info, vlim=(1, None), axes=ax)
fig.savefig("topography.png")

print(f"average SNR (all channels): {snrs_stim_hz_chaverage.mean()}")
print(f"average SNR (occipital ROI): {snrs_target.mean()}")