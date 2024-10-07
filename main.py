import matplotlib.pyplot as plt
import numpy as np
import mne

FILENAME = "test-sin-reversal-5_raw"
raw = mne.io.read_raw_edf(f"roei-data/{FILENAME}.edf", preload=True, verbose=False)
tmin, tmax = 15, 180-10  # in s
raw = raw.crop(tmin,tmax)

print("read data")

raw.rename_channels(lambda s: s.replace("EEG ", "").replace("-Pz", ""), False)

raw.drop_channels(['Ax', 'Ay', 'Az'])
raw.drop_channels(['X3:', 'X2:', 'X1:', 'Event', 'CM'])
raw.set_eeg_reference()

raw.set_montage(montage='standard_1020')

# Set common average reference

raw.filter(l_freq=0.5, h_freq=None, fir_design="firwin", verbose=False, n_jobs=-1)

# raw.pick(["O1", "O2", "Pz"])

#Calculate PSD
fmin = 0.5
fmax = 30.0
sfreq = 300

spectrum = raw.compute_psd(
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
    mean_noise = np.pad(mean_noise, pad_width=pad_width, constant_values=np.nan)

    return psd / mean_noise

#Average every 3 bins
snrs = snr_spectrum(psds, noise_n_neighbor_freqs=15, noise_skip_neighbor_freqs=1)

print("got snr")
fig, axes = plt.subplots(2, 1, sharex="all", sharey="none", figsize=(8, 5))


psds_plot = 10 * np.log10(psds)
psds_mean = psds_plot.mean(axis=(0))
psds_std = psds_plot.std(axis=(0))
axes[0].plot(freqs, psds_mean, color="b")
axes[0].fill_between(
    freqs, psds_mean - psds_std, psds_mean + psds_std, color="b", alpha=0.1
)
axes[0].set(title="PSD spectrum", ylabel="Power Spectral Density [dB]")

# SNR spectrum
snr_mean = snrs.mean(axis=(0))
snr_std = snrs.std(axis=(0))

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
MIN_INDEX = 100
i_bin_6hz = np.argmax(snr_mean[MIN_INDEX:-MIN_INDEX]) + MIN_INDEX # adding because the arg is relative to the array
print(f"maximum freq is {freqs[i_bin_6hz]} with value of {snr_mean[i_bin_6hz]}")

# get average SNR at 1 Hz for ALL channels
snrs_6hz = snrs[:, i_bin_6hz]
snrs_6hz_chaverage = snrs_6hz

# plot SNR topography
fig, ax = plt.subplots(1)
mne.viz.plot_topomap(snrs_6hz_chaverage, raw.info, vlim=(1, None), axes=ax)
fig.savefig(f"{FILENAME}-topography.png")

print(f"average SNR (all channels): {snrs_6hz_chaverage.mean()}")
