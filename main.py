import matplotlib.pyplot as plt

import numpy as np
from scipy.stats import ttest_rel

import mne

# Load raw data
data_path = "S213_1.bdf"

TIME = 600

raw = mne.io.read_raw_bdf("S213_1.bdf", preload=True, verbose=False)
raw = raw.crop(600,600+4000)
print("read data")

raw.drop_channels(raw.ch_names[72:-1])
raw.set_channel_types(mapping={'Nose': 'eog', 'LHEOG': 'eog', 'RHEOG': 'eog', 'RVEOGS': 'eog',
                                   'RVEOGI': 'eog', 'M1': 'eog', 'M2': 'eog', 'LVEOGI': 'eog'})

# Set montage
raw.set_montage(montage='biosemi64')

# Set common average reference
raw.set_eeg_reference("average", projection=False, verbose=False)


roi_vis = [
    "POz",
    "Oz",
    "O1",
    "O2",
    "PO3",
    "PO4",
    "PO7",
    "PO8",
    "Status"
] 

# filter redundant data
raw.pick(roi_vis)
raw.filter(l_freq=0.5, h_freq=None, fir_design="firwin", verbose=False, n_jobs=-1)

# detect events and edit
events = mne.find_events(raw, stim_channel="Status", mask=255, min_duration=1.001 / raw.info['sfreq'])

events = mne.pick_events(events, include=[210,211,212,213,220,221,222,223])

#Only take the events after a break
events = np.array([e for (i,e) in enumerate(events) if events[i][0] - events[i-1][0] > 4*2048])

# minimum_duration = np.min(np.diff(events.T[0]))/2048
# print(f"set events minimum duration is {minimum_duration}")

# Construct epochs
tmin, tmax = 0, 30  # in s
baseline = None
epochs = mne.Epochs(
    raw,
    events=events,
    tmin=tmin,
    tmax=tmax,
    baseline=baseline,
    verbose=False,
)

#Calculate PSD
fmin = 0.5
fmax = 4.0
sfreq = epochs.info["sfreq"]

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
    mean_noise = np.pad(mean_noise, pad_width=pad_width, constant_values=np.nan)

    return psd / mean_noise

#Average every 3 bins
snrs = snr_spectrum(psds, noise_n_neighbor_freqs=3, noise_skip_neighbor_freqs=1)

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
fig.savefig("elad-post-breaks.png")
#Todo: Add a cropped version.