import matplotlib.pyplot as plt
import numpy as np
import mne
from typing import Tuple

FILENAME = "udi_ordered_digits_vs_hebrew_20hz_5ob_15s_constant"
raw = mne.io.read_raw_edf(
    f"/media/lab-server/roei.shimron/ssvep/experiments/{FILENAME}_raw.edf", preload=True, verbose=False)

BASE_FREQ = 20
ODDBALL_MODULATION = 5
TIME_MANIPULATED = False
TARGET_ELECTRODES = np.array(["T5"])
BAD_ELECTRODES = []
SUM_HARMONICS_UNTIL = 3
AMOUNT_OF_BLOCKS = 3
BLOCK_LENGTH = 5
TRIAL_START, TRIAL_DURATION = 3, AMOUNT_OF_BLOCKS*BLOCK_LENGTH  # in s
BLOCK_ELECTRODES = ["T5"]
TRIALS_RANGE = (0, 4)


AC_FREQ = 50
TOPO_WIDTH = 4
TOPO_HEIGHT = 2
RECORDING_FREQUENCY = 300
print("read data")

raw.rename_channels(lambda s: s.replace("EEG ", "").replace("-Pz", ""), False)

raw.drop_channels(['Ax', 'Ay', 'Az'])
raw.drop_channels(['Event', 'CM'])
raw.drop_channels([c for c in raw.ch_names if ":" in c])
raw.drop_channels(BAD_ELECTRODES)
raw.set_montage(montage='standard_1020')

# detect events and edit
events = mne.find_events(raw, stim_channel="Trigger", mask=255)
raw.drop_channels(["Trigger"])

# Set common average reference
raw.set_eeg_reference()


# Handle too close events:
diffs = np.diff(events[:, 0], append=raw.last_samp)
valids = np.argwhere(diffs > 1000).flatten()
events = events[valids][TRIALS_RANGE[0]:TRIALS_RANGE[1]]
print(f"found {len(events)} events")

# Construct epochs
epochs = mne.Epochs(
    raw,
    picks='data',
    events=events,
    tmin=TRIAL_START,
    tmax=TRIAL_DURATION + TRIAL_START,
    baseline=None,
    verbose=False,
)


# time-shifting parameters
SCALE = 20
A = 9

# take f(x) into f(r(x))


def exp_channel_time(c):
    f_source = np.linspace(0, TRIAL_DURATION, c.shape[-1])
    r_x = A*np.exp(f_source/SCALE) - A

    return np.exp(np.interp(r_x, f_source, np.log(c+1)))-1

# takes f(x) into f(t(x))


def log_channel_time(c):
    ERROR = 0.1
    f_source = np.linspace(0, TRIAL_DURATION, c.shape[-1]) + ERROR
    t_x = SCALE*np.log(f_source/A+1)
    return np.log(np.interp(t_x, f_source, np.exp(c)))


def into_spectrum(data: np.typing.NDArray) -> Tuple[np.typing.NDArray,
                                                    np.typing.NDArray,
                                                    np.typing.NDArray]:
    if TIME_MANIPULATED:
        data = np.apply_along_axis(log_channel_time, -1, arr=data)

    # Averaging the amps instead of the signal due to offsets in the different recordings
    amplitudes = np.average(np.abs(np.fft.rfft(data*np.hamming(data.shape[-1]),
                                               axis=-1)), axis=0)

    freqs = np.fft.rfftfreq(data.shape[-1], d=1/RECORDING_FREQUENCY)
    psd = amplitudes**2/freqs

    return (psd, freqs, amplitudes)


# Calculate PSD
fmin = 0.5
fmax = (BASE_FREQ/ODDBALL_MODULATION) * 16 * 2

channel_names = raw.ch_names
microvolt_data = epochs.get_data(units="mV")

psds, freqs, amplitudes = into_spectrum(microvolt_data)
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
    # applying along each electrode
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

# Shpuld be applied foreach electrode


def local_zscore(psd_slice, skip):
    center = int(psd_slice.shape[0]/2)
    centerless = list(psd_slice[:center-skip]) + list(psd_slice[center+skip:])

    return (psd_slice[center] - np.average(centerless)) / np.std(centerless)


def one_d_zscore(psd, window_relative_size=1/100):
    SKIP = 3
    half_window_size = int(psd.shape[0]*window_relative_size/2)
    jump = half_window_size + SKIP
    z_scores = [0] * jump
    for i in range(jump, psd.shape[0]-jump):
        z_scores.append(local_zscore(psd[i-jump: i+jump], SKIP))

    return np.array(z_scores + [0] * jump)


NOISE_NEIGHBORS = 7
NOISE_SKIP = 2

print(f'using {NOISE_NEIGHBORS} neighbors and skipping {NOISE_SKIP} bins')

snrs = snr_spectrum(psds, noise_n_neighbor_freqs=NOISE_NEIGHBORS,
                    noise_skip_neighbor_freqs=NOISE_SKIP)
z_scores = np.apply_along_axis(one_d_zscore, -1, amplitudes)

print("got snr")
_, axes = plt.subplots(4, 1, sharex="all", sharey="none", figsize=(
    8, 5), label=f"{FILENAME}-spectrum")

psds_plot = 10 * np.log10(psds)
psds_mean = psds_plot.mean(axis=0)
psds_std = psds_plot.std(axis=0)
axes[0].plot(freqs, psds_mean, color="b")
axes[0].fill_between(
    freqs, psds_mean - psds_std, psds_mean + psds_std, color="b", alpha=0.1
)
axes[0].set(title="PSD spectrum", ylabel="V^2/Hz")

# SNR spectrum
snr_mean = snrs.mean(axis=0)
snr_std = snrs.std(axis=0)

axes[1].plot(freqs, snr_mean, color="r")
axes[1].fill_between(
    freqs, snr_mean - snr_std, snr_mean + snr_std, color="r", alpha=0.1
)
axes[1].set(
    title="SNR spectrum",
    ylabel="SNR [P/N]",
    xlim=[fmin, fmax],
)

# draw the SNR of the target electrode (should be replaced with something cleverer, like RCA)
target_channel_indices = np.argwhere(np.isin(np.array(channel_names),
                                             TARGET_ELECTRODES)).flatten()
target_zscores = z_scores[target_channel_indices]

# z-score spectrum
z_score_mean = target_zscores.mean(axis=0)
z_score_std = target_zscores.std(axis=0)

axes[2].plot(freqs, z_score_mean, color="r")
axes[2].fill_between(
    freqs, z_score_mean - z_score_std, z_score_mean + z_score_std, color="r", alpha=0.1
)
axes[2].set(
    title="z-score spectrum",
    ylabel="z-score",
    xlim=[fmin, fmax],
)

# Amplitudes spectrum
amplitudes_plot = 10 * np.log10(amplitudes)
amplitudes_mean = amplitudes_plot.mean(axis=0)
amplitudes_std = amplitudes_plot.std(axis=0)

axes[3].plot(freqs, amplitudes_mean, color="g")
axes[3].fill_between(
    freqs, amplitudes_mean - amplitudes_std, amplitudes_mean + amplitudes_std, color="g", alpha=0.1
)
axes[3].set(
    title="Amplitudes spectrum",
    xlabel="Frequency [Hz]",
    ylabel="microV (logscale)",
    xlim=[fmin, fmax],
)


# Find corresponding indices using mne.pick_types()

# find index of frequency bin closest to stimulation frequency


HARMONEY_FREQS = np.array(range(min(TOPO_WIDTH * TOPO_HEIGHT, int(AC_FREQ / BASE_FREQ * ODDBALL_MODULATION - 1) )))+1

TARGET_FREQS = BASE_FREQ/ODDBALL_MODULATION*HARMONEY_FREQS


def into_channel_average(target_freq: np.float64) -> Tuple[int, np.typing.NDArray]:
    target_center = int(np.argmin(np.abs(freqs - target_freq)))

    # get average SNR at target Hz for ALL channels
    snrs_stim_hz = z_scores[:, target_center]
    return (freqs[target_center], snrs_stim_hz)


freqs_with_channel_averages = list(map(into_channel_average, TARGET_FREQS))

upper_limit = np.max(np.array([t[1] for t in freqs_with_channel_averages]).flatten())
# plot SNR topography
fig, axs = plt.subplots(TOPO_HEIGHT, TOPO_WIDTH,  sharex="none",
                        sharey="none", label=f"{FILENAME}-topomap")
fig.suptitle(f"clearity up to {upper_limit:.0f}")
for ((freq, channel_average), ax) in zip(freqs_with_channel_averages, axs.flatten()):

    print(f"looking at freq {freq:.2f}")
    ax.set_title(f"SNR at F*{freq/BASE_FREQ*ODDBALL_MODULATION:.0f} ({
                 freq:.2f})")

    mne.viz.plot_topomap(channel_average, raw.info,
                         vlim=(1, upper_limit), axes=ax, show=False)

    print(f"average SNR (target channels): {
          channel_average[target_channel_indices].mean()}")

# for the grand_average
SBA_TARGET_FREQS = np.array([BASE_FREQ/ODDBALL_MODULATION * i for i in range(1, SUM_HARMONICS_UNTIL+1)
                             if i % ODDBALL_MODULATION != 0 and BASE_FREQ/ODDBALL_MODULATION * i != AC_FREQ])


def signal_into_sba_average(signal, freqs, target_freqs=SBA_TARGET_FREQS):
    harmonic_indices = np.array(
        [np.argmin(np.abs(freqs - t)) for t in target_freqs])
    return np.average(signal[:, harmonic_indices], 1)


def extract_z_score_and_freqs(data: np.typing.NDArray):
    _, freqs, amps = into_spectrum(data)
    return np.apply_along_axis(one_d_zscore, -1, amps), freqs


# print the SBA
mne.viz.plot_topomap(signal_into_sba_average(z_scores, freqs, SBA_TARGET_FREQS), raw.info, show=False)

COMPARE_TARGET_TO = 0.5

fig, axs = plt.subplots(1, AMOUNT_OF_BLOCKS,  sharex="none",
                        sharey="none", label=f"{FILENAME}-target-vs-noise-sba-per-block")

# calculate the coherence of stimuli to coherence of signal
BLOCK_ELECTRODE_INDICES = np.array([i for i in range(
    len(raw.info["chs"])) if raw.info["chs"][i]["ch_name"] in BLOCK_ELECTRODES])

block_target = []
block_noise = []

CUT_FROM_START = 0
CUT_FROM_END = 0

for (i, ax) in enumerate(axs):
    start = int(RECORDING_FREQUENCY*(i*BLOCK_LENGTH+CUT_FROM_START))
    current_data = microvolt_data[:, :,
                                  start:start+(BLOCK_LENGTH-CUT_FROM_END)*RECORDING_FREQUENCY]
    z_score_and_freqs = extract_z_score_and_freqs(current_data)

    target_block_sba = signal_into_sba_average(
        *z_score_and_freqs, SBA_TARGET_FREQS)
    target_block_noise = signal_into_sba_average(*z_score_and_freqs, SBA_TARGET_FREQS+COMPARE_TARGET_TO) / \
        2 + signal_into_sba_average(*z_score_and_freqs,
                                      SBA_TARGET_FREQS - COMPARE_TARGET_TO)/2

    block_target.append(np.average(target_block_sba[BLOCK_ELECTRODE_INDICES]))
    block_noise.append(np.average(target_block_noise[BLOCK_ELECTRODE_INDICES]))

    # plot
    ax.set_title(f"Block #{i}")
    mne.viz.plot_topomap(target_block_sba - target_block_noise,
                         raw.info, axes=ax, show=False)

fig, ax = plt.subplots()
fig.suptitle("SBA coherence at each block")
# ax.plot((np.arange(AMOUNT_OF_BLOCKS)+1) * (BLOCK_LENGTH),
#         np.array(block_target) - np.array(block_noise), label="signal")
ax.plot((np.arange(AMOUNT_OF_BLOCKS)+1) * (BLOCK_LENGTH),
        np.array(block_target), label="signal")
ax.plot((np.arange(AMOUNT_OF_BLOCKS)+1) * (BLOCK_LENGTH),
        np.array(block_noise), label="noise")

# TODO: Consider showing the noise-vs-signal difference
# using z-scores or something that respects the structure
# that beign close to noise by E is MUCH "less indicative" than close by 2E etc.

plt.legend()
plt.show(block=True)
