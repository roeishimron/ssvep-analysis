import matplotlib.pyplot as plt
import numpy as np
import mne
from typing import Tuple

FILENAME = "roei_area_8hz_3ob_12-24"
raw = mne.io.read_raw_edf(
    f"/media/lab-server/roei.shimron/ssvep/experiments/{FILENAME}_raw.edf", preload=True, verbose=False)

BASE_FREQ = 8
ODDBALL_MODULATION = 3
TIME_MANIPULATED = False
TARGET_ELECTRODES = np.array(["Pz", "O1", "O2", "T5", "P3", "P4", "T6"])
BAD_ELECTRODES = []
SUM_HARMONICS_UNTIL = 1
AMOUNT_OF_BLOCKS = 7
TRIAL_START, TRIAL_DURATION = 0, AMOUNT_OF_BLOCKS*15-1 # in s
BLOCK_ELECTRODES = ["Pz", "O1", "O2", "T5", "P3", "P4", "T6"]
TRIALS_RANGE = (0, 3)


TOPO_WIDTH = 4
TOPO_HEIGHT = 4
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
    tmax=TRIAL_DURATION,
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


def into_spectrum(data: np.typing.NDArray) -> Tuple[np.typing.NDArray, np.typing.NDArray, np.typing.NDArray]:
    data = np.average(data, axis=0)

    if TIME_MANIPULATED:
        data = np.apply_along_axis(log_channel_time, -1, arr=data)

    freqs = np.fft.rfftfreq(data.shape[-1], d=1/RECORDING_FREQUENCY)
    amplitudes = np.abs(np.fft.rfft(data))
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


NOISE_NEIGHBORS = 7
NOISE_SKIP = 2

print(f'using {NOISE_NEIGHBORS} neighbors and skipping {NOISE_SKIP} bins')

snrs = snr_spectrum(psds, noise_n_neighbor_freqs=NOISE_NEIGHBORS,
                    noise_skip_neighbor_freqs=NOISE_SKIP)

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
target_snrs = snrs[target_channel_indices]

# SNR spectrum
target_snr_mean = target_snrs.mean(axis=0)
target_snr_std = target_snrs.std(axis=0)

axes[2].plot(freqs, target_snr_mean, color="r")
axes[2].fill_between(
    freqs, target_snr_mean - target_snr_std, target_snr_mean + target_snr_std, color="r", alpha=0.1
)
axes[2].set(
    title="Target SNR spectrum",
    ylabel="SNR",
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


HARMONEY_FREQS = np.array(range(TOPO_WIDTH * TOPO_HEIGHT))+1

TARGET_FREQS = BASE_FREQ/ODDBALL_MODULATION*HARMONEY_FREQS
RANGE_OF_TOPO_SEARCH = int(0.001 * RECORDING_FREQUENCY * BASE_FREQ)


def into_chaverage(target_freq: np.float64) -> Tuple[int, np.typing.NDArray]:
    target_center = int(np.argmin(np.abs(freqs - target_freq)))

    # get average SNR at target Hz for ALL channels
    snrs_stim_hz = snrs[:, target_center]
    return (freqs[target_center], snrs_stim_hz)


freqs_with_chaverages = list(map(into_chaverage, TARGET_FREQS))

upper_limit = np.max(np.array([t[1] for t in freqs_with_chaverages]).flatten())
# plot SNR topography
fig, axs = plt.subplots(TOPO_HEIGHT, TOPO_WIDTH,  sharex="none",
                        sharey="none", label=f"{FILENAME}-topomap")
fig.suptitle(f"clearity up to {upper_limit:.0f}")
for ((freq, chaverage), ax) in zip(freqs_with_chaverages, axs.flatten()):

    print(f"looking at freq {freq:.2f}")
    ax.set_title(f"SNR at F*{freq/BASE_FREQ*ODDBALL_MODULATION:.0f} ({
                 freq:.2f})")

    mne.viz.plot_topomap(chaverage, raw.info,
                         vlim=(1, upper_limit), axes=ax, show=False)

    print(f"average SNR (target channels): {
          chaverage[target_channel_indices].mean()}")

AC_FREQ = 50
# for the grand_average
SBA_TARGET_FREQS = np.array([BASE_FREQ/ODDBALL_MODULATION * i for i in range(1, SUM_HARMONICS_UNTIL+1)
                             if i % ODDBALL_MODULATION != 0 and BASE_FREQ/ODDBALL_MODULATION * i != AC_FREQ])


def extract_sba_average(data: np.typing.NDArray, target_freqs=SBA_TARGET_FREQS):
    _, freqs, amps = into_spectrum(data)

    harmonic_indices = np.array(
        [np.argmin(np.abs(freqs - t)) for t in target_freqs])
    return np.average(amps[:, harmonic_indices], 1)


# print the SBA
mne.viz.plot_topomap(extract_sba_average(
    microvolt_data, SBA_TARGET_FREQS), raw.info, show=False)

COMPARE_TARGET_TO = 0.5

# Calculate the target voltage vs noise voltage
# TODO: Consider presenting it by block
mne.viz.plot_topomap(extract_sba_average(microvolt_data, SBA_TARGET_FREQS)
                     - extract_sba_average(microvolt_data,
                                           SBA_TARGET_FREQS+COMPARE_TARGET_TO)/2
                     - extract_sba_average(microvolt_data,
                                           SBA_TARGET_FREQS-COMPARE_TARGET_TO)/2,
                     raw.info, show=False)


# calculate the coherence of stimuli to coherence of signal
TRIAL_MARGIN = 1
BLOCK_SIZE = int((TRIAL_DURATION+1)/AMOUNT_OF_BLOCKS - 2*TRIAL_MARGIN)

# TODO: Convert to list
BLOCK_ELECTRODE_INDICES = np.array([i for i in range(
    len(raw.info["chs"])) if raw.info["chs"][i]["ch_name"] in BLOCK_ELECTRODES])

block_diffs = []
for i in range(AMOUNT_OF_BLOCKS):
    start = RECORDING_FREQUENCY*(i*(BLOCK_SIZE+2*TRIAL_MARGIN))
    current_data = microvolt_data[:, :,
                                  start:start+BLOCK_SIZE*RECORDING_FREQUENCY]
    block_data = current_data[:, BLOCK_ELECTRODE_INDICES, :]

    target_block_sba = np.average(extract_sba_average(
        block_data, SBA_TARGET_FREQS), axis=0)
    target_block_noise = np.average(extract_sba_average(
        block_data, SBA_TARGET_FREQS+COMPARE_TARGET_TO)/2
        + extract_sba_average(block_data, SBA_TARGET_FREQS - COMPARE_TARGET_TO)/2)

    block_diffs.append(target_block_sba - target_block_noise)

fig, ax = plt.subplots()
fig.suptitle("SBA coherence at each block")
ax.plot(np.arange(AMOUNT_OF_BLOCKS) *
        (BLOCK_SIZE+2*TRIAL_MARGIN), np.array(block_diffs))

plt.show(block=True)
