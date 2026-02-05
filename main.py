import os
from os.path import isfile, join
import matplotlib.pyplot as plt
import numpy as np
import mne
from typing import Any, List, Tuple
from scipy.signal.windows import kaiser
from scipy.stats import linregress

from analyze_spectrum import analyze_spectrum, plot_snrs
from subjects import Condition, AllConditionSubjects

SUBJECT_NAME = "roei"

EXPERIMENT_NAME = "english_vs_mirror"
BASE_FREQ = 10
ODDBALL_MODULATION = 2

TARGET_ELECTRODES = np.array(["T5", "T6", "P3", "P4"])
BAD_ELECTRODES = []
SUM_HARMONICS_UNTIL = 1
AMOUNT_OF_BLOCKS = 5
BLOCK_LENGTH = 12
TRIAL_MARGIN = 0.5  # remove this amount of seconds from beggining to end of trial
TRIAL_START, TRIAL_DURATION = 3 - \
    TRIAL_MARGIN, AMOUNT_OF_BLOCKS*BLOCK_LENGTH  # in s

BLOCK_ELECTRODES = ["T5", "T6"]
TRIALS_RANGE = (0, 3)
TIME_MANIPULATED = False

RECORDING_FREQUENCY = 300
BASE_PATH = "/media/lab-server/roei.shimron/ssvep/experiments"


def subject_name_into_filename(subject_name: str):
    return f"""{BASE_PATH}/{EXPERIMENT_NAME}_{BASE_FREQ}hz/{subject_name}_{EXPERIMENT_NAME}_{BASE_FREQ}hz_{ODDBALL_MODULATION}ob_{TRIAL_DURATION}s_raw.edf"""

def parse_file(filename: str) -> Tuple[np.ndarray, Any]:

    raw = mne.io.read_raw_edf(filename, preload=True, verbose=False)

    raw.rename_channels(lambda s: s.replace(
        "EEG ", "").replace("-Pz", ""), False)

    raw.drop_channels(['Ax', 'Ay', 'Az'])
    raw.drop_channels(['Event', 'CM'])
    raw.drop_channels([c for c in raw.ch_names if ":" in c])
    raw.drop_channels(BAD_ELECTRODES)
    raw.set_montage(montage='standard_1020')

    # detect events and edit
    events = mne.find_events(raw, stim_channel="Trigger", mask=8)
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
    )
    print(f"{epochs.drop_log}")

    return epochs.get_data(units="mV"), raw.info

def parse_folder(foldername: str):
    paths = [p for p in [join(foldername, f) for f in os.listdir(foldername)] if isfile(p)]
    datas_with_info = [parse_file(p) for p in paths]

    for file, (data, _) in zip(paths, datas_with_info):
        if data.shape != datas_with_info[0][0].shape:
            print(f"Bad data shape for {file} (shape is {data.shape})")
            exit()
    
    all_subjects = np.array([d[0] for d in datas_with_info])
    # tempurarly, unify the subjects as if there are simply more repetitions
    return all_subjects, datas_with_info[0][1]


# Can't eat the variable array dimentions!
def into_spectrum(data: np.typing.NDArray) -> Tuple[np.typing.NDArray,
                                                    np.typing.NDArray,
                                                    np.typing.NDArray]:
    WINDOW_SIZE = int(RECORDING_FREQUENCY) * 2
    segments = np.lib.stride_tricks.sliding_window_view(data,
                                                        WINDOW_SIZE, -1)[..., ::WINDOW_SIZE, :]
    segments = segments - np.mean(segments, axis=-1, keepdims=True)

    fourier_components = np.fft.rfft(segments*kaiser(segments.shape[-1], 4),
                                     axis=-1)
    freqs = np.fft.rfftfreq(segments.shape[-1], d=1/RECORDING_FREQUENCY)

    average_component = np.average(fourier_components, -2)
    amplitudes = np.abs(average_component)**2

    return (freqs, amplitudes, fourier_components)

# Calculate PSD
fmin = 0.5
fmax = BASE_FREQ + 5

#TODO: Name the folder in a "subject-like" manner allowing configuration from the constants
microvolt_data, raw_info = parse_file(subject_name_into_filename(SUBJECT_NAME))
microvolt_data, raw_info = parse_folder(f"{BASE_PATH}/all_hebrew_vs_mirror/{BASE_FREQ}hz_{ODDBALL_MODULATION}ob_{TRIAL_DURATION}s")#

channels = raw_info["chs"]
channel_names = [c["ch_name"] for c in channels]
microvolt_data = microvolt_data[..., :-1]

V1_ELECTRODE_INDICES = np.array([i for i in range(
    len(channels)) if channels[i]["ch_name"] in set(["O1", "O2"])])

_freqs, _amplitudes, fourier_components = into_spectrum(microvolt_data)
print("got psds")

target_channel_indices = np.argwhere(np.isin(np.array(channel_names),
                                             TARGET_ELECTRODES)).flatten()

subject = AllConditionSubjects(_amplitudes, _freqs, SUBJECT_NAME, 
                  np.float64(BASE_FREQ/ODDBALL_MODULATION),
                  np.float64(BASE_FREQ))

target_subject = AllConditionSubjects(_amplitudes[...,target_channel_indices, :], _freqs, SUBJECT_NAME, 
                  np.float64(BASE_FREQ/ODDBALL_MODULATION),
                  np.float64(BASE_FREQ))

analyze_spectrum(target_subject, 1, 20)
plot_snrs(subject, raw_info)

# Find corresponding indices using mne.pick_types()

# find index of frequency bin closest to stimulation frequency


# phase analysis:
# target_snrs = snrs[target_channel_indices]
# SIGNAL_PHASE_ELECTRODE = target_channel_indices[np.argmax(
#     target_snrs[:, freqs == BASE_FREQ//ODDBALL_MODULATION])]
# BASE_PHASE_ELECTRODE = V1_ELECTRODE_INDICES[np.argmax(
#     snrs[V1_ELECTRODE_INDICES, freqs == BASE_FREQ])]
# print(
#     f"signal phase; electrode: {SIGNAL_PHASE_ELECTRODE}, base: {BASE_PHASE_ELECTRODE}")
# TARGET_FREQ = BASE_FREQ/ODDBALL_MODULATION
# signal_components = np.squeeze(
#     fourier_components[..., freqs == TARGET_FREQ][:, SIGNAL_PHASE_ELECTRODE])

# noise_mask = ((freqs > TARGET_FREQ + 0.5) & (freqs < TARGET_FREQ + 2)) | ((freqs < TARGET_FREQ - 0.5) & (freqs > TARGET_FREQ - 2))
# noise_components = np.squeeze(
#     fourier_components[..., noise_mask][:, SIGNAL_PHASE_ELECTRODE].mean(2))
# assert(noise_components.shape == signal_components.shape)

# base_components = np.squeeze(
#     fourier_components[..., freqs == BASE_FREQ][:, BASE_PHASE_ELECTRODE])


# fig, ax = plt.subplots(subplot_kw={'projection': 'polar'},layout='constrained')
# ax.grid(True)
# for signal_component in signal_components:
#     ax.plot(np.angle(signal_component), np.abs(signal_component),  "o")

# _, ax = plt.subplots(2)
# for signal_component, noise_component in zip(signal_components, noise_components):
#     smoothed_signal = np.abs(np.convolve(signal_component, np.ones(5)/5, mode="valid"))
#     smoothed_noise = np.abs(np.convolve(noise_component, np.ones(5)/5, mode="valid"))

#     ax[0].plot(smoothed_signal)
#     ax[0].plot(smoothed_noise, color="r")
#     windows = np.lib.stride_tricks.sliding_window_view(signal_component, 5)
#     slopes = [slope for slope, intercept, r, p, se in 
#               [linregress(np.arange(len(win)), np.abs(win)) for win in windows]]
#     ax[1].plot(slopes)


# # Rotate the signal `ODDBALL_MODULATION`-fold so it'll be on the same rotation as the faster base. Then substract for difference.
# base_timescale = signal_components.mean(
#     -1)**ODDBALL_MODULATION / base_components.mean(-1)

# base_time_diff = (np.angle(base_timescale) + np.pi)/2/np.pi/BASE_FREQ*1000
# print(f"differences in ms are {base_time_diff}")

plt.show(block=True)
