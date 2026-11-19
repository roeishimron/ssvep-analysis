"""FrequencyTagging (full-length spectrum, BCA / z / SNR) on synthetic data."""
import mne
import numpy as np
import pytest

from analysis import (
    FrequencyTagging, RawRecording, bin_measures, is_carrier_harmonic, neighbour_offsets,
    oddball_harmonics,
)
from core import Study
from core_types import ConditionProperties

FS = 300.0
N = int(56 * FS)                       # 56 s: 1/56 Hz bins, 5 Hz on bin 280
PROPS = ConditionProperties(np.float64(5.0), np.float64(15.0))


def _signal(amp_uv, n_subjects=1, n_trials=3, n_channels=2, noise_uv=1.0, seed=0):
    """(S, T, C, N) microvolts: a 5 Hz sinusoid of `amp_uv` (per subject) plus white noise."""
    rng = np.random.default_rng(seed)
    t = np.arange(N) / FS
    amp = np.broadcast_to(np.asarray(amp_uv, float), (n_subjects,))
    x = rng.normal(0, noise_uv, (n_subjects, n_trials, n_channels, N))
    x += amp[:, None, None, None] * np.sin(2 * np.pi * 5.0 * t)
    return x


def _rec(x, bad=None, names=None):
    names = names or [f"C{i}" for i in range(x.shape[2])]
    return RawRecording("synthetic", x, FS, names, PROPS, bad_channels=bad)


def test_neighbours_match_reference_rule():
    # 10 per side, the adjacent bin skipped on each side.
    np.testing.assert_array_equal(neighbour_offsets(), np.r_[-11:-1, 2:12])


def test_amplitude_scaling_and_bca():
    ft = FrequencyTagging(_rec(_signal(1.0, noise_uv=0.5)))
    assert ft.bin_index(5.0) == 280
    m = ft.measures_at(5.0)
    assert m["amp"][0] == pytest.approx(1.0, abs=0.05)          # peak amplitude in uV
    assert m["bca"][0] == pytest.approx(m["amp"][0] - m["noise"][0])
    assert m["z"][0] > 10
    assert m["snr"][0] == pytest.approx(m["amp"][0] / m["noise"][0])


def test_noise_only_is_not_detected():
    ft = FrequencyTagging(_rec(_signal(0.0)))
    assert abs(ft.measures_at(5.0)["z"][0]) < 3.1


def test_bin_measures_formulas():
    spec = np.ones(40)
    spec[20] = 3.0
    spec[20 + neighbour_offsets()[::2]] = 1.5             # some spread in the neighbours
    nb = spec[20 + neighbour_offsets()]
    m = bin_measures(spec, 20, neighbour_offsets())
    assert m["bca"] == pytest.approx(3.0 - nb.mean())
    assert m["z"] == pytest.approx((3.0 - nb.mean()) / nb.std(ddof=1))


def test_off_bin_frequency_is_rejected():
    x = _signal(1.0)[..., : N - 150]                     # 55.5 s: 5 Hz falls between bins
    with pytest.raises(ValueError):
        FrequencyTagging(_rec(x)).bin_index(5.0)


def test_bad_channels_drop_out_of_roi():
    x = _signal(1.0, n_channels=2, noise_uv=0.5)
    x[:, :, 1] *= 100                                    # channel 1 is junk
    bad = np.array([[False, True]])
    ft = FrequencyTagging(_rec(x, bad=bad))
    assert np.isnan(ft.amplitude()[0, 1]).all()
    assert ft.measures_at(5.0)["amp"][0] == pytest.approx(1.0, abs=0.05)


def test_subject_with_all_roi_channels_bad_is_excluded():
    x = _signal([1.0, 1.0, 1.0], n_subjects=3, noise_uv=0.5)
    bad = np.zeros((3, 2), dtype=bool)
    bad[2] = True
    r = FrequencyTagging(_rec(x, bad=bad)).existence(5.0)
    assert (r.n, r.n_excluded) == (2, 1)


def test_existence_statistics_and_detection_counts():
    amps = [1.0] * 6 + [0.0] * 4                         # 6 responders, 4 non-responders
    x = _signal(amps, n_subjects=10, noise_uv=1.0, seed=3)
    r = FrequencyTagging(_rec(x)).existence(5.0)
    assert r.n == 10
    assert r.detected[3.1] == 6
    assert 6 <= r.detected[1.64] <= 8                    # chance hits at p = .05 allowed
    assert r.mean_bca > 0 and r.p_t < 0.05 and r.p_wilcoxon < 0.05
    assert r.grand_z > 3.1
    assert r.cohen_d == pytest.approx(r.mean_bca / r.sd_bca)


def test_summed_harmonics_adds_bca():
    t = np.arange(N) / FS
    x = _signal(1.0, noise_uv=0.3) + 0.5 * np.sin(2 * np.pi * 10.0 * t)
    ft = FrequencyTagging(_rec(x))
    single = ft.measures_at(5.0)["bca"][0]
    summed = ft.summed_at([5.0, 10.0])["bca"][0]
    assert summed == pytest.approx(single + ft.measures_at(10.0)["bca"][0], rel=1e-6)


@pytest.mark.parametrize("carrier, contaminated", [
    (10.0, [10, 20, 30]), (15.0, [15, 30]), (20.0, [20]),
])
def test_oddball_harmonics_use_the_condition_carrier(carrier, contaminated):
    harmonics = oddball_harmonics(5.0, carrier, 35.0)
    assert [f for f, _ in harmonics] == [5, 10, 15, 20, 25, 30, 35]
    assert [f for f, clean in harmonics if not clean] == contaminated
    assert is_carrier_harmonic(30.0, 15.0) and not is_carrier_harmonic(10.0, 15.0)


def test_study_bads_reach_the_analysis():
    """info['bads'] from the loader -> Study blob -> ConditionView -> FrequencyTagging."""
    names = ["T5", "T6", "O1"]
    info_a = mne.create_info(names, FS, "eeg")
    info_b = mne.create_info(names, FS, "eeg")
    info_b["bads"] = ["T6"]
    x = _signal([1.0, 1.0], n_subjects=2, n_channels=3, noise_uv=0.5)
    x[1, :, 1] *= 100
    study = Study(iter([("P01", PROPS, info_a, FS, x[:1]), ("P02", PROPS, info_b, FS, x[1:])]))
    view = study.conditions()[PROPS].take_channels(["T5", "T6"])
    np.testing.assert_array_equal(view.bad_channels(), [[False, False], [False, True]])
    amp = FrequencyTagging(view).measures_at(5.0)["amp"]
    np.testing.assert_allclose(amp, 1.0, atol=0.05)
