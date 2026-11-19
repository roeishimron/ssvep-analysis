import os
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import mne
import numpy as np

from loader import (
    StudyLoader, group_sessions, reference_channels, screen_channels, summarize_sessions,
)

FS = 300.0
SCALP = ["Fp1", "Fp2", "F3", "F4", "C3", "C4", "T5", "T6", "O1", "O2"]


def _dsi_raw(n_bursts=4, seconds_per_stream=70.0, noisy=(), seed=0):
    """Synthetic DSI-24-like recording saved against Pz (all-zero Pz channel).

    Trigger rests at 9 and steps 9 -> 6 -> 4 -> 9 over 270 ms per burst.
    Returns (raw, sample index of each burst's final 4 -> 9 step).
    """
    rng = np.random.default_rng(seed)
    n = int((5 + n_bursts * seconds_per_stream) * FS)
    labels = [f"EEG {c}-Pz" for c in SCALP] + ["EEG A1-Pz", "EEG A2-Pz", "Pz",
              "CM", "EEG X1:S3-Pz", "Trigger", "Event", "Ax", "Ay", "Az"]
    data = rng.normal(0, 5e-6, (len(labels), n))                # volts
    data[labels.index("Pz")] = 0.0
    for c in noisy:
        data[labels.index(f"EEG {c}-Pz")] *= 50
    trig = np.full(n, 9.0)
    ends = []
    for b in range(n_bursts):
        s0 = int((5 + b * seconds_per_stream) * FS)
        trig[s0:s0 + 40] = 6
        trig[s0 + 40:s0 + 81] = 4
        ends.append(s0 + 81)
    data[labels.index("Trigger")] = trig
    types = ["eeg"] * len(labels)
    types[labels.index("Trigger")] = "stim"
    info = mne.create_info(labels, FS, types)
    return mne.io.RawArray(data, info, verbose=False), ends


def _quiet():
    return StudyLoader(log=lambda _: None)


class TestParseFolderName(unittest.TestCase):
    def test_parse_folder_name(self):
        props, duration = _quiet()._parse_folder_name("10hz_2ob_60s")
        self.assertEqual(props.carrier_frequency, 10.0)
        self.assertEqual(props.target_frequency, 5.0)
        self.assertEqual(duration, 60.0)

        props, duration = _quiet()._parse_folder_name("10hz_2ob")
        self.assertEqual(duration, 60.0)  # Default

    def test_load_non_existent_dir(self):
        with self.assertRaises(FileNotFoundError):
            list(_quiet().load("/non/existent/path"))

    def test_load_no_files(self):
        with tempfile.TemporaryDirectory() as root:
            self.assertEqual(list(_quiet().load(root)), [])


class TestSessions(unittest.TestCase):
    t0 = datetime(2026, 1, 5, 10, 0, tzinfo=timezone.utc)

    def test_groups_by_time_not_name(self):
        t = self.t0
        entries = [
            # session 1: three conditions within ~20 minutes, files named differently
            (t, 10, "10/a.edf"), (t + timedelta(minutes=8), 15, "15/a_.edf"),
            (t + timedelta(minutes=16), 20, "20/A.edf"),
            # session 2, two hours later
            (t + timedelta(hours=2), 10, "10/b.edf"),
            (t + timedelta(hours=2, minutes=9), 15, "15/b.edf"),
        ]
        sessions = group_sessions(reversed(entries))
        self.assertEqual([s.id for s in sessions], ["P01", "P02"])
        self.assertEqual(sorted(sessions[0].files), [10, 15, 20])
        self.assertEqual(sessions[0].files[15], "15/a_.edf")
        self.assertEqual(sorted(sessions[1].files), [10, 15])

    def test_repeated_condition_opens_new_session(self):
        t = self.t0
        sessions = group_sessions([(t, 10, "x"), (t + timedelta(minutes=5), 10, "y")])
        self.assertEqual(len(sessions), 2)

    def test_gap_is_measured_from_previous_file(self):
        t = self.t0
        entries = [(t, 10, "a"), (t + timedelta(minutes=50), 15, "b"),
                   (t + timedelta(minutes=100), 20, "c")]
        self.assertEqual(len(group_sessions(entries)), 1)
        self.assertEqual(len(group_sessions(entries, max_gap_s=40 * 60)), 3)

    def test_summary_has_no_paths(self):
        text = summarize_sessions(group_sessions([(self.t0, 10, "/x/secret.edf")]))
        self.assertNotIn("secret", text)
        self.assertIn("1 sessions", text)

    def test_list_sessions_ignores_bads_subfolder(self):
        with tempfile.TemporaryDirectory() as root:
            for folder in ("10hz_2ob_60s", "15hz_3ob_60s",
                           os.path.join("15hz_3ob_60s", "bads"), "notes"):
                os.makedirs(os.path.join(root, folder))
            paths = {
                os.path.join(root, "10hz_2ob_60s", "s1.edf"): self.t0,
                os.path.join(root, "15hz_3ob_60s", "s1x.edf"): self.t0 + timedelta(minutes=9),
                os.path.join(root, "15hz_3ob_60s", "bads", "s1y.edf"): self.t0,
            }
            for p in paths:
                open(p, "w").close()

            def fake_read(path, **_):
                m = MagicMock()
                m.info = {"meas_date": paths[path]}
                return m

            with patch("loader.mne.io.read_raw_edf", side_effect=fake_read):
                sessions = _quiet().list_sessions(root)
        self.assertEqual(len(sessions), 1)
        self.assertEqual(len(sessions[0].files), 2)
        self.assertTrue(all("bads" not in p for p in sessions[0].files.values()))


class TestScreen(unittest.TestCase):
    def test_flags_flat_and_noisy_not_reference(self):
        rng = np.random.default_rng(1)
        names = SCALP + ["A1", "A2", "Pz"]
        x = rng.normal(0, 5, (3, len(names), 3000))
        x[:, names.index("Pz")] = 0
        x[:, names.index("F4")] = 0                      # flat
        x[1, names.index("O2")] *= 20                    # noisy in one trial only
        bad = screen_channels(x, names, FS)
        self.assertEqual([n for n, b in zip(names, bad) if b], ["F4", "O2"])

    def test_reference_channels_exclude_earclips_and_bads_keep_pz(self):
        names = ["O1", "O2", "A1", "A2", "Pz"]
        bad = np.array([False, True, False, False, False])
        self.assertEqual(reference_channels(names, bad), [0, 4])


class TestLoadEdf(unittest.TestCase):
    def _load(self, raw, **kwargs):
        logs = []
        loader = StudyLoader(log=logs.append, **kwargs)
        with patch("loader.mne.io.read_raw_edf", return_value=raw):
            data, info = loader._load_edf("/d/10hz_2ob_60s/Some Name.edf", 60.0, label="P01 10 Hz")
        return loader, data, info, logs

    def test_screened_reference_and_trial_count(self):
        raw, ends = _dsi_raw(n_bursts=4, noisy=("F3",))
        loader, data, info, logs = self._load(raw.copy(), epoch_s=StudyLoader.SPECTRUM_EPOCH_S)
        self.assertEqual(data.shape[:2], (1, 3))                # first 3 of 4 bursts
        self.assertEqual(data.shape[-1], 56 * 300)              # 56 s: 5 Hz is on a bin
        self.assertEqual(info["bads"], ["F3"])
        self.assertEqual(loader.records[0]["n_events"], 4)
        self.assertEqual(loader.records[0]["n_used"], 3)
        self.assertIn("4 trigger bursts", logs[0])
        self.assertNotIn("Some Name", " ".join(logs))

        names = info.ch_names
        self.assertNotIn("Trigger", names)
        ref = [i for i, c in enumerate(names) if c not in ("A1", "A2", "F3")]
        self.assertIn(names.index("Pz"), ref)
        # Good scalp channels (Pz included) average to zero after referencing.
        np.testing.assert_allclose(data[0][:, ref].mean(1), 0, atol=1e-9)

        # The epoch starts 4.5 s after the burst's last trigger step; data in uV.
        picks = [c for c in SCALP if c != "F3"] + ["Pz"]
        x = raw.get_data(picks=[f"EEG {c}-Pz" if c != "Pz" else c for c in picks], units="uV")
        start = ends[0] + int(4.5 * FS)
        expected = x[picks.index("O1"), start:start + 10] - x[:, start:start + 10].mean(0)
        np.testing.assert_allclose(data[0, 0, names.index("O1"), :10], expected, rtol=1e-6)

    def test_incomplete_last_stream_is_not_used(self):
        raw, _ = _dsi_raw(n_bursts=3)
        raw.crop(tmax=5 + 2 * 70 + 30)                          # third stream cut after ~30 s
        loader, data, _, logs = self._load(raw)
        self.assertEqual(data.shape[1], 2)
        self.assertEqual(loader.records[0]["n_complete"], 2)
        self.assertIn("not every trigger burst used", logs[0])

    def test_legacy_average_reference(self):
        raw, _ = _dsi_raw(n_bursts=3, noisy=("F3",))
        _, data, info, _ = self._load(raw, reference="average")
        np.testing.assert_allclose(data[0].mean(1), 0, atol=1e-9)  # all channels
        self.assertEqual(data.shape[-1], 17251)                      # legacy 2.5-60 s epoch
        self.assertEqual(info["bads"], ["F3"])


class TestLoadBySession(unittest.TestCase):
    def test_subjects_are_session_ids(self):
        with tempfile.TemporaryDirectory() as root:
            t0 = datetime(2026, 1, 5, 10, 0, tzinfo=timezone.utc)
            files = {}
            for i, (folder, name) in enumerate([("10hz_2ob_60s", "Alpha Beta"),
                                                ("15hz_3ob_60s", "Alpha  Betta")]):
                os.makedirs(os.path.join(root, folder))
                p = os.path.join(root, folder, f"{name}.edf")
                open(p, "w").close()
                files[p] = t0 + timedelta(minutes=10 * i)

            def fake_read(path, preload=False, **_):
                if not preload:
                    m = MagicMock()
                    m.info = {"meas_date": files[path]}
                    return m
                return _dsi_raw(n_bursts=3)[0]

            logs = []
            with patch("loader.mne.io.read_raw_edf", side_effect=fake_read):
                out = list(StudyLoader(log=logs.append).load(root))
        self.assertEqual([o[0] for o in out], ["P01", "P01"])
        self.assertEqual([float(o[1].carrier_frequency) for o in out], [10.0, 15.0])
        self.assertFalse(any("Alpha" in line for line in logs))


if __name__ == '__main__':
    unittest.main()
