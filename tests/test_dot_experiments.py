import unittest
from unittest.mock import patch

import mne
import numpy as np

from core_types import AttentionFrequency
from experiments import compose_folders, dot_experiments
from loader import StudyLoader


def _info(ch_names=("O1", "O2", "P3"), sfreq=100.0):
    return mne.create_info(ch_names=list(ch_names), sfreq=sfreq, ch_types="eeg")


class TestComposeFoldersGenericInK(unittest.TestCase):
    """compose_folders preserves the user-supplied key type K (no widening to Hashable)."""

    def test_preserves_typed_keys(self):
        info = _info()
        # synthetic per-folder loader: yields one fake subject per folder.
        def fake_load(folder: str):
            yield (
                f"subj_from_{folder}",
                info,
                100.0,
                np.zeros((1, 3, 3, 200), dtype=np.float64),
            )

        keyed = (
            (AttentionFrequency(np.float64(10.0)), "folderA"),
            (AttentionFrequency(np.float64(15.0)), "folderB"),
        )
        out = list(compose_folders(keyed, fake_load))
        self.assertEqual(len(out), 2)
        self.assertEqual(out[0][1], AttentionFrequency(np.float64(10.0)))
        self.assertEqual(out[1][1], AttentionFrequency(np.float64(15.0)))


class TestDotExperimentsWiring(unittest.TestCase):
    """End-to-end wiring: dot_experiments → compose_folders → Study."""

    def test_produces_two_attention_conditions(self):
        info = _info()

        def fake_load_folder(folder: str):
            # Same three subjects in both folders, three trials each.
            for subj in ("alice", "bob", "carol"):
                yield (
                    subj,
                    info,
                    100.0,
                    np.random.rand(1, 3, 3, 200).astype(np.float64),
                )

        with patch.object(StudyLoader, "load_folder", side_effect=fake_load_folder):
            exp = dot_experiments("./fake")

        self.assertEqual(set(exp.subjects().keys()), {"alice", "bob", "carol"})
        self.assertEqual(
            set(exp.conditions().keys()),
            {AttentionFrequency(np.float64(10.0)), AttentionFrequency(np.float64(15.0))},
        )
        # Each subject participated in both conditions.
        for subj in exp.subjects().values():
            self.assertEqual(len(subj.conditions()), 2)

    def test_view_props_returns_attention_frequency(self):
        """The typed key flows through to view.props() — this is what makes
        SSVEPAnalysis(view) a static type error for non-SSVEP recordings."""
        info = _info()

        def fake_load_folder(folder: str):
            yield (
                "alice", info, 100.0,
                np.random.rand(1, 3, 3, 200).astype(np.float64),
            )

        with patch.object(StudyLoader, "load_folder", side_effect=fake_load_folder):
            exp = dot_experiments("./fake")

        for key, view in exp.conditions().items():
            self.assertIsInstance(view.props(), AttentionFrequency)
            self.assertEqual(view.props(), key)


if __name__ == "__main__":
    unittest.main()
