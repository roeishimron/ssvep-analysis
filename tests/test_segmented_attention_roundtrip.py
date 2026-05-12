"""System test for the segmented-attention pipeline.

Builds two pure-sine source recordings (10 Hz and 15 Hz), merges them via
AttentionColorParser semantics (random segment partition + 50 Hz response-time
fillers between segments), parses the result, and asserts each output
condition equals the original source — sample-for-sample and spectrum-wise.
"""

import json
import unittest
from typing import Dict, List, Tuple

import numpy as np

from analysis import RawRecording, Spectral
from attention_metadata import AttentionColorParser
from core_types import AttentionColor


SAMPLE_RATE = 300.0
QUANTA_DURATION_S = 2.0
SAMPLES_PER_QUANTA = int(QUANTA_DURATION_S * SAMPLE_RATE)  # 600
N_TRIALS = 3
N_CHANNELS = 1
QUANTAS_PER_CONDITION_PER_TRIAL = 25
SAMPLES_PER_TRIAL = QUANTAS_PER_CONDITION_PER_TRIAL * SAMPLES_PER_QUANTA  # 15000


def _sine(freq_hz: float, n_samples: int) -> np.ndarray:
    t = np.arange(n_samples) / SAMPLE_RATE
    return np.sin(2.0 * np.pi * freq_hz * t).astype(np.float64)


def _make_recording(freq_hz: float) -> np.ndarray:
    """(1, 3, 1, 15000) pure sine. Each trial holds 50 s of the same sine."""
    s = _sine(freq_hz, SAMPLES_PER_TRIAL)
    out = np.empty((1, N_TRIALS, N_CHANNELS, SAMPLES_PER_TRIAL), dtype=np.float64)
    out[:] = s
    return out


def _wrap(data: np.ndarray, color: int) -> RawRecording:
    return RawRecording(
        name=f"sine_{color}",
        raw_data=data,
        sample_rate=SAMPLE_RATE,
        channel_names=["test_ch"],
        props=AttentionColor(color),
    )


def _random_partition(total: int, n_parts: int, rng: np.random.Generator) -> List[int]:
    """Partition `total` into `n_parts` positive integers (stars-and-bars)."""
    if n_parts == 1:
        return [total]
    cuts = sorted(rng.choice(range(1, total), size=n_parts - 1, replace=False).tolist())
    parts = [cuts[0]] + [cuts[i] - cuts[i - 1] for i in range(1, len(cuts))]
    parts.append(total - cuts[-1])
    return parts


def _build_merged(
    source_15hz: np.ndarray,
    source_10hz: np.ndarray,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, str]:
    """Merge two source recordings into one segmented recording.

    Per trial: random partition of 25 quantas of 15 Hz into N1 segments and
    25 quantas of 10 Hz into N2 segments; interleave segments randomly;
    between consecutive segments insert one 2 s quanta of 50 Hz filler.
    Per-trial-per-color cursors guarantee the same-color quantas, concatenated
    in timeline order, reconstruct the source's per-trial sequence exactly.

    Returns the merged (1, 3, 1, T_max) ndarray (zero-padded across trials) and
    the metadata JSON string consumed by AttentionColorParser.
    """
    filler_quanta = _sine(50.0, SAMPLES_PER_QUANTA)[np.newaxis, :]  # (1, 600)

    trial_timelines: List[np.ndarray] = []
    trials_meta: List[List[Dict[str, object]]] = []

    for trial_idx in range(N_TRIALS):
        n1 = int(rng.integers(2, 8))
        n2 = int(rng.integers(2, 8))
        quantas_plus = _random_partition(QUANTAS_PER_CONDITION_PER_TRIAL, n1, rng)
        quantas_minus = _random_partition(QUANTAS_PER_CONDITION_PER_TRIAL, n2, rng)

        segs: List[Tuple[int, int]] = (
            [(1, q) for q in quantas_plus] + [(-1, q) for q in quantas_minus]
        )
        rng.shuffle(segs)

        cursor_plus = 0
        cursor_minus = 0
        timeline_parts: List[np.ndarray] = []
        metadata_segs: List[Dict[str, object]] = []

        for i, (color, n) in enumerate(segs):
            if i > 0:
                timeline_parts.append(filler_quanta)
            n_samples = n * SAMPLES_PER_QUANTA
            if color == 1:
                chunk = source_15hz[0, trial_idx, :, cursor_plus:cursor_plus + n_samples]
                cursor_plus += n_samples
            else:
                chunk = source_10hz[0, trial_idx, :, cursor_minus:cursor_minus + n_samples]
                cursor_minus += n_samples
            timeline_parts.append(chunk)
            metadata_segs.append({
                "steady_state_quantas": int(n),
                "attention_color": int(color),
                "coherent_color": 1,
                "coherent_direction": 0.0,
            })

        trial_timelines.append(np.concatenate(timeline_parts, axis=-1))
        trials_meta.append(metadata_segs)

    max_len = max(t.shape[-1] for t in trial_timelines)
    merged = np.zeros((1, N_TRIALS, N_CHANNELS, max_len), dtype=np.float64)
    for i, t in enumerate(trial_timelines):
        merged[0, i, :, :t.shape[-1]] = t

    return merged, json.dumps(trials_meta)


class TestSegmentedAttentionRoundtrip(unittest.TestCase):
    def setUp(self) -> None:
        self.rng = np.random.default_rng(42)
        self.source_15hz = _make_recording(15.0)
        self.source_10hz = _make_recording(10.0)

    def _assert_pure_sine_spectrum(self, data: np.ndarray, freq_hz: float) -> None:
        rec = _wrap(data, color=0)
        spectral = Spectral(rec)
        freqs = spectral.frequencies()
        mean_power, _ = spectral.power_spectrum()

        normalized = mean_power / mean_power.max()
        signal_bin = int(np.argmin(np.abs(freqs - freq_hz)))
        np.testing.assert_allclose(normalized[signal_bin], 1.0, atol=1e-12)

        far_mask = np.abs(np.arange(len(freqs)) - signal_bin) > 4
        far_max = float(normalized[far_mask].max())
        self.assertLess(
            far_max, 1e-3,
            f"side-lobe leakage at {freq_hz}Hz: max normalized = {far_max:.6g}",
        )

    def test_phase1_pure_sine_recordings(self) -> None:
        """Each source recording's spectrum peaks at its frequency, ~0 elsewhere."""
        self._assert_pure_sine_spectrum(self.source_15hz, 15.0)
        self._assert_pure_sine_spectrum(self.source_10hz, 10.0)

    def test_phase3_roundtrip(self) -> None:
        """Merge → parse → each condition exactly equals its source recording."""
        merged, metadata = _build_merged(self.source_15hz, self.source_10hz, self.rng)

        results = dict(AttentionColorParser(sample_rate=SAMPLE_RATE).parse(metadata, merged))

        self.assertIn(AttentionColor(1), results)
        self.assertIn(AttentionColor(-1), results)

        np.testing.assert_array_equal(results[AttentionColor(1)], self.source_15hz)
        np.testing.assert_array_equal(results[AttentionColor(-1)], self.source_10hz)

        self._assert_pure_sine_spectrum(results[AttentionColor(1)], 15.0)
        self._assert_pure_sine_spectrum(results[AttentionColor(-1)], 10.0)

    def test_phase4_misalignment_0_5s_early(self) -> None:
        """Each segment starts 0.5 s earlier than the metadata declares.

        Simulated by shifting the whole merged recording 0.5 s (150 samples)
        backward relative to the metadata's cursor positions. The parser still
        slices at the metadata-claimed positions, so each segment's slice
        runs 0.5 s past its actual content — into the next thing in the
        timeline.

        Because the parser always interposes a response-time filler between
        consecutive segments, "the next thing" is always the 50 Hz filler
        — never the other condition's signal directly. We therefore expect:
          - signal frequency still dominant (peak normalized to 1.0)
          - 50 Hz filler leaks measurably (≳ 1e-4 of peak)
          - the OTHER condition's frequency stays at the Kaiser side-lobe floor
            (~2e-5; cross-condition contamination is shielded by the filler)
        """
        merged, metadata = _build_merged(self.source_15hz, self.source_10hz, self.rng)

        shift = int(0.5 * SAMPLE_RATE)  # 150 samples
        shifted = np.zeros_like(merged)
        shifted[..., :-shift] = merged[..., shift:]

        results = dict(AttentionColorParser(sample_rate=SAMPLE_RATE).parse(metadata, shifted))

        print()  # readable diagnostic output
        for cond, expected_freq, other_freq in [
            (AttentionColor(1), 15.0, 10.0),
            (AttentionColor(-1), 10.0, 15.0),
        ]:
            rec = _wrap(results[cond], color=cond.color)
            spectral = Spectral(rec)
            freqs = spectral.frequencies()
            mean_power, _ = spectral.power_spectrum()
            normalized = mean_power / mean_power.max()
            at = {
                f: float(normalized[int(np.argmin(np.abs(freqs - f)))])
                for f in (10.0, 15.0, 50.0)
            }
            print(f"  {cond}: normalized power @ 10/15/50 Hz = "
                  f"{at[10.0]:.6f} / {at[15.0]:.6f} / {at[50.0]:.6f}")

            # The condition's own frequency is still the spectrum peak.
            self.assertEqual(
                max(at, key=lambda f: at[f]), expected_freq,
                f"{cond}: expected {expected_freq}Hz to dominate, got {at}",
            )
            # The 50 Hz filler leaks measurably (≳ 1e-4 of peak).
            self.assertGreater(
                at[50.0], 1e-4,
                f"{cond}: expected filler leak at 50 Hz, got {at[50.0]:.6g}",
            )
            # The other condition's frequency stays at the side-lobe floor
            # because the filler always separates the conditions.
            self.assertLess(
                at[other_freq], 1e-4,
                f"{cond}: unexpected leak from other condition at "
                f"{other_freq}Hz, got {at[other_freq]:.6g}",
            )


if __name__ == "__main__":
    unittest.main()
