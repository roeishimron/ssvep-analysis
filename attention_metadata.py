"""MetadataParser for the segmented attention paradigm.

Metadata JSON layout: a list with one entry per trial; each trial is a
list of segment dicts with keys `steady_state_quantas` (count of 2 s
windows), `attention_color` (+/-1, the condition key), `coherent_color`
and `coherent_direction` (segment-level metadata, ignored here).

Each input trial is split into one output trial per condition: within a
trial, quanta belonging to the same condition are concatenated in
timeline order. Between every pair of consecutive segments one 2 s
quanta is dropped from the timeline — the subject's response-time gap.
Per-condition quanta-counts must be the same across input trials so the
output array stays rectangular: per condition the result is shape
(1, n_input_trials, C, n_quantas_per_trial * samples_per_quanta).
"""

import json
from collections import defaultdict
from typing import Dict, Iterable, List, Tuple

import numpy as np

from core_types import AttentionColor, RawStudyData


class AttentionColorParser:
    """Parses the segmented attention metadata JSON, splitting by `attention_color`.

    Implements the `MetadataParser[AttentionColor]` Protocol from interfaces.py.
    Quanta size is 2 s; pass the recording's `sample_rate` so the parser
    can convert to samples.
    """

    def __init__(self, sample_rate: float = 300.0):
        self._samples_per_quanta = int(2.0 * sample_rate)

    def trial_durations_s(self, metadata: str) -> List[float]:
        """Per-trial durations in seconds: steady-state quantas + response-time gaps.

        Each segment of `steady_state_quantas=N` contributes N quantas; one
        further quanta is added per pair of consecutive segments for the
        response-time gap. Quanta is 2 s.
        """
        return [
            (sum(seg["steady_state_quantas"] for seg in segs) + max(len(segs) - 1, 0)) * 2.0
            for segs in json.loads(metadata)
        ]

    def parse(
        self,
        metadata: str,
        trial_data: RawStudyData,
    ) -> Iterable[Tuple[AttentionColor, RawStudyData]]:
        trials_meta = json.loads(metadata)
        s, t = trial_data.shape[:2]
        assert s == 1, f"expected single-subject array, got S={s}"
        assert t == len(trials_meta), (
            f"metadata declares {len(trials_meta)} trials, data has T={t}"
        )

        spq = self._samples_per_quanta
        per_trial: List[Dict[AttentionColor, List[np.ndarray]]] = []
        for trial_idx, segs in enumerate(trials_meta):
            trial = trial_data[0, trial_idx]
            cursor = 0
            collected: Dict[AttentionColor, List[np.ndarray]] = defaultdict(list)
            for seg_idx, seg in enumerate(segs):
                if seg_idx > 0:
                    cursor += spq  # response-time quanta, dropped
                key = AttentionColor(int(seg["attention_color"]))
                for _ in range(seg["steady_state_quantas"]):
                    collected[key].append(trial[:, cursor:cursor + spq])
                    cursor += spq
            per_trial.append(collected)

        all_keys = sorted(set().union(*(d.keys() for d in per_trial)))
        for key in all_keys:
            counts = [len(d.get(key, [])) for d in per_trial]
            assert all(c == counts[0] for c in counts), (
                f"quanta count for {key} varies across trials: {counts}"
            )

        for key in all_keys:
            stacked = np.stack(
                [np.concatenate(d[key], axis=-1) for d in per_trial],
                axis=0,
            )
            yield key, stacked[np.newaxis, ...].astype(np.float64)
