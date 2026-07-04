"""Does a smaller rectangular window give better SNR?  Self-contained sweep.

Validates/refutes the claim: with NO taper (rectangular window), lowering the
window size raises SNR ("kills more noise without harming the signal").

Own FFT at each window size (independent of analysis.py). Three sources:
  - generated stationary tone  (prediction: SNR ~flat vs ws)
  - generated drifting tone     (prediction: smaller ws -> higher SNR)
  - real data (target 5 Hz, carrier 20 Hz)
Neighbor-bin SNR is confounded by resolution (coarser bins at small ws), so a
fixed-Hz-band noise version is also reported.

Run: MPLBACKEND=Agg python compare_snr_window.py
"""
from __future__ import annotations
import contextlib, io
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from core import Study
from core_types import ConditionProperties
from loader import StudyLoader

FS = 300.0
F_T, CARRIER = 5.0, 20.0
WS_LIST = [0.2, 0.4, 0.8, 1.0, 2.0, 4.0]          # seconds; all keep 5/10/15/20 on-bin
ELECTRODES = ["T5", "T6", "O1", "O2"]
OUT = Path("figures/comparison/compare-snr-vs-window-size.png")


def coherent_power(trial_sig, ws_samp):
    """Cut one trial into non-overlapping rectangular windows, coherently average
    (complex mean of rffts), return the power spectrum |mean|**2."""
    n = len(trial_sig) // ws_samp
    if n < 1:
        return None
    seg = trial_sig[:n * ws_samp].reshape(n, ws_samp).astype(float)
    seg = seg - seg.mean(axis=1, keepdims=True)
    F = np.fft.rfft(seg, axis=1)
    return np.abs(F.mean(axis=0)) ** 2


def _neighbor_snr(power, b, n_nb=3, n_skip=1):
    lo = power[max(0, b - n_skip - n_nb): max(0, b - n_skip)]
    hi = power[b + n_skip + 1: b + n_skip + 1 + n_nb]
    nb = np.concatenate([lo, hi])
    return power[b] / np.mean(nb) if nb.size and np.mean(nb) > 0 else np.nan


def _band_snr(power, b, res, lo_hz=1.5, hi_hz=4.0):
    """SNR vs a FIXED Hz band [lo,hi] on each side (comparable across ws)."""
    lo_b, hi_b = int(round(lo_hz / res)), int(round(hi_hz / res))
    left = power[max(0, b - hi_b): max(0, b - lo_b)]
    right = power[b + lo_b: b + hi_b + 1]
    nb = np.concatenate([left, right])
    return power[b] / np.mean(nb) if nb.size and np.mean(nb) > 0 else np.nan


def sweep(trials, f, band=False):
    """trials: (T, time). Return SNR at f for each ws (avg power over trials)."""
    out = []
    for ws in WS_LIST:
        wss = int(round(ws * FS)); res = FS / wss
        b = round(f / res)
        if abs(b - f / res) > 1e-6:   # off-bin: skip honestly
            out.append(np.nan); continue
        powers = [p for tr in trials if (p := coherent_power(tr, wss)) is not None]
        if not powers:
            out.append(np.nan); continue
        P = np.mean(powers, axis=0)
        if b >= len(P) - 4:
            out.append(np.nan); continue
        out.append(_band_snr(P, b, res) if band else _neighbor_snr(P, b))
    return out


# ---- generated ----
def gen_trials(drift_rad, noise_sd, n_trials=3, dur_s=57.5, seed=0):
    rng = np.random.default_rng(seed)
    n = int(dur_s * FS); t = np.arange(n) / FS
    trials = []
    for k in range(n_trials):
        drift = np.linspace(0, drift_rad, n)          # linear phase drift across trial
        onset = rng.uniform(-0.02, 0.02)              # per-trial onset jitter
        sig = (np.cos(2 * np.pi * F_T * (t - onset) + drift)
               + np.cos(2 * np.pi * CARRIER * (t - onset) + drift))
        sig = sig + rng.normal(0, noise_sd, n)
        trials.append(sig)
    return np.array(trials)


# ---- real ----
def real_sweep(study, f, ref_ws=2.0):
    reqs = {ConditionProperties(np.float64(F_T), np.float64(c)) for c in (10., 15., 20.)}
    curves, bandcurves = [], []
    carrier = 20.0 if f == CARRIER else 20.0
    for sub in study.filter_subjects(reqs):
        try:
            rec = sub[ConditionProperties(np.float64(F_T), np.float64(carrier))].take_channels(ELECTRODES)
        except Exception:
            continue
        data = rec.raw_data()[0]  # (T, C, time)
        # pick best channel for f at the reference window size
        wss = int(round(ref_ws * FS)); res = FS / wss; b = round(f / res)
        best, bestsnr = 0, -1
        for c in range(data.shape[1]):
            s = _neighbor_snr(np.mean([coherent_power(data[tr, c], wss) for tr in range(data.shape[0])], axis=0), b)
            if np.isfinite(s) and s > bestsnr:
                bestsnr, best = s, c
        trials = data[:, best, :]
        curves.append(sweep(trials, f))
        bandcurves.append(sweep(trials, f, band=True))
    return np.array(curves), np.array(bandcurves)


def main():
    print("generated...")
    stat = gen_trials(0.0, 3.0, seed=1)
    drift = gen_trials(6 * np.pi, 3.0, seed=2)     # ~3 full turns over the trial
    g_stat = sweep(stat, F_T); g_drift = sweep(drift, F_T)
    g_stat_c = sweep(stat, CARRIER); g_drift_c = sweep(drift, CARRIER)

    print("real...")
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        study = Study(StudyLoader().load("data/hebrew_vs_mirror"))
        rt, rt_b = real_sweep(study, F_T)
        rc, rc_b = real_sweep(study, CARRIER)
    rt_m = np.nanmedian(rt, axis=0); rc_m = np.nanmedian(rc, axis=0)
    rt_bm = np.nanmedian(rt_b, axis=0); rc_bm = np.nanmedian(rc_b, axis=0)

    def row(name, v):
        print(f"  {name:<28}" + "  ".join(f"{x:6.1f}" if np.isfinite(x) else "   nan" for x in v))
    print("\nSNR vs window size (ws, s):    " + "  ".join(f"{w:6.2f}" for w in WS_LIST))
    row("GEN stationary  (5Hz)", g_stat)
    row("GEN drifting    (5Hz)", g_drift)
    row("GEN stationary  (carrier)", g_stat_c)
    row("GEN drifting    (carrier)", g_drift_c)
    row("REAL target 5Hz (neighbor)", rt_m)
    row("REAL target 5Hz (fixed band)", rt_bm)
    row("REAL carrier    (neighbor)", rc_m)
    row("REAL carrier    (fixed band)", rc_bm)

    fig, (a, b) = plt.subplots(1, 2, figsize=(13, 5.5), label="compare-snr-vs-window-size")
    for ax, tt in ((a, "Target 5 Hz"), (b, "Carrier 20 Hz")):
        ax.set_title(tt); ax.set_xlabel("window size (s)"); ax.set_ylabel("SNR")
        ax.set_xscale("log"); ax.grid(True, which="both", ls="--", alpha=0.4)
    a.plot(WS_LIST, g_stat, "o-", color="#1E88E5", label="GEN stationary (predict flat)")
    a.plot(WS_LIST, g_drift, "o-", color="#D81B60", label="GEN drifting")
    a.plot(WS_LIST, rt_m, "s--", color="#2E7D32", label="REAL (neighbor-bin SNR)")
    a.plot(WS_LIST, rt_bm, "^:", color="#6A1B9A", label="REAL (fixed-band SNR)")
    b.plot(WS_LIST, g_stat_c, "o-", color="#1E88E5", label="GEN stationary")
    b.plot(WS_LIST, g_drift_c, "o-", color="#D81B60", label="GEN drifting")
    b.plot(WS_LIST, rc_m, "s--", color="#2E7D32", label="REAL (neighbor-bin)")
    b.plot(WS_LIST, rc_bm, "^:", color="#6A1B9A", label="REAL (fixed-band)")
    a.legend(fontsize=8); b.legend(fontsize=8)
    fig.suptitle("SNR vs window size (rectangular): is smaller better?", weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=160, bbox_inches="tight")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
