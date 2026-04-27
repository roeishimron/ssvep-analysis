"""
Composite "SSVEP Phase Ambiguity Resolution" figure.

Three rows:
  Row 1 — Step 1: extract the two positive 10 Hz carrier peaks inside one
          200 ms (5 Hz oddball) cycle and project them onto a discrete
          timeline as T1, T2.
  Row 2 — Step 2: place the same markers on a 200 ms clock face, apply a
          fixed cortical-arrival delay, draw candidate processing-time arcs
          to the 150 ms target, highlight the shortest (≈50 ms) as the
          biological latency.
  Row 3 — Step 3: same construction for 15 Hz (3 markers) and 20 Hz
          (4 markers); demonstrate that the rule generalises and that
          loser-arc crowding gets worse at higher rates.

Writes:
  ../documents/figures/phase-ambiguity.pdf  (vector, embedded in seminarion.lyx)
  ../documents/figures/phase-ambiguity.png  (high-DPI raster preview)

Run:
  MPLBACKEND=Agg python figure_phase_ambiguity.py
"""
from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Arc, Circle

# ---------------------------------------------------------------------------
# Parameters (single source of truth)
# ---------------------------------------------------------------------------
CYCLE_MS        = 200            # 5 Hz oddball cycle
TARGET_MS       = 150            # target-response phase
BIO_DELAY_MS    = 30             # cortical-arrival shift (Rows 2–3)
CARRIER_HZ_ROW1 = 10
PHYSICAL_10HZ   = (70.0, 170.0)  # → after +30 ms shift: (100, 0) → winner 50 ms

# Row 3 markers, picked so (a) the winner is a clean ≈50 ms arc and (b) no loser
# arc is below ~30 ms (which would otherwise be picked by argmin and is anyway
# biologically implausible at these latencies).
SHIFTED_15HZ = (100.0, 33.333, 166.667)        # arcs: 50, 116.7, 183.3 → winner 50 ms
SHIFTED_20HZ = (105.0, 55.0, 5.0, 155.0)       # arcs: 45,  95, 145, 195 → winner 45 ms

# Palette
C_TARGET   = "#D81B60"   # rose      — target wave + target marker (all rows)
C_CARRIER  = "#1E88E5"   # blue      — carrier wave + carrier markers in Rows 1 & 2
C_WINDOW   = "#FFE0B2"   # warm pale — Row-1 analytical-window highlight

# Row-3 only: each candidate option gets its own color (winner is additionally
# picked out by a thicker solid stroke; losers keep their own colour but dashed
# and semi-transparent so the reader can still trace them).
OPTION_COLORS = ["#1976D2",  # blue
                 "#FB8C00",  # orange
                 "#388E3C",  # green
                 "#7B1FA2"]  # purple

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def time_to_angle_deg(t_ms: float) -> float:
    """Clock convention: 12 o'clock = 0 ms, clockwise = forward time."""
    return 90.0 - (t_ms / CYCLE_MS) * 360.0


def time_to_xy(t_ms: float, r: float = 1.0) -> tuple[float, float]:
    a = np.deg2rad(time_to_angle_deg(t_ms))
    return r * np.cos(a), r * np.sin(a)


def draw_clock_frame(ax, *, label_ticks: bool = True) -> None:
    """Unit circle with light hour-tick marks every 50 ms (=quarter cycle).

    Tick labels are placed *inside* the circle so the outside is reserved for
    carrier/target marker labels.
    """
    ax.add_patch(Circle((0, 0), 1.0, fill=False, lw=1.0, color="black"))
    for t in (0, 50, 100, 150):
        x_in, y_in = time_to_xy(t, 0.94)
        x_out, y_out = time_to_xy(t, 1.06)
        ax.plot([x_in, x_out], [y_in, y_out], color="black", lw=0.8)
        if label_ticks:
            x_lab, y_lab = time_to_xy(t, 0.82)
            ax.text(x_lab, y_lab, f"{t}", ha="center", va="center",
                    fontsize=7, color="dimgray", style="italic")
    ax.set_xlim(-1.55, 1.55)
    ax.set_ylim(-1.45, 1.45)
    ax.set_aspect("equal")
    ax.axis("off")


def time_arc(ax, t_start_ms: float, t_end_ms: float, *,
             radius: float, color: str, lw: float, ls: str = "-",
             alpha: float = 1.0, zorder: int = 3) -> None:
    """Arc along the clock from t_start_ms to t_end_ms in the forward (CW) time direction."""
    th_start = time_to_angle_deg(t_start_ms)
    th_end   = time_to_angle_deg(t_end_ms)
    # mpl Arc draws CCW from theta1 to theta2; to render a CW (forward time) sweep
    # from start → end we set theta1=end, theta2=start.
    arc = Arc((0, 0), 2 * radius, 2 * radius,
              theta1=th_end, theta2=th_start,
              color=color, lw=lw, linestyle=ls, alpha=alpha, zorder=zorder)
    ax.add_patch(arc)


def add_marker(ax, t_ms: float, *, color: str, label: str | None = None,
               label_offset: float = 0.22, marker: str = "o", size: float = 70,
               zorder: int = 5) -> None:
    x, y = time_to_xy(t_ms, 1.0)
    ax.scatter([x], [y], s=size, color=color, zorder=zorder,
               edgecolor="white", linewidth=1.2)
    if label:
        x_lab, y_lab = time_to_xy(t_ms, 1.0 + label_offset)
        # Anchor text so it grows AWAY from the circle, never into the marker.
        a = np.deg2rad(time_to_angle_deg(t_ms))
        ha = "left" if np.cos(a) > 0.15 else "right" if np.cos(a) < -0.15 else "center"
        va = "bottom" if np.sin(a) > 0.15 else "top" if np.sin(a) < -0.15 else "center"
        ax.text(x_lab, y_lab, label, ha=ha, va=va,
                fontsize=9, color=color, weight="bold")


# ---------------------------------------------------------------------------
# Row 1 — extraction
# ---------------------------------------------------------------------------

def draw_row1(ax) -> None:
    """Continuous waves + shaded window + plumb lines to discrete T1/T2 timeline."""
    t = np.linspace(0, 400, 4001)

    # 5 Hz target wave: peak at 150 ms.  sin(2π·5·t/1000 + φ) peaks where the arg=π/2.
    # Choose φ such that 2π·5·150/1000 + φ = π/2 → φ = π/2 - 1.5π = -π. Use φ=π (== -π mod 2π).
    target_wave = np.sin(2 * np.pi * 5 * t / 1000 + np.pi)
    # 10 Hz carrier wave: peaks at 70 ms and 170 ms inside the window.
    # 2π·10·70/1000 + φ = π/2 → φ = π/2 - 1.4π = -0.9π
    carrier_wave = np.sin(2 * np.pi * 10 * t / 1000 - 0.9 * np.pi)

    # Shade the analytical window (one 5 Hz cycle).
    ax.axvspan(0, CYCLE_MS, color=C_WINDOW, alpha=0.55, lw=0, zorder=1)

    # Waves
    ax.plot(t, target_wave + 1.6, color=C_TARGET, lw=2.0, zorder=2)
    ax.plot(t, carrier_wave - 1.6, color=C_CARRIER, lw=2.0, zorder=2)

    # Inline labels for the waves — placed clear of each wave's swing range
    ax.text(398, 2.7, "5 Hz target", ha="right", va="bottom",
            color=C_TARGET, fontsize=10, weight="bold")
    ax.text(398, -2.7, "10 Hz carrier", ha="right", va="top",
            color=C_CARRIER, fontsize=10, weight="bold")

    # Target-peak marker + plumb line on the 5 Hz wave
    discrete_y = -3.6
    target_peak_y = 1.6 + 1.0  # target baseline = +1.6, amplitude = 1
    ax.scatter([TARGET_MS], [target_peak_y], s=85, color=C_TARGET,
               edgecolor="white", linewidth=1.2, zorder=5)
    ax.plot([TARGET_MS, TARGET_MS], [target_peak_y, discrete_y + 0.05],
            color=C_TARGET, lw=1.0, linestyle=(0, (3, 3)), alpha=0.9, zorder=3)
    ax.scatter([TARGET_MS], [discrete_y], s=95, color=C_TARGET,
               edgecolor="white", linewidth=1.2, zorder=6)
    ax.text(TARGET_MS, discrete_y - 0.45, "target", ha="center", va="top",
            color=C_TARGET, fontsize=10, weight="bold")

    # Carrier-peak markers + plumb lines
    for tp, name in zip(PHYSICAL_10HZ, ("$T_1$", "$T_2$")):
        peak_y = -1.6 + 1.0  # carrier amplitude is 1, baseline is -1.6
        ax.scatter([tp], [peak_y], s=70, color=C_CARRIER,
                   edgecolor="white", linewidth=1.2, zorder=5)
        # Dashed plumb line down to the discrete timeline
        ax.plot([tp, tp], [peak_y, discrete_y + 0.05],
                color=C_CARRIER, lw=1.0, linestyle=(0, (3, 3)), alpha=0.85, zorder=3)
        # Discrete-timeline marker
        ax.scatter([tp], [discrete_y], s=85, color=C_CARRIER,
                   edgecolor="white", linewidth=1.2, zorder=6)
        ax.text(tp, discrete_y - 0.45, name, ha="center", va="top",
                color=C_CARRIER, fontsize=11, weight="bold")

    # Discrete timeline baseline
    ax.plot([0, 400], [discrete_y, discrete_y], color="black", lw=1.0, zorder=4)
    for tick in (0, 50, 100, 150, 200, 250, 300, 350, 400):
        ax.plot([tick, tick], [discrete_y - 0.08, discrete_y + 0.08],
                color="black", lw=0.8, zorder=4)
        ax.text(tick, discrete_y + 0.18, str(tick), ha="center", va="bottom",
                fontsize=7, color="dimgray")
    ax.text(400, discrete_y - 0.9, "ms", ha="right", va="top",
            fontsize=9, color="dimgray")

    # Window annotation
    ax.text(CYCLE_MS / 2, 3.05, "one analytical cycle (200 ms)",
            ha="center", va="bottom", fontsize=9, color="#8E5A1F", style="italic")

    ax.set_xlim(-5, 415)
    ax.set_ylim(-4.8, 3.4)
    ax.set_title("Step 1 — Extracting carrier onsets (10 Hz)",
                 fontsize=12, weight="bold", loc="left", pad=4)
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)


# ---------------------------------------------------------------------------
# Row 2 — biological correction
# ---------------------------------------------------------------------------

def draw_clock_with_arcs(ax, *, target_ms: float, carriers_ms: tuple[float, ...],
                         carrier_label_prefix: str = "T",
                         arc_radius: float = 0.78,
                         per_option_colors: bool = False) -> None:
    """Clock face with target + N carrier markers + processing-time arcs.

    Arc winner = shortest forward arc (target − carrier) mod CYCLE_MS. The
    winner gets a thick solid stroke; losers are thin/dashed.

    If `per_option_colors` is True, each candidate option is drawn in its own
    colour from `OPTION_COLORS` (marker AND arc share the colour) so the reader
    can trace each option independently. Otherwise all carriers use C_CARRIER
    and the winner arc is green while losers are gray.
    """
    draw_clock_frame(ax)
    add_marker(ax, target_ms, color=C_TARGET, label="target", label_offset=0.22)

    arcs = [(target_ms - c) % CYCLE_MS for c in carriers_ms]
    winner_index = int(np.argmin(arcs))

    for i, c in enumerate(carriers_ms):
        arc_ms = arcs[i]
        is_winner = (i == winner_index)
        if per_option_colors:
            col = OPTION_COLORS[i % len(OPTION_COLORS)]
            marker_col = col
            arc_col = col
        else:
            marker_col = C_CARRIER
            arc_col = "#2E7D32" if is_winner else "#616161"
        # stagger loser arc radii so overlapping arcs don't visually merge
        r_offset = 0.0 if is_winner else 0.04 * (i + 1)
        time_arc(ax, c, target_ms,
                 radius=arc_radius - r_offset,
                 color=arc_col,
                 lw=3.4 if is_winner else 1.5,
                 ls="-" if is_winner else (0, (5, 3)),
                 alpha=1.0 if is_winner else 0.6,
                 zorder=4 if is_winner else 3)
        add_marker(ax, c, color=marker_col,
                   label=f"${carrier_label_prefix}_{i+1}$" if carrier_label_prefix else None,
                   label_offset=0.20)
    # Only the winner's arc carries a numeric label, placed near the arc's
    # midpoint so it's unambiguous which arc the number refers to. Loser arcs
    # remain colour-coded but unlabelled — keeps the inside of the clock from
    # collapsing into a soup of mid-arc text.
    w_c = carriers_ms[winner_index]
    w_arc = arcs[winner_index]
    mid_ms = (w_c + w_arc / 2) % CYCLE_MS
    xa, ya = time_to_xy(mid_ms, arc_radius - 0.16)
    win_col = OPTION_COLORS[winner_index % len(OPTION_COLORS)] if per_option_colors else "#2E7D32"
    ax.text(xa, ya, f"{w_arc:.0f} ms", ha="center", va="center",
            fontsize=10, color=win_col, weight="bold")


def draw_clock_panel(ax, *, carriers_ms: tuple[float, ...],
                     freq_hz: int, label_prefix: str = "C") -> None:
    """One clock for an N-carrier condition (used in the unified Step 2)."""
    draw_clock_with_arcs(ax, target_ms=TARGET_MS, carriers_ms=carriers_ms,
                         carrier_label_prefix=label_prefix,
                         per_option_colors=True)
    ax.set_title(f"{freq_hz} Hz carrier — N = {len(carriers_ms)}",
                 fontsize=11, weight="bold", loc="center", pad=2)


# ---------------------------------------------------------------------------
# Compose the figure
# ---------------------------------------------------------------------------

def build_figure() -> plt.Figure:
    # Two-row layout: extraction on top, three clocks (10/15/20 Hz) below.
    fig = plt.figure(figsize=(9.5, 6.6), constrained_layout=False)
    gs = fig.add_gridspec(2, 3, height_ratios=[1.05, 1.0],
                          hspace=0.40, wspace=0.10,
                          left=0.03, right=0.99, top=0.93, bottom=0.03)

    ax_row1 = fig.add_subplot(gs[0, :])
    draw_row1(ax_row1)

    ax_10 = fig.add_subplot(gs[1, 0])
    ax_15 = fig.add_subplot(gs[1, 1])
    ax_20 = fig.add_subplot(gs[1, 2])
    draw_clock_panel(ax_10, carriers_ms=PHYSICAL_10HZ,
                     freq_hz=10, label_prefix="T")
    draw_clock_panel(ax_15, carriers_ms=SHIFTED_15HZ,
                     freq_hz=15, label_prefix="C")
    draw_clock_panel(ax_20, carriers_ms=SHIFTED_20HZ,
                     freq_hz=20, label_prefix="C")
    fig.text(0.5, 0.49,
             "Step 2 — Mapping onsets to phase & picking the shortest arc (10 Hz, 15 Hz, 20 Hz)",
             ha="center", va="bottom", fontsize=11, weight="bold")

    return fig


def main() -> None:
    out_dir = Path(__file__).resolve().parent.parent / "documents" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    fig = build_figure()
    pdf_path = out_dir / "phase-ambiguity.pdf"
    png_path = out_dir / "phase-ambiguity.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=180, bbox_inches="tight")
    print(f"wrote {pdf_path}")
    print(f"wrote {png_path}")


if __name__ == "__main__":
    main()
