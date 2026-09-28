"""Draws the package icon, the pub.dev screenshot and the GitHub social preview.

Takes the CSV that icon.dart writes, the CSV that figure.dart writes, and the
directory to write into.
"""
import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

BLUE = "#4c72b0"
ORANGE = "#dd8452"
INK = "#333333"
TILE = "#f7f8fb"

icon_rows = list(csv.DictReader(open(sys.argv[1])))
trend_rows = list(csv.DictReader(open(sys.argv[2])))
out = Path(sys.argv[3])


def series(rows, kind):
    picked = [r for r in rows if r["kind"] == kind]
    t = [float(r["time"]) for r in picked]
    v = [float(r["value"]) for r in picked]
    lo = [float(r["lo"]) for r in picked if r["lo"]]
    hi = [float(r["hi"]) for r in picked if r["hi"]]
    return t, v, lo, hi


def mark(ax, rows, line, dot, gap=None):
    """Band, trend, forecast cone and readings, with no axes."""
    if gap:
        ax.axvspan(*gap, color="#999999", alpha=0.08, linewidth=0)
    t, v, lo, hi = series(rows, "trend")
    ax.fill_between(t, lo, hi, color=BLUE, alpha=0.28, linewidth=0)
    ax.plot(t, v, color=BLUE, linewidth=line, solid_capstyle="round")
    t, v, lo, hi = series(rows, "forecast")
    ax.fill_between(t, lo, hi, color=ORANGE, alpha=0.28, linewidth=0)
    ax.plot(t, v, color=ORANGE, linewidth=line, linestyle=(0, (1.6, 1.4)),
            dash_capstyle="round")
    t, v, _, _ = series(rows, "reading")
    ax.scatter(t, v, s=dot, color=INK, zorder=3, linewidths=0)
    ax.set_axis_off()


def emblem(fig, left, bottom, size):
    """The icon: the mark on a rounded light square, placed and sized in inches.

    The tile keeps the mark readable on dark pages too. Lines and dots scale
    with the tile, so the icon and the social preview draw the same picture.
    """
    width, height = fig.get_size_inches()
    rect = (left / width, bottom / height, size / width, size / height)
    fig.patches.append(FancyBboxPatch(
        rect[:2], rect[2], rect[3],
        boxstyle=f"round,pad=0,rounding_size={0.18 * rect[2]}",
        mutation_aspect=width / height, transform=fig.transFigure,
        facecolor=TILE, edgecolor="none", zorder=-1))
    ax = fig.add_axes((
        (left + 0.088 * size) / width, (bottom + 0.118 * size) / height,
        0.824 * size / width, 0.764 * size / height))
    ax.patch.set_alpha(0)
    scale = size / 4.352
    mark(ax, icon_rows, line=7 * scale, dot=420 * scale ** 2)
    ax.set_xlim(-0.4, 10)


def icon(path):
    fig = plt.figure(figsize=(5.12, 5.12), dpi=100)
    fig.patch.set_alpha(0)
    emblem(fig, 0, 0, 5.12)
    fig.savefig(path, transparent=True, metadata={"Date": None})
    plt.close(fig)


icon(out / "icon.png")
icon(out / "icon.svg")

# pub.dev shows screenshots small, so this one drops the axis text and legend
# of trend.png and keeps the readings, the gap, the band and the forecast.
fig = plt.figure(figsize=(6, 6), dpi=160)
fig.patch.set_facecolor("white")
ax = fig.add_axes((0.04, 0.06, 0.92, 0.88))
mark(ax, trend_rows, line=2.6, dot=26, gap=(45, 62))
ax.set_xlim(-1, 130)
fig.savefig(out / "screenshot.png")
plt.close(fig)

# GitHub's social preview: 1280 x 640, the mark on the left, the name and one
# line on the right.
fig = plt.figure(figsize=(12.8, 6.4), dpi=100)
fig.patch.set_facecolor("white")
emblem(fig, 0.768, 1.024, 4.352)
fig.text(0.45, 0.54, "state_space", fontsize=64, fontfamily="monospace",
         fontweight="bold", color=INK, va="bottom")
fig.text(0.45, 0.44,
         "Smooth trends with uncertainty bands\n"
         "for irregular time series. Pure Dart.",
         fontsize=22, color="#555555", va="top", linespacing=1.4)
fig.savefig(out / "social-preview.png")
plt.close(fig)
