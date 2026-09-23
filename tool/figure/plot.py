"""Draws doc/images/trend.png from the CSV that figure.dart writes."""
import csv
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

rows = list(csv.DictReader(open(sys.argv[1])))


def series(kind):
    picked = [r for r in rows if r["kind"] == kind]
    t = [float(r["time"]) for r in picked]
    v = [float(r["value"]) for r in picked]
    lo = [float(r["lo"]) for r in picked if r["lo"]]
    hi = [float(r["hi"]) for r in picked if r["hi"]]
    return t, v, lo, hi


fig, ax = plt.subplots(figsize=(8, 3.6), dpi=150)
t, v, lo, hi = series("trend")
ax.fill_between(t, lo, hi, color="#4c72b0", alpha=0.25, linewidth=0,
                label="95% credible band")
ax.plot(t, v, color="#4c72b0", linewidth=1.8, label="smoothed trend")
t, v, lo, hi = series("forecast")
ax.fill_between(t, lo, hi, color="#dd8452", alpha=0.25, linewidth=0)
ax.plot(t, v, color="#dd8452", linewidth=1.8, linestyle="--", label="forecast")
t, v, _, _ = series("reading")
ax.scatter(t, v, s=9, color="#333333", zorder=3, label="readings")
ax.axvspan(45, 62, color="#999999", alpha=0.08, linewidth=0)
ax.text(53.5, max(v) + 0.1, "no readings", ha="center", fontsize=8, color="#666666")
ax.set_xlabel("day")
ax.set_ylabel("kg")
ax.set_xlim(0, 130)
ax.spines[["top", "right"]].set_visible(False)
ax.legend(loc="upper right", fontsize=8, frameon=False)
fig.tight_layout()
fig.savefig(sys.argv[2])
