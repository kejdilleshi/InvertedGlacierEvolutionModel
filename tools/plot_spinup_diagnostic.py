"""Figure: does the model reach the 1880 state if run from the 2017 geometry?

Uses the numbers from the spin-up test (hardcoded below -- it's a 3-minute,
4-point diagnostic, not worth re-running through a JSON file). Regenerate with:

    for t in 1700 1600 1500 1400; do ...constant-T run to 1880, record vol/terminus...; done
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = "results/lia_1700_2017/spinup_diagnostic.png"

# t_start, years of spin-up, volume (km3), terminus northing (km)
rows = [
    (1700, 180, 18.77, 5137.45),
    (1600, 280, 20.05, 5136.35),
    (1500, 380, 20.67, 5135.65),
    (1400, 480, 20.93, 5135.65),
]
target_vol, target_term = 20.67, 5136.45

t_start = [r[0] for r in rows]
years = [r[1] for r in rows]
vol = [r[2] for r in rows]
term = [r[3] for r in rows]

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)

ax = axes[0]
ax.plot(years, vol, "o-", color="#1f77b4", markersize=8, linewidth=2)
ax.axhline(target_vol, color="k", linestyle="--", linewidth=1,
          label=f"observed 1880 volume ({target_vol:.1f} km³)")
for y, v, ts in zip(years, vol, t_start):
    ax.annotate(f"{ts}", (y, v), textcoords="offset points", xytext=(0, 8),
               ha="center", fontsize=9)
ax.set_xlabel("Years of spin-up before 1880 (start year annotated)")
ax.set_ylabel("Simulated volume at 1880 (km³)")
ax.set_title("Ice volume needs ~280 years to catch up\nstarting from the 2017 geometry")
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

ax = axes[1]
offset = [t - target_term for t in term]
ax.plot(years, offset, "o-", color="#d62728", markersize=8, linewidth=2)
ax.axhline(0, color="k", linestyle="--", linewidth=1, label="observed terminus")
for y, o, ts in zip(years, offset, t_start):
    ax.annotate(f"{ts}", (y, o), textcoords="offset points", xytext=(0, 8),
               ha="center", fontsize=9)
ax.set_xlabel("Years of spin-up before 1880 (start year annotated)")
ax.set_ylabel("Terminus offset from observed (km, +up-valley)")
ax.set_title("A short spin-up leaves the tongue\ntoo short (too far up-valley)")
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

fig.suptitle("Spin-up test: constant T=7°C, precip=1.8 m/yr, run to 1880, "
            "started from the 2017 geometry", fontsize=11)
fig.savefig(OUT, dpi=150, bbox_inches="tight")
print("wrote", OUT)
