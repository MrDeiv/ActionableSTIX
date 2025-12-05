import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import json

# ----------------------------
# Global style configuration
# ----------------------------
mpl.rcParams.update({
    "text.usetex": False,            # avoid external LaTeX
    "mathtext.fontset": "stix",      # math glyphs similar to Times
    "font.family": "STIXGeneral",    # STIX serif font
    "font.size": 12,
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "legend.fontsize": 8,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "pdf.fonttype": 42,              # embed TrueType fonts
    "ps.fonttype": 42,
    "figure.figsize": (10, 10),
})

# ----------------------------
# Sample data
# ----------------------------
data = json.load(open("ablation/mitre/mitre_coverage.json"))
labels = list(data.keys())
values = list(data.values())
total = sum(values)

# consistent color palette
colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B3"]

# ----------------------------
# Bar Chart
# ----------------------------
fig2, ax2 = plt.subplots()

bars = ax2.bar(labels, values, color=colors)

ax2.set_xlabel("MITRE Tactics")
ax2.set_ylabel("Related Attack Patterns")
ax2.set_title("MITRE Tactics Coverage", pad=20)
ax2.set_xticklabels(labels, rotation=45, ha="right")


# add raw values above bars
for bar, size in zip(bars, values):
    height = bar.get_height()
    percent = 100.0 * size / total
    ax2.text(
        bar.get_x() + bar.get_width() / 2,
        height + 5,
        f"{size}\n({percent:.1f}%)",   # show both count & percentage
        ha="center", va="bottom", fontsize=9
    )

ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)

plt.tight_layout()

# save bar chart
fig2.savefig("tactics_bar_chart.pdf", bbox_inches="tight", facecolor="white")

plt.show()
plt.savefig("ablation/mitre/mitre_plots.png", bbox_inches="tight")  # save as raster for quick preview
plt.savefig("ablation/mitre/mitre_plots.pdf", bbox_inches="tight", facecolor="white")  # save as vector for publication
plt.close()