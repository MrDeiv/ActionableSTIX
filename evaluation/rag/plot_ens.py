import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import colormaps
import numpy as np
import json

# --- Publication-style defaults without LaTeX ---
mpl.rcParams.update({
    "text.usetex": False,                 # avoid external LaTeX
    "mathtext.fontset": "stix",           # math glyphs similar to Times
    "font.family": "STIXGeneral",         # STIX serif font
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
})

# Embed fonts in vector outputs for publisher workflows
mpl.rcParams["pdf.fonttype"] = 42    # TrueType in PDF
mpl.rcParams["ps.fonttype"]  = 42

results_ens = json.load(open("abl_rag_ens.json"))

x = [r["weights"] for r in results_ens]
y1 = [r["non_llm_context_precision_with_reference"] for r in results_ens]
y2 = [r["non_llm_context_recall"] for r in results_ens]

fig, ax = plt.subplots(figsize=(3.5, 2.5))  # single-column size

width = 0.28
x_pos = np.arange(len(x))

# Black & white friendly bars
ax.bar(x_pos - width, y1, width,
       label='Precision',
       color='green')

ax.bar(x_pos, y2, width,
       label='Recall',
       color='red')

# F1: white with black edge for visibility
y_f1 = [2*(p*r)/(p+r) if (p+r) > 0 else 0 for p, r in zip(y1, y2)]
ax.bar(x_pos + width, y_f1, width,
       label='F1',
       color="blue",
       linewidth=1)

# Tick labels
ax.set_xticks(x_pos)
ax.set_xticklabels([f"({w[0]:.1f},{w[1]:.1f})" for w in x])

ax.set_xlabel(r"$(w_{bm25},w_{vdb})$")
ax.grid(True, alpha=0.3)

# Move legend outside to avoid overlap
ax.legend(frameon=False, loc='upper center',
          bbox_to_anchor=(0.5, 1.20), ncol=3)

fig.tight_layout()
fig.savefig("figure_ens.pdf")
plt.show()
