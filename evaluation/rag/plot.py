import matplotlib as mpl
import matplotlib.pyplot as plt
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

# Data

results_bm25 = json.load(open("abl_rag_bm25.json"))
results_vs = json.load(open("abl_rag_vs.json"))

# --- Figure 1: BM25 ---
fig1, ax1 = plt.subplots(figsize=(4, 3))  # single-column size

x = [r["BM25_k"] for r in results_bm25]
y1 = [r["non_llm_context_precision_with_reference"] for r in results_bm25]
ax1.plot(x, y1, linewidth=1.5, linestyle="--", label=r"$BM25 - Precision$", color="red")

y3 = [r["non_llm_context_recall"] for r in results_bm25]
ax1.plot(x, y3, linewidth=1.5, linestyle="--", label=r"$BM25 - Recall$", color="green")

y_f1_bm25 = [2*(p*r)/(p+r) if (p+r)>0 else 0 for p,r in zip(y1, y3)]
ax1.plot(x, y_f1_bm25, linewidth=1.5, linestyle="-", label=r"$BM25 - F1$", color="blue")

ax1.set_xlabel(r"$k_{bm25}$")
ax1.set_xticks(x)
ax1.set_yticks(np.arange(0, 1, 0.1))
ax1.grid(True, alpha=0.3)
ax1.legend(frameon=False)

fig1.tight_layout()
fig1.savefig("figure_bm25.pdf")   # save separately


# --- Figure 2: Vector DB ---
fig2, ax2 = plt.subplots(figsize=(4, 3))  # single-column size

y2 = [r["non_llm_context_precision_with_reference"] for r in results_vs]
ax2.plot(x, y2, linewidth=1.5, linestyle="--", label=r"$VDB - Precision$", color="red")

y4 = [r["non_llm_context_recall"] for r in results_vs]
ax2.plot(x, y4, linewidth=1.5, linestyle="--", label=r"$VDB - Recall$", color="green")

y_f1_vs = [2*(p*r)/(p+r) if (p+r)>0 else 0 for p,r in zip(y2, y4)]
ax2.plot(x, y_f1_vs, linewidth=1.5, linestyle="-", label=r"$VDB - F1$", color="blue")

ax2.set_xlabel(r"$k_{vdb}$")
ax2.set_xticks(x)
ax2.set_yticks(np.arange(0, 1, 0.1))
ax2.grid(True, alpha=0.3)
ax2.legend(frameon=False)

fig2.tight_layout()
fig2.savefig("figure_vdb.pdf")   # save separately
