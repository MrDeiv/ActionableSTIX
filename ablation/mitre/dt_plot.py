import matplotlib.pyplot as plt
import matplotlib as mpl
import json
import numpy as np

# --- your style settings ---
mpl.rcParams.update({
    "text.usetex": False,
    "mathtext.fontset": "stix",
    "font.family": "STIXGeneral",
    "font.size": 12,
    "axes.labelsize": 8,
    "axes.titlesize": 12,
    "legend.fontsize": 8,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "figure.figsize": (6, 6),  # square pie by default
})

# --- Verdict distribution ---

data = json.load(open("ablation/mitre/results/dt_stats.json"))

labels = ["Malicious", "Benign", "Suspicious", "Unknown"]
sizes = [data["verdict_malicious"], data["verdict_benign"], data["verdict_suspicious"], data["verdict_unknown"]]

# --- create pie chart ---
fig, ax = plt.subplots()
wedges, texts = ax.pie(
    sizes,
    #labels=labels,
    #autopct="%1.1f%%",
    startangle=90,
    counterclock=False,
    colors=["#b43f3f", "#227dd8", "#ff8000", "#6f6f6f"],
)

# --- Draw connectors + labels manually ---
# compute the angle of each wedge's center
for i, wedge in enumerate(wedges):
    angle = (wedge.theta2 + wedge.theta1) / 2
    x = np.cos(np.deg2rad(angle))
    y = np.sin(np.deg2rad(angle))

    # line endpoint (just outside the wedge)
    line_x = 1.1 * x
    line_y = 1.1 * y
    # label position (further outside)
    label_x = 1.3 * x
    label_y = 1.3 * y

    # draw the connector
    ax.plot([x*0.8, line_x, label_x], [y*0.8, line_y, label_y], color='gray', lw=1)

    # place the label (aligned left/right)
    alignment = 'left' if x > 0 else 'right'
    percentages = sizes[i] / sum(sizes) * 100
    # include percentage in label
    ax.text(label_x, label_y, f"{labels[i]} ({percentages:.1f}%)",
            ha=alignment, va='center', fontsize=10)

# style labels & percentages
plt.setp(texts, size=10, weight="normal")
#plt.setp(autotexts, size=9, weight="bold", color="white")

ax.set_title("Any.Run Verdict Distribution, 400 Samples", fontsize=12, pad=20)
ax.axis("equal")  # keep circle shape

plt.show()
plt.savefig("ablation/mitre/dt_pie_chart.png", bbox_inches="tight")  # save as raster for quick preview
plt.savefig("ablation/mitre/dt_pie_chart.pdf", bbox_inches="tight", facecolor="white")  # save as vector for publication
plt.close()

exit()

# --- OS distribution ---
labels = ["Windows", "Linux", "Unknown"]
sizes = [data["os_windows"], data["os_linux"], data["os_unknown"]]
# --- create pie chart ---
fig, ax = plt.subplots()
wedges, texts, autotexts = ax.pie(
    sizes,
    labels=labels,
    autopct="%1.1f%%",
    startangle=90,
    counterclock=False,
    colors=["#227dd8", "#ff8000", "#6f6f6f"],
)

# style labels & percentages
plt.setp(texts, size=10, weight="normal")
plt.setp(autotexts, size=9, weight="bold", color="white")
ax.set_title("Operating System Distribution, 400 Samples", fontsize=12, pad=20)
ax.axis("equal")  # keep circle shape

plt.show()
plt.savefig("ablation/mitre/dt_pie_chart_os.png", bbox_inches="tight")  # save as raster for quick preview
plt.savefig("ablation/mitre/dt_pie_chart_os.pdf", bbox_inches="tight", facecolor="white")  # save as vector for publication
plt.close()

# --- Upload month distribution ---
labels = ["June", "July", "August", "September", "Unknown"]
sizes = [data["upl_june"], data["upl_july"], data["upl_august"], data["upl_september"], data["upl_unknown"]]
# --- create pie chart ---
fig, ax = plt.subplots()
wedges, texts, autotexts = ax.pie(
    sizes,
    labels=labels,
    autopct="%1.1f%%",
    startangle=90,
    counterclock=False,
    colors=["#227dd8", "#ff8000", "#6f6f6f", "#b43f3f", "#6f6f6f"],
)

# style labels & percentages
plt.setp(texts, size=10, weight="normal")
plt.setp(autotexts, size=9, weight="bold", color="white")
ax.set_title("Upload Month Distribution, 400 Samples", fontsize=12, pad=20)
ax.axis("equal")  # keep circle shape
plt.show()
plt.savefig("ablation/mitre/dt_pie_chart_month.png", bbox_inches="tight")  # save as raster for quick preview
plt.savefig("ablation/mitre/dt_pie_chart_month.pdf", bbox_inches="tight", facecolor="white")  # save as vector for publication
plt.close()