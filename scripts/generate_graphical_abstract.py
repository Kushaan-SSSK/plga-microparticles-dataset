"""
generate_graphical_abstract.py
================================
Graphical abstract using GridSpec — no manual add_axes, no label bleed.

Layout:
  Row 0 (narrow): navy title banner
  Row 1 (tall):   3 data panels with generous wspace
  Row 2 (medium): 3 coloured insight boxes

Output: figures/graphical_abstract.png
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch, Rectangle
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT  = os.path.join(BASE, "figures", "graphical_abstract.png")
os.makedirs(os.path.join(BASE, "figures"), exist_ok=True)

# ── Data ──────────────────────────────────────────────────────────────────────
bench = pd.read_csv(os.path.join(BASE, "benchmark_results.csv"))
perm  = pd.read_csv(os.path.join(BASE, "permutation_importance_n.csv"))

# ── Palette ───────────────────────────────────────────────────────────────────
NAVY   = "#152744"
BG     = "#F8F9FB"
PANEL  = "#EEF2F7"
BLUE   = "#2C7BB6"
TEAL   = "#1A9480"
RED    = "#EF5350"
LGRAY  = "#B0BEC5"
DGRAY  = "#4A4A4A"
AMBER  = "#E67E22"
GREEN  = "#1A9641"
WHITE  = "#FFFFFF"

MODEL_COLOR = {
    "Linear":          LGRAY,
    "RandomForest":    TEAL,
    "XGBoost":         RED,
    "StackedEnsemble": NAVY,
}
MODEL_LABEL = {
    "Linear":          "Linear",
    "RandomForest":    "Random Forest",
    "XGBoost":         "XGBoost",
    "StackedEnsemble": "Stacked Ensemble",
}
FEAT_LABEL = {
    "Drug Encapsulation Efficiency": "Encapsulation Eff.",
    "Drug Loading Capacity":         "Loading Capacity",
    "NumHDonors":                    "H-Bond Donors",
    "Polymer MW":                    "Polymer MW",
    "Hydrophilicity_Index":          "Hydrophilicity Idx",
    "RotatableBonds":                "Rotatable Bonds",
}

# ── Figure with GridSpec ───────────────────────────────────────────────────────
# 18 × 7.44 in @ 300 dpi → after tight-trim lands ~2.50:1 matching journal spec (1328×531 px)
fig = plt.figure(figsize=(18, 7.44), facecolor=BG)

gs = gridspec.GridSpec(
    3, 3,
    figure=fig,
    height_ratios=[0.11, 0.55, 0.34],
    hspace=0.38,
    wspace=0.55,
    left=0.06,
    right=0.97,
    top=0.97,
    bottom=0.03,
)

# ── Row 0: Title banner ────────────────────────────────────────────────────────
banner_ax = fig.add_subplot(gs[0, :])   # spans all 3 columns
banner_ax.set_facecolor(NAVY)
banner_ax.set_xlim(0, 1)
banner_ax.set_ylim(0, 1)
for sp in banner_ax.spines.values():
    sp.set_visible(False)
banner_ax.tick_params(left=False, bottom=False,
                      labelleft=False, labelbottom=False)
banner_ax.text(0.5, 0.68,
    "Diagnosing Predictability Limits in Literature-Mined PLGA Microparticle Drug Release Data",
    transform=banner_ax.transAxes, ha="center", va="center",
    fontsize=15, fontweight="bold", color=WHITE)
banner_ax.text(0.5, 0.22,
    "321 formulations  ·  113 studies  ·  89 drugs  ·  "
    "15 engineered features  ·  Formulation-grouped 10-fold CV",
    transform=banner_ax.transAxes, ha="center", va="center",
    fontsize=10.5, color="#8BAFD4")

# ── Row 1 col 0: Benchmark — Peppas n ─────────────────────────────────────────
ax1 = fig.add_subplot(gs[1, 0])
nd = bench[bench["Target"] == "Peppas_n"].sort_values("R2").reset_index(drop=True)
colors1 = [MODEL_COLOR[m] for m in nd["Model"]]
labels1 = [MODEL_LABEL[m] for m in nd["Model"]]
bars1 = ax1.barh(labels1, nd["R2"], color=colors1,
                 edgecolor=WHITE, linewidth=0.6, height=0.52)
for bar, v in zip(bars1, nd["R2"]):
    ax1.text(v + 0.01, bar.get_y() + bar.get_height() / 2,
             f"{v:.3f}", va="center", ha="left",
             fontsize=9.5, color=DGRAY, fontweight="bold",
             clip_on=True)
ax1.set_xlim(0, 0.60)
ax1.set_facecolor(PANEL)
ax1.spines[["top", "right", "left"]].set_visible(False)
ax1.tick_params(left=False, labelsize=10)
ax1.set_xlabel("$R^2$", fontsize=10.5, labelpad=4)
ax1.set_title("Benchmark: Peppas $n$\n(Formulation-Grouped 10-Fold CV)",
              fontsize=10.5, fontweight="bold", color=NAVY, pad=10)


# ── Row 1 col 1: Benchmark — Peppas K ─────────────────────────────────────────
ax2 = fig.add_subplot(gs[1, 1])
kd = bench[bench["Target"] == "Peppas_K"].sort_values("R2").reset_index(drop=True)
colors2 = [MODEL_COLOR[m] for m in kd["Model"]]
labels2 = [MODEL_LABEL[m] for m in kd["Model"]]
bars2 = ax2.barh(labels2, kd["R2"], color=colors2,
                 edgecolor=WHITE, linewidth=0.6, height=0.52)
for bar, v in zip(bars2, kd["R2"]):
    ax2.text(v + 0.006, bar.get_y() + bar.get_height() / 2,
             f"{v:.3f}", va="center", ha="left",
             fontsize=9.5, color=DGRAY, fontweight="bold",
             clip_on=True)
ax2.set_xlim(0, 0.40)
ax2.set_facecolor(PANEL)
ax2.spines[["top", "right", "left"]].set_visible(False)
ax2.tick_params(left=False, labelsize=10)
ax2.set_xlabel("$R^2$", fontsize=10.5, labelpad=4)
ax2.set_title("Benchmark: Peppas $K$\n(Formulation-Grouped 10-Fold CV)",
              fontsize=10.5, fontweight="bold", color=NAVY, pad=10)


# ── Row 1 col 2: Permutation importance ───────────────────────────────────────
ax3 = fig.add_subplot(gs[1, 2])
top6 = perm.nlargest(6, "Permutation_Importance_Mean").iloc[::-1].copy()
top6["Label"] = top6["Feature"].map(
    lambda f: FEAT_LABEL.get(f, f.replace("_", " ")))
bcols = [BLUE if i == len(top6) - 1 else TEAL for i in range(len(top6))]
bars3 = ax3.barh(top6["Label"].values,
                 top6["Permutation_Importance_Mean"].values,
                 color=bcols, edgecolor=WHITE, linewidth=0.6,
                 height=0.52)
# Overlay error bars separately so caps sit correctly at bar ends
y_pos = [bar.get_y() + bar.get_height() / 2 for bar in bars3]
ax3.errorbar(top6["Permutation_Importance_Mean"].values, y_pos,
             xerr=top6["Permutation_Importance_Std"].values,
             fmt="none", ecolor="#777777", elinewidth=1.4,
             capsize=4, capthick=1.4)
for bar, v, err in zip(bars3, top6["Permutation_Importance_Mean"].values, top6["Permutation_Importance_Std"].values):
    ax3.text(v + err + 0.012, bar.get_y() + bar.get_height() / 2,
             f"{v:.3f}", va="center", ha="left",
             fontsize=9.5, color=DGRAY, fontweight="bold",
             clip_on=True)
ax3.set_xlim(0, 0.70)
ax3.set_facecolor(PANEL)
ax3.spines[["top", "right", "left"]].set_visible(False)
ax3.tick_params(left=False, labelsize=10)
ax3.set_xlabel("Permutation Importance (Peppas $n$)", fontsize=10, labelpad=4)
ax3.set_title("Feature Importance\n(Permutation, Peppas $n$)",
              fontsize=10.5, fontweight="bold", color=NAVY, pad=10)

# ── Row 2: Three insight boxes ─────────────────────────────────────────────────
INSIGHTS = [
    (0, "#EBF2FB", NAVY,
     "321 formulations · 113 studies · 89 drugs\n"
     "15 features from composition & RDKit\n"
     "physicochemical descriptors"),
    (1, "#FEF3E2", AMBER,
     "All model families converge at $R^2$ ≤ 0.42\n"
     "LOSO $R^2$ < 0 → cross-study prediction fails\n"
     "Ceiling is data-driven, not model-driven"),
    (2, "#EBF7EC", GREEN,
     "ICC(1) = 0.68 → study identity dominates variance\n"
     "78% of process variables unreported\n"
     "MIADR checklist proposed for future studies"),
]

for col, bgcol, edgecol, txt in INSIGHTS:
    iax = fig.add_subplot(gs[2, col])
    iax.set_facecolor(bgcol)
    iax.set_xlim(0, 1)
    iax.set_ylim(0, 1)
    for sp in iax.spines.values():
        sp.set_linewidth(2.0)
        sp.set_edgecolor(edgecol)
        sp.set_visible(True)
    iax.tick_params(left=False, bottom=False,
                    labelleft=False, labelbottom=False)
    iax.text(0.5, 0.50, txt,
             transform=iax.transAxes,
             ha="center", va="center",
             fontsize=10, color=DGRAY,
             multialignment="center", linespacing=1.6)

# Arrow annotations between insight boxes (drawn in figure coords)
for xf in [0.355, 0.645]:
    fig.text(xf, 0.145, "▶", ha="center", va="center",
             fontsize=24, color="#888888")

plt.savefig(OUT, dpi=300, bbox_inches="tight", facecolor=BG)
plt.close()
print(f"Saved: {OUT}")
