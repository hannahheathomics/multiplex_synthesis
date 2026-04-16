"""
CSA Analysis Script — CSV input version (QuPath ROI export)
-------------------------------------------------------------
Input:  QuPath CSA outpute file.csv
Output: fig1_violin.png, fig2_stackedbar.png, fig3_boxplot.png

Requirements:
    pip install pandas numpy matplotlib scipy

Google Colab quick-start:
    from google.colab import files
    uploaded = files.upload()   # upload the CSV
    # then run this script
"""

import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from scipy.stats import kruskal, chi2_contingency, rankdata
from itertools import combinations


# ══════════════════════════════════════════════════════════════════════════════
# USER SETTINGS
# ══════════════════════════════════════════════════════════════════════════════

INPUT_FILE = "ROIs_Updated_Data_Sweep_No_Filtering.csv"

# Pixel → µm² conversion factor
# Derived from calibration: Rectangle ROI = 600×600 px = 100×100 µm
# → (100/600)² = 0.02778 µm²/px²
# Update this if your calibration is different

# Map abbreviated treatment names (from filename) to full display labels
TREATMENT_MAP = {
    "Uninjured":      "Uninjured",
    "Uninj+Ibup":     "Uninjured + Ibuprofen",
    "Uninj+Ibup+Car": "Uninjured + Ibu + Carnitine",
    "SBI":            "SBI",
    "SBI+Ibuprofen":  "SBI + Ibuprofen",
    "SBI+Ibu+Car":    "SBI + Ibu + Carnitine",
}

# Display order: injured groups first, then uninjured
TREATMENT_ORDER = [
    "Uninjured",
    "Uninjured + Ibuprofen",
    "Uninjured + Ibu + Carnitine",
    "SBI",
    "SBI + Ibuprofen",
    "SBI + Ibu + Carnitine",
]

SHORT_LABELS = [
    "Uninj",
    "Uninj +\nIbup",
    "Uninj +\nIbu+Car",
    "SBI",
    "SBI +\nIbup",
    "SBI +\nIbu+Car",
]

# SET BIN EDGES AFTER INSPECTING YOUR HISTOGRAM DATA
# Updated bin edges — must match BIN_LABELS exactly
BIN_EDGES  = [0, 250, 500, 750, 1000, 1250, 1500, 2000, 3000, np.inf]
BIN_LABELS = ["0–250", "250–500", "500–750", "750–1k",
              "1–1.25k", "1.25–1.5k", "1.5–2k", "2–3k", "3k+"]

MAX_FIBERS_SHOWN = 300   # individual dots shown per group in Fig 3
RANDOM_SEED      = 42


# ══════════════════════════════════════════════════════════════════════════════
# COLORS
# ══════════════════════════════════════════════════════════════════════════════

DOT_COLORS = {
    "Uninjured":                    "#E07B4F",
    "Uninjured + Ibuprofen":        "#E8A07A",
    "Uninjured + Ibu + Carnitine":  "#C4603A",
    "SBI":                          "#5B7FBD",
    "SBI + Ibuprofen":              "#7B9FCC",
    "SBI + Ibu + Carnitine":        "#3A5FA0",
}
BG_SBI   = "#D6E4F7"
BG_UNINJ = "#FDF3E3"
BIN_COLORS = [
    "#EFF3FF", "#C6DBEF", "#9ECAE1", "#6BAED6",
    "#4292C6", "#2171B5", "#08519C", "#08306B", "#03152F",
]


# ══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def parse_image_filename(img):
    """
    Parse mouse_ID, treatment abbreviation, and image_number from filename.

    Normal format:  {mouse}_{treatment}_{imgnum}_10X_ROI_...
    Special case:   32N_01_SBI_10X_ROI_... (number before treatment)
    """
    stem  = re.sub(r'_10X_ROI.*$', '', img)
    parts = stem.split("_")
    mouse_id = parts[0]

    # Special case: number immediately after mouse ID (e.g. 32N_01_SBI)
    if len(parts) >= 3 and re.match(r'^\d{2}$', parts[1]):
        return mouse_id, "_".join(parts[2:]), parts[1]

    # Normal case: find the two-digit image number
    image_number = None
    treat_parts  = []
    for p in parts[1:]:
        if re.match(r'^\d{2}$', p) and image_number is None:
            image_number = p
        else:
            treat_parts.append(p)

    return mouse_id, "_".join(treat_parts), image_number


def stars(p):
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "ns"


def dunns_test(df, val_col, group_col, p_adjust="bonferroni"):
    """Dunn's post-hoc using the global rank pool. Returns {(g1,g2): p_adj}."""
    from scipy.stats import norm
    all_vals   = df[val_col].values
    all_groups = df[group_col].values
    n_total    = len(all_vals)
    ranks      = rankdata(all_vals)

    group_info = {}
    for g in np.unique(all_groups):
        mask = all_groups == g
        group_info[g] = {"n": mask.sum(), "mean_rank": ranks[mask].mean()}

    pairs = list(combinations(np.unique(all_groups), 2))
    _, counts = np.unique(ranks, return_counts=True)
    tie_corr  = np.sum(counts ** 3 - counts) / (12 * (n_total - 1))

    results = {}
    for g1, g2 in pairs:
        n1, mr1 = group_info[g1]["n"], group_info[g1]["mean_rank"]
        n2, mr2 = group_info[g2]["n"], group_info[g2]["mean_rank"]
        se = np.sqrt(
            (n_total * (n_total + 1) / 12 - tie_corr) * (1/n1 + 1/n2)
        )
        z = abs(mr1 - mr2) / se
        p = 2 * (1 - norm.cdf(z))
        results[(g1, g2)] = p
        results[(g2, g1)] = p

    n_pairs = len(pairs)
    if p_adjust == "bonferroni":
        results = {k: min(v * n_pairs, 1.0) for k, v in results.items()}
    return results


def add_background(ax, y_max):
  ax.axvspan(-0.5, 2.5, color=BG_UNINJ, zorder=0)
  ax.axvspan( 2.5, 5.5, color=BG_SBI,   zorder=0)
  ax.text(1.0, ylim_top * 0.975, "SBI (injured)", ha="center", va="top",
          fontsize=10, color="#3A5FA0", fontweight="bold")
  ax.text(4.0, ylim_top * 0.975, "Uninjured",     ha="center", va="top",
          fontsize=10, color="#C4603A", fontweight="bold")


def subsample(arr, n, rng):
    return rng.choice(arr, n, replace=False) if len(arr) > n else arr


# ══════════════════════════════════════════════════════════════════════════════
# 1. LOAD & PARSE
# ══════════════════════════════════════════════════════════════════════════════

print("Loading data...")
raw = pd.read_csv(INPUT_FILE)
raw["Image"] = raw["Image"].astype(str)

# Keep only fiber detections (Polygon = fiber; Rectangle = ROI boundary)
fibers = raw[raw["ROI"] == "Polygon"].copy()
print(f"Retained {len(fibers):,} fiber polygons "
      f"(dropped {(raw['ROI']=='Rectangle').sum()} Rectangle ROI rows).\n")

# Parse filename → mouse_ID, treatment abbreviation, image_number
parsed = fibers["Image"].apply(
    lambda x: pd.Series(parse_image_filename(x),
                         index=["mouse_ID", "treatment_abbrev", "image_number"])
)
fibers = pd.concat([fibers, parsed], axis=1)

# Map abbreviated treatment to full display label
fibers["group"] = fibers["treatment_abbrev"].map(TREATMENT_MAP)

# Flag any rows that didn't map (unknown treatment abbreviation)
unmapped = fibers[fibers["group"].isna()]
if not unmapped.empty:
    print("=== WARNING: Unmapped treatment abbreviations ===")
    print(unmapped[["Image","treatment_abbrev"]].drop_duplicates().to_string(index=False))
    print("Add these to TREATMENT_MAP and re-run.\n")
    fibers = fibers.dropna(subset=["group"])

# Convert area from px² to µm²
area_col = [c for c in fibers.columns if "Area" in c][0]
fibers["CSA"] = fibers[area_col]


# ══════════════════════════════════════════════════════════════════════════════
# 2. DATA EXPLORATION — inspect before finalising bin edges
# ══════════════════════════════════════════════════════════════════════════════

print("=== CSA distribution summary (µm²) ===")
print(fibers["CSA"].describe().round(1))
print()

# Uncomment to save an exploration histogram:
fig_exp, ax_exp = plt.subplots(figsize=(8, 4))
ax_exp.hist(fibers["CSA"], bins=80, color="#5B7FBD",
             edgecolor="white", linewidth=0.3)
ax_exp.set_xlabel("Fiber CSA (µm²)")
ax_exp.set_ylabel("Count")
ax_exp.set_title("All fibers — CSA histogram (use to choose bin edges)")
fig_exp.tight_layout()
fig_exp.savefig("csa_histogram_exploration.png", dpi=150)
plt.close(fig_exp)
print("Exploration histogram saved.\n")

# ══════════════════════════════════════════════════════════════════════════════
# 3. PER-MOUSE SUMMARIES & SANITY CHECKS
# ══════════════════════════════════════════════════════════════════════════════

per_mouse = (fibers.groupby(["mouse_ID", "group"])["CSA"]
                   .agg(median="median", mean="mean", n="count")
                   .reset_index())

print("=== Fiber counts per mouse ===")
print(per_mouse[["mouse_ID", "group", "n"]].to_string(index=False))
print()

print("=== Fiber counts per group ===")
print(fibers.groupby("group")["CSA"].count().reindex(TREATMENT_ORDER))
print()

# Flag mice with extreme fiber counts vs group median
grp_med = per_mouse.groupby("group")["n"].median()
per_mouse["grp_median_n"] = per_mouse["group"].map(grp_med)
per_mouse["ratio"]        = per_mouse["n"] / per_mouse["grp_median_n"]
flagged = per_mouse[(per_mouse["ratio"] > 3) | (per_mouse["ratio"] < 0.33)]
if not flagged.empty:
    print("=== Flagged mice (extreme fiber count imbalance) ===")
    print(flagged[["mouse_ID","group","n","grp_median_n","ratio"]].to_string(index=False))
    print()
else:
    print("Fiber count imbalance check: no mice flagged.\n")

# ══════════════════════════════════════════════════════════════════════════════
# 4. DESCRIPTIVE STATISTICS
# ══════════════════════════════════════════════════════════════════════════════

print("=== Descriptive statistics per group (µm²) ===")
desc = fibers.groupby("group")["CSA"].agg(
    n="count",
    median="median",
    q25=lambda x: x.quantile(0.25),
    q75=lambda x: x.quantile(0.75),
    mean="mean",
    sd="std",
).reindex(TREATMENT_ORDER)
print(desc.round(1).to_string())
print()

# ══════════════════════════════════════════════════════════════════════════════
# 5. KRUSKAL-WALLIS + DUNN'S
# ══════════════════════════════════════════════════════════════════════════════

group_arrays = [fibers.loc[fibers["group"] == t, "CSA"].values
                for t in TREATMENT_ORDER]

stat, p_kw = kruskal(*group_arrays)
print(f"=== Kruskal-Wallis: H = {stat:.3f},  p = {p_kw:.4e} ===")
print("Significant — proceeding to Dunn's.\n" if p_kw < 0.05 else "Not significant.\n")

dunn = dunns_test(fibers, "CSA", "group", p_adjust="bonferroni")

print("=== Dunn's post-hoc vs Uninjured (Bonferroni) ===")
for t in TREATMENT_ORDER:
    if t == "Uninjured": continue
    p = dunn.get(("Uninjured", t), dunn.get((t, "Uninjured"), np.nan))
    print(f"  Uninjured vs {t:<30}: p = {p:.4f}  {stars(p)}")
print()

print("=== Dunn's post-hoc: SBI-treated vs SBI alone ===")
for t in ["SBI + Ibuprofen", "SBI + Ibu + Carnitine"]:
    p = dunn.get(("SBI", t), dunn.get((t, "SBI"), np.nan))
    print(f"  SBI vs {t:<30}: p = {p:.4f}  {stars(p)}")
print()

# ══════════════════════════════════════════════════════════════════════════════
# 6. CHI-SQUARE ON BINNED DISTRIBUTIONS
# ══════════════════════════════════════════════════════════════════════════════

fibers["csa_bin"] = pd.cut(fibers["CSA"], bins=BIN_EDGES,
                            labels=BIN_LABELS, right=True)
bin_counts = (fibers.groupby(["group", "csa_bin"], observed=True)
                    .size().unstack(fill_value=0).reindex(TREATMENT_ORDER))
bin_props  = bin_counts.div(bin_counts.sum(axis=1), axis=0) * 100

contingency = bin_counts.values
min_exp = chi2_contingency(contingency)[3].min()
print(f"=== Chi-square bin check: min expected count = {min_exp:.1f} ===")
print("WARNING: consider merging bins.\n" if min_exp < 5 else "Expected counts OK.\n")

chi2_stat, p_chi2, dof, _ = chi2_contingency(contingency)
print(f"Overall chi-square: χ²({dof}) = {chi2_stat:.2f},  p = {p_chi2:.4e}\n")

print("=== Pairwise chi-square vs Uninjured (Bonferroni) ===")
n_comp = len(TREATMENT_ORDER) - 1
for t in TREATMENT_ORDER:
    if t == "Uninjured": continue
    pair = bin_counts.loc[["Uninjured", t]].copy()

    # Drop bins where either group has zero fibers
    pair = pair.loc[:, (pair > 0).all(axis=0)]

    if pair.shape[1] < 2:
        print(f"  Uninjured vs {t:<30}: insufficient non-zero bins — skip")
        continue

    chi2, p_raw, dof_p, _ = chi2_contingency(pair)
    p_adj = min(p_raw * n_comp, 1.0)
    print(f"  Uninjured vs {t:<30}: χ²({dof_p}) = {chi2:7.2f},  "
          f"p_adj = {p_adj:.4f}  {stars(p_adj)}")

# ── NEW CONSTANTS ─────────────────────────────────────────────────────────────
BG_SBI   = "#E8E0F5"   # light purple for SBI groups
BG_UNINJ = "#FDF8E1"   # light yellow for Uninjured groups

XAXIS_IBUPROFEN = ["–", "+", "+", "–", "+", "+"]
XAXIS_CARNITINE = ["–", "–", "+", "–", "–", "+"]

DOT_COLORS = {
    "SBI":                          "#888780",
    "SBI + Ibuprofen":              "#6B4FA0",
    "SBI + Ibu + Carnitine":        "#D85A30",
    "Uninjured":                    "#888780",
    "Uninjured + Ibuprofen":        "#6B4FA0",
    "Uninjured + Ibu + Carnitine":  "#D85A30",
}

BIN_COLORS = [
    "#F7FCF5", "#C7E9C0", "#A1D99B", "#74C476",
    "#41AB5D", "#238B45", "#006D2C", "#00441B",
]

# ── NEW HELPER FUNCTIONS ──────────────────────────────────────────────────────
def add_background(ax, y_max):
    ax.axvspan(-0.5, 2.5, color=BG_UNINJ, zorder=0)
    ax.axvspan( 2.5, 5.5, color=BG_SBI,   zorder=0)


def set_xaxis_labels(ax, y_max):
    ax.set_xticks(np.arange(len(TREATMENT_ORDER)))
    ax.set_xticklabels([])
    row_labels = ["Ibuprofen", "Carnitine"]
    row_values = [XAXIS_IBUPROFEN, XAXIS_CARNITINE]

    for row_idx, (row_label, row_vals) in enumerate(zip(row_labels, row_values)):
        y_ax = -0.09 - row_idx * 0.075
        ax.text(-0.5, y_ax, row_label,
                transform=ax.get_xaxis_transform(),
                ha="right", va="center", fontsize=9, style="italic", color="#333333")
        for col_idx, val in enumerate(row_vals):
            ax.text(col_idx, y_ax, val,
                    transform=ax.get_xaxis_transform(),
                    ha="center", va="center", fontsize=10, color="black",
                    fontweight="bold" if val == "+" else "normal")

    ax.axhline(y=0, color="black", linewidth=0.5, clip_on=False)

    # Group labels directly under x axis line
    ax.text(1.0, -0.27, "Uninjured",
            transform=ax.get_xaxis_transform(),
            ha="center", va="top", fontsize=10,
            color="#6B4FA0", fontweight="bold")
    ax.text(4.0, -0.27, "SBI (injured)",
            transform=ax.get_xaxis_transform(),
            ha="center", va="top", fontsize=10,
            color="#6B4FA0", fontweight="bold")


def draw_bracket(ax, x1, x2, y, label, y_max, color="black", lw=1.2, fontsize=9):
    tick_h = y_max * 0.018
    ax.plot([x1, x1, x2, x2], [y, y+tick_h, y+tick_h, y],
            color=color, lw=lw, clip_on=False, zorder=6)
    ax.text((x1+x2)/2, y + tick_h*1.8, label,
            ha="center", va="bottom", fontsize=fontsize, color=color, zorder=6)


def add_brackets(ax, dunn, y_max):
    idx = {t: i for i, t in enumerate(TREATMENT_ORDER)}

    # Build list of all significant pairwise comparisons
    comparisons = []
    seen = set()
    for (g1, g2), p in dunn.items():
        pair = tuple(sorted([g1, g2]))
        if pair in seen:
            continue
        seen.add(pair)
        if p < 0.05:
            comparisons.append((g1, g2, p))

    # Sort by span width (shortest brackets drawn lowest)
    comparisons_sorted = sorted(comparisons,
                                key=lambda c: abs(idx[c[1]] - idx[c[0]]))

    y_start = y_max * 0.76
    y_step  = y_max * 0.065
    for i, (g1, g2, p) in enumerate(comparisons_sorted):
        label = stars(p)
        y     = y_start + i * y_step
        draw_bracket(ax, idx[g1], idx[g2], y, label, y_max)

# ══════════════════════════════════════════════════════════════════════════════
# PLOTTING SETUP
# ══════════════════════════════════════════════════════════════════════════════

positions = np.arange(len(TREATMENT_ORDER))
rng       = np.random.default_rng(RANDOM_SEED)
Y_MAX     = fibers["CSA"].quantile(0.999) * 1.5
colors    = [DOT_COLORS[t] for t in TREATMENT_ORDER]


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — VIOLIN PLOT
# ══════════════════════════════════════════════════════════════════════════════

fig1, ax1 = plt.subplots(figsize=(10, 7))
fig1.subplots_adjust(bottom=0.18)   # room for x-axis label table

add_background(ax1, Y_MAX)

parts = ax1.violinplot(group_arrays, positions=positions,
                       showmedians=False, showextrema=False, widths=0.62)
for pc, t in zip(parts["bodies"], TREATMENT_ORDER):
    pc.set_facecolor(DOT_COLORS[t])
    pc.set_alpha(0.65)
    pc.set_edgecolor("none")
    pc.set_zorder(2)

# Group median line
for i, arr in enumerate(group_arrays):
    ax1.hlines(np.median(arr), positions[i]-0.26, positions[i]+0.26,
               color="black", linewidth=2.0, zorder=4)

# Per-mouse median dots
for i, t in enumerate(TREATMENT_ORDER):
    meds = per_mouse.loc[per_mouse["group"]==t, "median"].values
    jit  = rng.uniform(-0.08, 0.08, size=len(meds))
    ax1.scatter(positions[i]+jit, meds, s=72, color="white",
                edgecolors="black", linewidths=1.4, zorder=5)

add_brackets(ax1, dunn, Y_MAX)
set_xaxis_labels(ax1, Y_MAX)

ax1.set_ylabel("Fiber CSA (µm²)", fontsize=11)
ax1.set_xlim(-0.5, 5.5)
ax1.set_ylim(0, Y_MAX)
ax1.spines[["top","right","bottom"]].set_visible(False)
ax1.tick_params(axis="x", length=0)

leg1 = [
    mpatches.Patch(facecolor="#888780", alpha=0.75, label="No drug"),
    mpatches.Patch(facecolor="#6B4FA0", alpha=0.75, label="+ Ibuprofen"),
    mpatches.Patch(facecolor="#D85A30", alpha=0.75, label="+ Ibu + Carnitine"),
    Line2D([0],[0], color="black", linewidth=2,   label="Group median"),
    Line2D([0],[0], marker="o", color="w", markerfacecolor="white",
           markeredgecolor="black", markersize=8, label="Per-mouse median"),
]
ax1.legend(handles=leg1, frameon=False, fontsize=8, loc="upper right")
ax1.set_title("Fiber CSA — violin plot", fontsize=12, pad=8)

fig1.savefig("fig1_violin.png", dpi=300, bbox_inches="tight")
plt.close(fig1)
print("fig1_violin.png saved.")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — STACKED BAR
# ══════════════════════════════════════════════════════════════════════════════

fig2, ax2 = plt.subplots(figsize=(10, 6))
fig2.subplots_adjust(bottom=0.18)

# Split background
ax2.axvspan(-0.5, 2.5, color=BG_UNINJ, zorder=0)
ax2.axvspan( 2.5, 5.5, color=BG_SBI,   zorder=0)

bottom = np.zeros(len(TREATMENT_ORDER))
for bl, col in zip(BIN_LABELS, BIN_COLORS):
    vals = bin_props[bl].values
    ax2.bar(positions, vals, bottom=bottom, color=col, label=bl,
            width=0.6, edgecolor="white", linewidth=0.5, zorder=2)
    bottom += vals

set_xaxis_labels(ax2, 100)

ax2.set_ylabel("% of fibers", fontsize=11)
ax2.set_ylim(0, 100)
ax2.set_xlim(-0.5, 5.5)
ax2.spines[["top","right","bottom"]].set_visible(False)
ax2.tick_params(axis="x", length=0)
ax2.legend(title="CSA bin (µm²)", bbox_to_anchor=(1.01,1),
           loc="upper left", fontsize=8, frameon=False)
ax2.set_title("Fiber CSA bin distribution", fontsize=12, pad=8)

fig2.savefig("fig2_stackedbar.png", dpi=300, bbox_inches="tight")
plt.close(fig2)
print("fig2_stackedbar.png saved.")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 3 — BOX PLOT
# ══════════════════════════════════════════════════════════════════════════════

fig3, ax3 = plt.subplots(figsize=(10, 7))
fig3.subplots_adjust(bottom=0.18)

add_background(ax3, Y_MAX)

bp = ax3.boxplot(
    group_arrays, positions=positions, widths=0.45,
    patch_artist=True, showfliers=False,
    medianprops=dict(color="black",  linewidth=2.0),
    whiskerprops=dict(color="#444",  linewidth=1.0),
    capprops=dict(color="#444",      linewidth=1.0),
    boxprops=dict(linewidth=0.8),
    zorder=3,
)
for patch, t in zip(bp["boxes"], TREATMENT_ORDER):
    patch.set_facecolor(DOT_COLORS[t])
    patch.set_alpha(0.55)

# Individual fibers (subsampled)
for i, t in enumerate(TREATMENT_ORDER):
    arr = fibers.loc[fibers["group"]==t, "CSA"].values
    sub = subsample(arr, MAX_FIBERS_SHOWN, rng)
    jit = rng.uniform(-0.19, 0.19, size=len(sub))
    ax3.scatter(positions[i]+jit, sub,
                color=DOT_COLORS[t], alpha=0.30, s=5, zorder=2, linewidths=0)

# Per-mouse mean diamonds
for i, t in enumerate(TREATMENT_ORDER):
    means = per_mouse.loc[per_mouse["group"]==t, "mean"].values
    jit   = rng.uniform(-0.07, 0.07, size=len(means))
    ax3.scatter(positions[i]+jit, means,
                marker="D", s=60, color="white",
                edgecolors="black", linewidths=1.4, zorder=5)

add_brackets(ax3, dunn, Y_MAX)
set_xaxis_labels(ax3, Y_MAX)

ax3.set_ylabel("Fiber CSA (µm²)", fontsize=11)
ax3.set_xlim(-0.5, 5.5)
ax3.set_ylim(0, Y_MAX)
ax3.spines[["top","right","bottom"]].set_visible(False)
ax3.tick_params(axis="x", length=0)

leg3 = [
    mpatches.Patch(facecolor="#888780", alpha=0.75, label="No drug"),
    mpatches.Patch(facecolor="#6B4FA0", alpha=0.75, label="+ Ibuprofen"),
    mpatches.Patch(facecolor="#D85A30", alpha=0.75, label="+ Ibu + Carnitine"),
    Line2D([0],[0], marker="o", color="w", markerfacecolor="#888",
           alpha=0.5, markersize=6,
           label=f"Individual fiber ({MAX_FIBERS_SHOWN}/group)"),
    Line2D([0],[0], color="black", linewidth=2, label="Group median"),
    Line2D([0],[0], marker="D", color="w", markerfacecolor="white",
           markeredgecolor="black", markersize=8, label="Per-mouse mean"),
]
ax3.legend(handles=leg3, frameon=False, fontsize=8, loc="upper right")
ax3.set_title("Fiber CSA — box plot with individual fibers", fontsize=12, pad=8)

fig3.savefig("fig3_boxplot.png", dpi=300, bbox_inches="tight")
plt.close(fig3)
print("fig3_boxplot.png saved.")

print("\nAll done.")
