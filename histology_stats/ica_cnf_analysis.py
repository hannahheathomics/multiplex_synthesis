"""
# CNF Rate Analysis — Centrally Nucleated Fiber Quantification

Quantifies the percentage of centrally nucleated fibers (CNF) per treatment
group in a skeletal muscle histology dataset, using CellProfiler output
filtered to QuPath-defined ROIs.

## Inputs

| File | Description |
|------|-------------|
| `ShrunkMuscle.csv` | CellProfiler per-fiber output. Requires `Metadata_FileLocation` and `Classify_CNF` columns. |
| `ROIs_Updated_Data_Sweep_No_Filtering.csv` | QuPath ROI table. Requires an `Image` column with filenames encoding mouse ID and image number. |

## QuPath cross-check

The QuPath ROI table is used to confirm that CellProfiler and QuPath processed
the same set of images. Mouse ID and image number are parsed from filenames in
both files and inner-joined.

## Analysis pipeline

1. **Per-ROI CNF rate** — `n_CNF / total_fibers × 100` per `(mouse, image_number)`.
2. **Per-mouse CNF rate** — mean of per-ROI rates (equal weighting across ROIs).
3. **Kruskal-Wallis test** on individual fiber CNF calls across all groups.
4. **Dunn's post-hoc test** (Bonferroni correction) for pairwise comparisons.
5. **Bar plot** — group means ± SE, individual mouse dots (jittered), and
   significant brackets drawn above the bars.

## Output

- `cnf_barplot.png` — 300 dpi bar plot.
- Printed Kruskal-Wallis and Dunn's results in the terminal.

## Treatment groups

Six groups in a 2 × 3 design (injury × treatment):

```
Uninjured | Uninjured + Ibuprofen | Uninjured + Ibu + Carnitine
SBI       | SBI + Ibuprofen       | SBI + Ibu + Carnitine
```
"""

import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from scipy.stats import kruskal, rankdata, norm
from itertools import combinations

# ══════════════════════════════════════════════════════════════════════════════
# USER SETTINGS
# ══════════════════════════════════════════════════════════════════════════════

CNF_FILE    = "ShrunkMuscle.csv"
QUPATH_FILE = "ROIs_Updated_Data_Sweep_No_Filtering.csv"

TREATMENT_MAP = {
    "Uninjured":      "Uninjured",
    "Uninj+Ibup":     "Uninjured + Ibuprofen",
    "Uninj+Ibup+Car": "Uninjured + Ibu + Carnitine",
    "SBI":            "SBI",
    "SBI+Ibuprofen":  "SBI + Ibuprofen",
    "SBI+Ibu+Car":    "SBI + Ibu + Carnitine",
}

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

# ══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def parse_image_filename(img):
    stem  = re.sub(r'_10X_ROI.*$', '', img)
    parts = stem.split("_")
    mouse_id = parts[0]
    if len(parts) >= 3 and re.match(r'^\d{2}$', parts[1]):
        return mouse_id, "_".join(parts[2:]), parts[1]
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
    from scipy.stats import norm
    all_vals   = df[val_col].values
    all_groups = df[group_col].values
    n_total    = len(all_vals)
    ranks      = rankdata(all_vals)
    group_info = {}
    for g in np.unique(all_groups):
        mask = all_groups == g
        group_info[g] = {"n": mask.sum(), "mean_rank": ranks[mask].mean()}
    pairs    = list(combinations(np.unique(all_groups), 2))
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

def draw_bracket(ax, x1, x2, y, label, y_max, color="black"):
    h = y_max * 0.012
    ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y],
            color=color, linewidth=1.2)
    ax.text((x1+x2)/2, y+h, label, ha="center", va="bottom", fontsize=9)

def add_brackets(ax, dunn, y_max):
    idx = {t: i for i, t in enumerate(TREATMENT_ORDER)}
    comparisons = []
    seen = set()
    for (g1, g2), p in dunn.items():
        pair = tuple(sorted([g1, g2]))
        if pair in seen: continue
        seen.add(pair)
        if p < 0.05:
            comparisons.append((g1, g2, p))
    comparisons_sorted = sorted(comparisons,
                                key=lambda c: abs(idx[c[1]] - idx[c[0]]))
    y_start = y_max * 0.68
    y_step  = y_max * 0.065
    for i, (g1, g2, p) in enumerate(comparisons_sorted):
        draw_bracket(ax, idx[g1], idx[g2],
                     y_start + i * y_step, stars(p), y_max)

# ══════════════════════════════════════════════════════════════════════════════
# 1. LOAD & PARSE CNF DATA
# ══════════════════════════════════════════════════════════════════════════════

print("Loading CNF data...")
cnf_raw = pd.read_csv(CNF_FILE)

# Decode URL-encoded file path and extract filename
cnf_raw["filename"] = (cnf_raw["Metadata_FileLocation"]
                       .str.replace("%2B", "+", regex=False)
                       .str.replace("%20", " ", regex=False)
                       .str.replace(r".*[/\\]", "", regex=True))

# Parse mouse ID, treatment, ROI number
parsed = cnf_raw["filename"].apply(
    lambda x: pd.Series(parse_image_filename(x),
                        index=["mouse_ID", "treatment_abbrev", "image_number"])
)
cnf_raw = pd.concat([cnf_raw, parsed], axis=1)
cnf_raw["group"] = cnf_raw["treatment_abbrev"].map(TREATMENT_MAP)

# ══════════════════════════════════════════════════════════════════════════════
# 2. CROSS-CHECK WITH QUPATH ROIs
# ══════════════════════════════════════════════════════════════════════════════

print("Filtering to QuPath ROIs...")
qupath = pd.read_csv(QUPATH_FILE)
qupath["mouse_ID"] = qupath["Image"].str.extract(r'^([^_]+)_')
qupath["image_number"] = qupath["Image"].str.extract(r'^[^_]+_[^_]+_(\d+)_')
valid_rois = qupath[["mouse_ID", "image_number"]].drop_duplicates()

cnf_filtered = cnf_raw.merge(valid_rois, on=["mouse_ID", "image_number"])
print(f"Retained {len(cnf_filtered):,} fibers across "
      f"{cnf_filtered['mouse_ID'].nunique()} mice.")

# ══════════════════════════════════════════════════════════════════════════════
# 3. PER-ROI THEN PER-MOUSE CNF RATE
# ══════════════════════════════════════════════════════════════════════════════

per_roi = (cnf_filtered
           .groupby(["mouse_ID", "group", "image_number"])
           .agg(total_fibers=("Classify_CNF", "count"),
                n_CNF=("Classify_CNF", "sum"))
           .assign(pct_CNF=lambda d: 100 * d["n_CNF"] / d["total_fibers"])
           .reset_index())

per_mouse = (per_roi
             .groupby(["mouse_ID", "group"])
             .agg(pct_CNF=("pct_CNF", "mean"))  # average across ROIs
             .reset_index())

print("\nPer-mouse CNF rates:")
print(per_mouse.sort_values(["group","mouse_ID"]).to_string(index=False))

# ══════════════════════════════════════════════════════════════════════════════
# 4. STATISTICS
# ══════════════════════════════════════════════════════════════════════════════

# Individual fiber CNF calls per group
groups_for_kw = [
    cnf_filtered.loc[cnf_filtered["group"]==t, "Classify_CNF"].values
    for t in TREATMENT_ORDER
    if t in cnf_filtered["group"].values
]

stat, p_kw = kruskal(*groups_for_kw)
print(f"\nKruskal-Wallis (individual fibers): H={stat:.3f}, p={p_kw:.4f}")

dunn = dunns_test(cnf_filtered, "Classify_CNF", "group", p_adjust="bonferroni")
print("\nDunn's post-hoc (Bonferroni, significant pairs only):")
seen = set()
for (g1, g2), p in dunn.items():
    pair = tuple(sorted([g1, g2]))
    if pair in seen: continue
    seen.add(pair)
    if p < 0.05:
        print(f"  {g1} vs {g2}: p={p:.4f} {stars(p)}")

# ══════════════════════════════════════════════════════════════════════════════
# 5. BAR PLOT with individual mouse dots + SE
# ══════════════════════════════════════════════════════════════════════════════

group_stats = (per_mouse.groupby("group")["pct_CNF"]
               .agg(mean="mean", se=lambda x: x.std()/np.sqrt(len(x)))
               .reindex(TREATMENT_ORDER))

positions = np.arange(len(TREATMENT_ORDER))
rng       = np.random.default_rng(42)
Y_MAX     = per_mouse["pct_CNF"].max() * 2.2

fig, ax = plt.subplots(figsize=(10, 7))
fig.subplots_adjust(bottom=0.22)

# Background shading
ax.axvspan(-0.5, 2.5, color=BG_UNINJ, zorder=0)
ax.axvspan( 2.5, 5.5, color=BG_SBI,   zorder=0)
ax.text(1.0, -0.12, "Uninjured", ha="center", va="top",
        fontsize=10, color="#C4603A", fontweight="bold",
        transform=ax.get_xaxis_transform())
ax.text(4.0, -0.12, "SBI (injured)", ha="center", va="top",
        fontsize=10, color="#3A5FA0", fontweight="bold",
        transform=ax.get_xaxis_transform())

# Bars
for i, t in enumerate(TREATMENT_ORDER):
    if t not in group_stats.index: continue
    ax.bar(i, group_stats.loc[t, "mean"],
           color=DOT_COLORS[t], alpha=0.75, width=0.6,
           edgecolor="white", linewidth=0.5, zorder=2)
    ax.errorbar(i, group_stats.loc[t, "mean"],
                yerr=group_stats.loc[t, "se"],
                fmt="none", color="black", linewidth=1.5,
                capsize=4, zorder=3)

# Individual mouse dots
for i, t in enumerate(TREATMENT_ORDER):
    vals = per_mouse.loc[per_mouse["group"]==t, "pct_CNF"].values
    jit  = rng.uniform(-0.12, 0.12, size=len(vals))
    ax.scatter(positions[i]+jit, vals,
               s=55, color="white", edgecolors="black",
               linewidths=1.3, zorder=5)

add_brackets(ax, dunn, Y_MAX)

# X axis labels
ax.set_xticks(positions)
ax.set_xticklabels(SHORT_LABELS, fontsize=10)
ax.set_ylabel("% Centrally Nucleated Fibers", fontsize=11)
ax.set_xlim(-0.5, 5.5)
ax.set_ylim(0, Y_MAX)
ax.spines[["top","right","bottom"]].set_visible(False)
ax.tick_params(axis="x", length=0)
ax.set_title("CNF Rate by Treatment Group", fontsize=12, pad=8)

fig.savefig("cnf_barplot.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print("\ncnf_barplot.png saved.")
