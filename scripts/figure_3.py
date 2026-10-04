"""
Figure 3: Pose feature encoding scores by rating target.

Four pose features compared across rating targets:
  3D body joints | 2D body joints | 3D social pose features | 2D social pose features

Each bar shows the test-set Pearson r for that pose feature → rating target encoding.
A single dot marks the value on each bar (no model distribution to spread).
Grey horizontal bands show split-half reliability (noise ceiling) per target.
"""

import os
from itertools import combinations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.figstyle import apply_style, save_figure
FIG_WIDTH_MM = 170
apply_style()   # Arial, 7 pt text, editable PDF glyphs
import matplotlib.patches as mpatches
import textwrap
from matplotlib import colors as mcolors

from src.encoding import pose_target_encoding
from src.config import RATING_OF_INTEREST
from src.plottings import default_color_dict, change_name


pose_features = [
    "3D body joints",
    "2D body joints",
    "3D social pose features",
    "2D social pose features",
]
targets = RATING_OF_INTEREST

# ── 1. Pose feature scores ────────────────────────────────────────────────────

pose_df = pose_target_encoding(
    pose_features=pose_features, targets=targets, eval_mode="test",
)[["x", "y", "score_r"]]

print("\nPose feature scores:")
for _, row in pose_df.iterrows():
    print(f"  [{row['x']}] [{row['y']}]  r = {row['score_r']:.4f}")

# ── 1b. Averaged performance across targets & pairwise differences ───────────

mean_by_feature = pose_df.groupby("x")["score_r"].mean().reindex(pose_features)

print("\nAveraged performance (mean r across targets):")
for f in pose_features:
    print(f"  {f:30s} mean r = {mean_by_feature[f]:.4f}")

print("\nPairwise differences in mean r:")
for a, b in combinations(pose_features, 2):
    diff = mean_by_feature[a] - mean_by_feature[b]
    print(f"  {a:28s} - {b:28s} = {diff:+.4f}")

# ── 2. Bar heights ────────────────────────────────────────────────────────────

bar_heights = {}
for _, row in pose_df.iterrows():
    if row["y"] in targets:
        bar_heights[(row["x"], row["y"])] = float(row["score_r"])

# ── 3. Plot ───────────────────────────────────────────────────────────────────

noise_ceiling = {
    "spatial expanse":      0.7194827819823470,
    "object directed":      0.9284563881296680,
    "interagent distance":  0.8854506352255450,
    "agents facing":        0.9574809908008190,
    "joint action":         0.7688673458027810,
    "communication":        0.7621707694002240,
}

# -- Save plotted data --
data_dir = "./results/plot_data/fig3"
os.makedirs(data_dir, exist_ok=True)
pose_df.rename(columns={"x": "predictor", "y": "target"}).to_csv(f"{data_dir}/bars.csv", index=False)
pd.DataFrame(list(noise_ceiling.items()), columns=["target", "noise_ceiling"]).to_csv(
    f"{data_dir}/noise_ceiling.csv", index=False)
print(f"Saved plotted data → {data_dir}/")

n_target    = len(targets)
n_hues      = len(pose_features)
total_width = 0.7
bar_width   = total_width / n_hues
x_idx       = np.arange(n_target)

mid_colors  = {k: mcolors.to_rgba(v, alpha=0.55) for k, v in default_color_dict.items()}
dark_colors = {k: mcolors.to_rgba(v, alpha=0.90) for k, v in default_color_dict.items()}

fig, ax = plt.subplots(figsize=(7.0, 2.8), dpi=300)
ax.grid(axis="y", linestyle="-", linewidth=0.5)
ax.grid(axis="x", visible=False)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# -- Bars and single dots --
for i, predictor in enumerate(pose_features):
    color     = mid_colors.get(predictor, mcolors.to_rgba(default_color_dict.get(predictor, "gray"), alpha=0.55))
    edgecolor = default_color_dict.get(predictor, "gray")
    means     = [bar_heights.get((predictor, t), np.nan) for t in targets]
    centers   = x_idx - total_width / 2 + (i + 0.5) * bar_width
    ax.bar(centers, means, width=bar_width, color=color, edgecolor=edgecolor)

    for t, cx, val in zip(targets, centers, means):
        if np.isfinite(val):
            ax.scatter(cx, val, s=10,
                       color=dark_colors.get(predictor, "gray"),
                       edgecolors="white", linewidths=0.4, zorder=4)

# -- Noise ceiling bands --
nc_face        = "#999999D0"
nc_band_height = 0.024
for ti, target in enumerate(targets):
    if target not in noise_ceiling:
        continue
    nc_val = noise_ceiling[target]
    rect = mpatches.Rectangle(
        (ti - total_width / 2, nc_val - nc_band_height / 2),
        width=total_width * 1.05,
        height=nc_band_height,
        facecolor=nc_face,
        edgecolor="none",
    )
    ax.add_patch(rect)

# -- Axis labels --
wrapped_labels = [
    textwrap.fill(change_name(t), width=12, break_long_words=False)
    for t in targets
]
ax.set_xticks(x_idx)
ax.set_xticklabels(wrapped_labels, ha="center", fontsize=7)
ax.set_ylabel("Score ($r$)", fontsize=7, weight="bold")
ax.set_ylim(-0.25, 1.0)
ax.tick_params(axis="y", labelsize=7)

# -- Legend --
# Each feature's key entry shows its bar with the dot on top (same value).
legend_handles = [
    (mpatches.Patch(facecolor=mid_colors.get(pf, "gray"), edgecolor=default_color_dict.get(pf, "gray")),
     plt.Line2D([0], [0], marker='o', linestyle='',
                color=dark_colors.get(pf, "gray"),
                markeredgecolor='white', markeredgewidth=0.4, markersize=3.8))
    for pf in pose_features
]
legend_labels = [change_name(pf) for pf in pose_features]
legend_handles.append(mpatches.Patch(facecolor=nc_face, edgecolor="none"))
legend_labels.append("Split-half reliability")
ax.legend(legend_handles, legend_labels, bbox_to_anchor=(1, 0.5), fancybox=True, fontsize=7)

save_figure(fig, "./results/fig3.pdf", FIG_WIDTH_MM, relayout=fig.tight_layout)
