"""
Supplemental Figure 2: 3D body joints under different temporal aggregations.

Compares encoding scores for:
    - '3D body joints'                  (per-frame mean over the video)
    - '3D body joints (concat)'         (all frames concatenated, preserving temporal order)
    - '3D body joints (mean+diff+std)'  (mean + mean frame-to-frame diff + std, concatenated)
"""

import os
import textwrap
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import colors as mcolors

from src.encoding import pose_target_encoding
from src.config import RATING_OF_INTEREST

targets = RATING_OF_INTEREST

pose_features = [
    '3D body joints',
    '3D body joints (concat)',
    '3D body joints (mean+diff+std)',
]

# Distinct greens (varied in hue, not just shade) so the three variants stay
# visually related (all green) but are easy to tell apart at a glance.
color_dict = {
    '3D body joints':                  "#55A829",  # grass green
    '3D body joints (concat)':         "#9DB234",  # olive/lime green
    '3D body joints (mean+diff+std)':  "#257E61",  # teal/forest green
}

_display_name = {
    '3D body joints (mean+diff+std)': '3D body joints\n(position + signed velocity + variability)',
}

_target_display_name = {
    'communication': 'communicative interaction',
    'joint action':  'physical interaction',
}

noise_ceiling = {
    "spatial expanse":     0.7194827819823470,
    "interagent distance": 0.8854506352255450,
    "agents facing":       0.9574809908008190,
    "communication":       0.7621707694002240,
    "joint action":        0.7688673458027810,
}

pose_df = pose_target_encoding(
    pose_features=pose_features, targets=targets, eval_mode='test',
)[["x", "y", "score_r"]]

bar_heights = {(row["x"], row["y"]): float(row["score_r"]) for _, row in pose_df.iterrows()}

print("\nScore (r) per rating and feature:")
for t in targets:
    print(f"  {t}:")
    for pf in pose_features:
        r = bar_heights.get((pf, t), np.nan)
        print(f"    {pf}: r = {r:.4f}")

print("\nAveraged pairwise differences between features (mean over ratings of feature_a - feature_b):")
for i, pf_a in enumerate(pose_features):
    for pf_b in pose_features[i + 1:]:
        diffs = [
            bar_heights[(pf_a, t)] - bar_heights[(pf_b, t)]
            for t in targets
            if (pf_a, t) in bar_heights and (pf_b, t) in bar_heights
        ]
        mean_diff = np.mean(diffs) if diffs else np.nan
        print(f"  {pf_a} - {pf_b}: mean diff = {mean_diff:.4f}")

n_target    = len(targets)
n_hues      = len(pose_features)
total_width = 0.7
bar_width   = total_width / n_hues
x_idx       = np.arange(n_target)

mid_colors  = {k: mcolors.to_rgba(v, alpha=0.55) for k, v in color_dict.items()}
dark_colors = {k: mcolors.to_rgba(v, alpha=0.90) for k, v in color_dict.items()}

fig, ax = plt.subplots(figsize=(12, 5), dpi=300)
ax.grid(axis="y", linestyle="-", linewidth=1.0)
ax.grid(axis="x", visible=False)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

for i, predictor in enumerate(pose_features):
    color     = mid_colors.get(predictor, "gray")
    edgecolor = color_dict.get(predictor, "gray")
    means     = [bar_heights.get((predictor, t), np.nan) for t in targets]
    centers   = x_idx - total_width / 2 + (i + 0.5) * bar_width
    ax.bar(centers, means, width=bar_width, color=color, edgecolor=edgecolor)

    for t, cx, val in zip(targets, centers, means):
        if np.isfinite(val):
            ax.scatter(cx, val, s=60,
                       color=dark_colors.get(predictor, "gray"),
                       edgecolors="white", linewidths=0.5, zorder=4)

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

wrapped_labels = [
    textwrap.fill(_target_display_name.get(t, t), width=10, break_long_words=False)
    for t in targets
]
ax.set_xticks(x_idx)
ax.set_xticklabels(wrapped_labels, ha="center", fontsize=16)
ax.set_ylabel("Score ($r$)", fontsize=16, weight="bold")
ax.set_ylim(-0.25, 1.0)
ax.tick_params(axis="y", labelsize=16)

legend_elements = [
    plt.Line2D([0], [0], marker='o', linestyle='',
               color=dark_colors.get(pf, "gray"),
               markeredgecolor='white', markeredgewidth=0.5, markersize=8,
               label=_display_name.get(pf, pf))
    for pf in pose_features
]
legend_elements.append(
    mpatches.Patch(facecolor=nc_face, edgecolor="none", label="Split-half reliability")
)
ax.legend(handles=legend_elements, bbox_to_anchor=(1, 0.5), fancybox=True)

out_dir = "./results/supp_fig_2.png"
os.makedirs(os.path.dirname(out_dir), exist_ok=True)
plt.tight_layout()
plt.savefig(out_dir, bbox_inches="tight")
plt.show()
print(f"Saved → {out_dir}")
