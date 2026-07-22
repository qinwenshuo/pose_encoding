"""
Supplemental Figure 3: Semipartial correlation analysis of 3D body joints.

Tests how much of the '3D body joints' encoding score survives after partialling
out related pose features (3D social pose features, 3D head directions, 3D head
positions), for each behavioral target.
"""

import os
import textwrap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import colors as mcolors
from joblib import Parallel, delayed

from src.encoding import pose_target_encoding
from src.semi_partial import get_semi_partial_scores

targets = ['spatial expanse', 'interagent distance', 'agents facing', 'communication', 'joint action']

pose_encoding_results = pose_target_encoding(pose_features=['3D body joints'], targets=targets, eval_mode='test')

# Define the Z-features to partial out and their labels
_partial_specs = [
    ("3D head positions",         "joints partial out \n3D positions"),
    ("3D head directions",        "joints partial out \n3D facing directions"),
    ("3D social pose features",   "joints partial out \n3D positions and facing directions"),
]

pose_features = ['3D body joints'] + [name for _, name in _partial_specs]

color_dict = {
    '3D body joints':                                              "#56AA29",
    'joints partial out \n3D positions':                           "#526031",
    'joints partial out \n3D facing directions':                   "#326C6C",
    'joints partial out \n3D positions and facing directions':     "#40342B",
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


def _run_one(target, pose_feat_z, partial_name):
    # Each job computes a single semipartial score dict/row
    return get_semi_partial_scores(
        pose_feat_x="3D body joints",
        pose_feat_z=pose_feat_z,
        target_y=target,
        partial_name=partial_name,
    )

# Build all jobs (cartesian product: targets × partial_specs)
_jobs = [(t, z, name) for t in targets for (z, name) in _partial_specs]

# Run in parallel; change n_jobs as you like (e.g., n_jobs=8)
semipartial_rows = Parallel(n_jobs=-1, backend="loky", prefer="processes")(
    delayed(_run_one)(t, z, name) for (t, z, name) in _jobs
)

pose_df = pd.concat(
    [pose_encoding_results, pd.DataFrame(semipartial_rows)], axis=0, ignore_index=True
)[["x", "y", "score_r"]]

bar_heights = {(row["x"], row["y"]): float(row["score_r"]) for _, row in pose_df.iterrows()}

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
               label=pf)
    for pf in pose_features
]
legend_elements.append(
    mpatches.Patch(facecolor=nc_face, edgecolor="none", label="Split-half reliability")
)
ax.legend(handles=legend_elements, bbox_to_anchor=(1, 0.5), fancybox=True)

out_dir = "./results/supp_fig_3.png"
os.makedirs(os.path.dirname(out_dir), exist_ok=True)
plt.tight_layout()
plt.savefig(out_dir, bbox_inches="tight")
plt.show()
print(f"Saved → {out_dir}")
