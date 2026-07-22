"""
Figure 2: DNN performance vs. pose features grouped by model family.

Visual design
─────────────
• All individual DNN model dots: lighter color, jittered within their bar slot.
• DNN family means: diamond marker, darker color — the independent units for stats.
• Pose feature dots/bars: plotted as usual.
• All plotted values are on the raw Pearson r scale.
• Split-half reliability bands: grey horizontal spans per target.

Statistical test
────────────────
One-sample sign-flip permutation test on family-level differences (two-tailed).
All averaging and differencing is done in Fisher z space; results are back-transformed
to r only for plotting.

For each pose feature and each rating target:
    z_f      = mean( arctanh(model_r) ) within family f        (Fisher z average)
    pose_z   = arctanh(pose_r)
    d_f      = pose_z - z_f,   f = 1…F families

    H0: mean(d_f) = 0
    H1: mean(d_f) ≠ 0  (two-tailed)

Signs of the F differences are randomly flipped to build the null distribution.
BH-FDR correction via statsmodels, applied per pose feature across targets.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import textwrap
from matplotlib import colors as mcolors
from statsmodels.stats.multitest import multipletests

from src.encoding import get_top_sota_scores, pose_target_encoding, get_4dhumans_top_scores
from src.config import RATING_OF_INTEREST, SOTA_PLOT_NAME
from src.plottings import default_color_dict, change_name
from src.stats import stars_from_p


targets = RATING_OF_INTEREST
pose_features = ["3D body joints"]
VIDEO_FAMILY = "Video"
N_PERM = 10_000

# ── helpers ───────────────────────────────────────────────────────────────────

def r_to_z(r):
    return np.arctanh(np.clip(r, -1 + 1e-7, 1 - 1e-7))

def z_to_r(z):
    return np.tanh(z)

# ── 1. Individual DNN model scores ────────────────────────────────────────────

sota_scores = get_top_sota_scores(targets=targets, collect="target")
grouped = pd.read_csv("data/processed/grouped_models.csv")[["Model UID", "model_family"]]
df_dnn = sota_scores.merge(grouped, left_on="model_name", right_on="Model UID", how="left")
df_dnn = df_dnn[df_dnn["model_family"] != "Other"]

# ── 2. Family-level means (Fisher z averaging) ────────────────────────────────

families     = sorted(df_dnn["model_family"].unique())
n_families   = len(families)

print(f"\n{'='*60}")
print(f"Model families (F = {n_families}):  Monte Carlo n_perm={N_PERM} sign-flip permutations")
for fam in families:
    n_models = df_dnn[df_dnn["model_family"] == fam]["model_name"].nunique()
    print(f"  {fam:<30s}  n = {n_models}")
print(f"{'='*60}\n")

fam_rows = []
for family in families:
    for target in df_dnn["y"].unique():
        g = df_dnn[(df_dnn["model_family"] == family) & (df_dnn["y"] == target)]
        if g.empty:
            continue
        z_vals = r_to_z(g["score_r"].values)
        mean_z = float(z_vals.mean())
        fam_rows.append({
            "family":  family,
            "y":       str(target),
            "score_z": mean_z,
            "score_r": float(z_to_r(mean_z)),   # back to r for plotting
        })
family_df = pd.DataFrame(fam_rows)

# ── 3. Pose feature scores ────────────────────────────────────────────────────

pose_df = pose_target_encoding(
    pose_features=pose_features, targets=targets, eval_mode="test",
)[["x", "y", "score_r"]]

# ── 3b. Pose model embeddings (4DHumans, plot-only — no significance testing) ──

POSE_MODEL_NAME = "3D pose model embeddings"
pose_model_df = get_4dhumans_top_scores(targets=targets, top_n=1, collect="target")[
    ["x", "y", "score_r"]
].copy()
pose_model_df["x"] = POSE_MODEL_NAME   # rename from SOTA_PLOT_NAME to its own slot

# ── 4. Sign-flip permutation test (Fisher z space, two-tailed) ────────────────

def sign_flip_test(diffs, n_perm: int = N_PERM, seed: int = 42):
    """
    One-sample sign-flip permutation test (two-tailed), via Monte Carlo
    sampling of n_perm random sign-flip patterns (with replacement).
    diffs : array of length F (one value per family, in Fisher z space)
    Returns (observed_mean_z, two-tailed p-value).
    """
    d     = np.asarray(diffs, float)
    obs   = d.mean()
    F     = len(d)
    rng   = np.random.default_rng(seed)
    signs = rng.choice([-1.0, 1.0], size=(n_perm, F))
    null  = (signs * d).mean(axis=1)
    p     = float((np.sum(np.abs(null) >= np.abs(obs)) + 1) / (n_perm + 1))
    return float(obs), p


sig_info: dict = {}

for pose_feat in pose_features:
    keys, raw_ps = [], []
    for target in targets:
        p_row = pose_df.loc[
            (pose_df["x"] == pose_feat) & (pose_df["y"] == target), "score_r"
        ]
        if p_row.empty:
            continue
        pose_r   = float(p_row.iloc[0])
        pose_z   = float(r_to_z(pose_r))
        fam_z    = family_df[family_df["y"] == target]["score_z"].values
        fam_r    = family_df[family_df["y"] == target]["score_r"].values
        diffs    = pose_z - fam_z           # differences in Fisher z space
        obs_mean_z, p_raw = sign_flip_test(diffs)
        mean_fam_r = float(z_to_r(fam_z.mean()))
        delta_r    = pose_r - mean_fam_r
        keys.append(target)
        raw_ps.append(p_raw)

        # Percentile reporting
        indiv_r   = df_dnn[df_dnn["y"] == target]["score_r"].values
        pct_indiv = float(np.mean(pose_r > indiv_r) * 100)
        pct_fam   = float(np.mean(pose_r > fam_r) * 100)
        n_indiv   = len(indiv_r)
        n_fam     = len(fam_r)

        print(
            f"[{pose_feat}] [{target}]  "
            f"pose_r={pose_r:.4f}  mean_fam_r={mean_fam_r:.4f}  "
            f"delta_r={delta_r:.4f}  p_raw={p_raw:.4f}\n"
            f"    exceeds {pct_indiv:.1f}% of individual checkpoints ({n_indiv} total)"
            f",  {pct_fam:.1f}% of family means ({n_fam} families)"
        )

    n_tests = len(raw_ps)
    _, p_bh, _, _ = multipletests(raw_ps, alpha=0.05, method="fdr_bh")
    print(f"\n[FDR] BH correction for '{pose_feat}' over {n_tests} targets"
          f"  (F={n_families}, n_perm={N_PERM}):")
    for t, p_r, p_c in zip(keys, raw_ps, p_bh):
        s = stars_from_p(float(p_c))
        sig_info[(pose_feat, t)] = {
            "p_raw":       float(p_r),
            "p_corrected": float(p_c),
            "stars":       s,
        }
        print(f"  [{t}]  p_raw={p_r:.4g}  p_BH={p_c:.4g}  {s}")

# ── 4b. Pose feature vs. Video DNN family mean (purple diamonds) ─────────────

video_fam_df = family_df[family_df["family"] == VIDEO_FAMILY]

print(f"\n{'='*60}")
print("Pose feature vs. Video DNN family mean (purple diamonds)")
print(f"{'='*60}")
for pose_feat in pose_features:
    diffs_r = []
    for target in targets:
        p_row = pose_df.loc[(pose_df["x"] == pose_feat) & (pose_df["y"] == target), "score_r"]
        v_row = video_fam_df.loc[video_fam_df["y"] == target, "score_r"]
        if p_row.empty or v_row.empty:
            continue
        pose_r  = float(p_row.iloc[0])
        video_r = float(v_row.iloc[0])
        delta_r = pose_r - video_r
        diffs_r.append(delta_r)
        print(f"  [{pose_feat}] [{target}]  pose_r={pose_r:.4f}  video_dnn_r={video_r:.4f}  delta_r={delta_r:+.4f}")
    if diffs_r:
        print(f"  --> [{pose_feat}] averaged delta_r across {len(diffs_r)} targets = {np.mean(diffs_r):+.4f}\n")

# ── 5. Build bar heights (raw r scale for display) ───────────────────────────

# DNN bar: tanh of mean z across families (consistent Fisher z chain)
bar_heights: dict = {}
for target in targets:
    fam_z_vals = family_df[family_df["y"] == target]["score_z"].values
    bar_heights[(SOTA_PLOT_NAME, target)] = float(z_to_r(fam_z_vals.mean()))
for _, row in pose_df.iterrows():
    if row["y"] in targets:
        bar_heights[(row["x"], row["y"])] = float(row["score_r"])

# ── 6. Plot ───────────────────────────────────────────────────────────────────

noise_ceiling = {
    "spatial expanse":      0.7194827819823470,
    "object directed":      0.9284563881296680,
    "interagent distance":  0.8854506352255450,
    "agents facing":        0.9574809908008190,
    "joint action":         0.7688673458027810,
    "communication":        0.7621707694002240,
}

x_to_plot   = [SOTA_PLOT_NAME] + pose_features
color_dict  = default_color_dict

n_target    = len(targets)
n_hues      = len(x_to_plot)
total_width = 0.7
bar_width   = total_width / n_hues
x_idx       = np.arange(n_target)

light_colors = {k: mcolors.to_rgba(v, alpha=0.25) for k, v in color_dict.items()}
mid_colors   = {k: mcolors.to_rgba(v, alpha=0.55) for k, v in color_dict.items()}
dark_colors  = {k: mcolors.to_rgba(v, alpha=0.90) for k, v in color_dict.items()}

fig, ax = plt.subplots(figsize=(12, 5), dpi=300)
ax.grid(axis="y", linestyle="-", linewidth=1.0)
ax.grid(axis="x", visible=False)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

bar_center: dict = {}

# -- Bars --
for i, predictor in enumerate(x_to_plot):
    color     = mid_colors.get(predictor, mcolors.to_rgba(color_dict.get(predictor, "gray"), alpha=0.55))
    edgecolor = color_dict.get(predictor, "gray")
    means     = [bar_heights.get((predictor, t), np.nan) for t in targets]
    centers   = x_idx - total_width / 2 + (i + 0.5) * bar_width
    ax.bar(centers, means, width=bar_width, color=color, edgecolor=edgecolor)
    for t, cx in zip(targets, centers):
        bar_center[(predictor, t)] = float(cx)

# -- Individual DNN dots (lighter, jittered, raw r) --
rng       = np.random.default_rng(42)
dnn_color = light_colors.get(SOTA_PLOT_NAME, "gray")
model_idx = x_to_plot.index(SOTA_PLOT_NAME)
for _, row in df_dnn.iterrows():
    if str(row["y"]) not in targets:
        continue
    t_idx  = targets.index(str(row["y"]))
    center = t_idx - total_width / 2 + (model_idx + 0.5) * bar_width
    jitter = rng.uniform(-bar_width * 0.35, bar_width * 0.35)
    ax.scatter(center + jitter, row["score_r"], s=8,
               color=dnn_color, zorder=2, linewidths=0)

# -- Family mean diamonds (darker, Fisher-z averaged back to r) --
fam_color               = dark_colors.get(SOTA_PLOT_NAME, "gray")
video_fam_color         = mcolors.to_rgba(color_dict.get("Video DNNs", "#9B59B6"), alpha=0.90)
family_label_added      = False
video_family_label_added = False
for _, row in family_df.iterrows():
    if str(row["y"]) not in targets:
        continue
    t_idx   = targets.index(str(row["y"]))
    center  = t_idx - total_width / 2 + (model_idx + 0.5) * bar_width
    is_video = row["family"] == VIDEO_FAMILY
    if is_video:
        label = "Video DNN family mean" if not video_family_label_added else None
        ax.scatter(center, row["score_r"], s=50, marker="D",
                   color=video_fam_color, edgecolors="black", linewidths=0.5,
                   zorder=4, label=label)
        video_family_label_added = True
    else:
        label = "DNN family mean" if not family_label_added else None
        ax.scatter(center, row["score_r"], s=30, marker="D",
                   color=fam_color, edgecolors="white", linewidths=0.5,
                   zorder=4, label=label)
        family_label_added = True

# -- Pose feature dots --
for _, row in pose_df.iterrows():
    if row["x"] not in x_to_plot or str(row["y"]) not in targets:
        continue
    t_idx  = targets.index(str(row["y"]))
    p_idx  = x_to_plot.index(row["x"])
    center = t_idx - total_width / 2 + (p_idx + 0.5) * bar_width
    ax.scatter(center, row["score_r"], s=35,
               color=dark_colors.get(row["x"], "gray"),
               edgecolors="white", linewidths=0.5, zorder=4)

# -- Pose model embedding dots (4DHumans) — overlaid on the SOTA bar --
pose_model_color   = color_dict.get(POSE_MODEL_NAME, "#25A2AE")
pose_model_label_added = False
for _, row in pose_model_df.iterrows():
    if str(row["y"]) not in targets:
        continue
    t_idx  = targets.index(str(row["y"]))
    center = t_idx - total_width / 2 + (model_idx + 0.5) * bar_width
    label  = POSE_MODEL_NAME if not pose_model_label_added else None
    ax.scatter(center, row["score_r"], s=90, marker="o",
               color=pose_model_color, edgecolors="black", linewidths=1.0,
               zorder=5, label=label)
    pose_model_label_added = True

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
    textwrap.fill(change_name(t), width=10, break_long_words=False)
    for t in targets
]
ax.set_xticks(x_idx)
ax.set_xticklabels(wrapped_labels, ha="center", fontsize=16)
ax.set_ylabel("Score ($r$)", fontsize=16, weight="bold")
ax.set_ylim(-0.25, 1.0)
ax.tick_params(axis="y", labelsize=16)

# -- Legend --
legend_elements = [
    plt.Line2D([0], [0], marker='o', linestyle='',
               color=light_colors.get(SOTA_PLOT_NAME, "gray"), markersize=6,
               markeredgewidth=0, label=f"Individual vision DNN embeddings"),
    plt.Line2D([0], [0], marker='D', linestyle='',
               color=dark_colors.get(SOTA_PLOT_NAME, "gray"),
               markeredgecolor='white', markeredgewidth=0.5, markersize=7,
               label="DNN family mean"),
    plt.Line2D([0], [0], marker='D', linestyle='',
               color=video_fam_color,
               markeredgecolor='black', markeredgewidth=0.5, markersize=7,
               label="Video DNN family mean"),
    plt.Line2D([0], [0], marker='o', linestyle='',
               color=pose_model_color, markeredgecolor='black', markeredgewidth=1.0,
               markersize=9, label=change_name(POSE_MODEL_NAME)),
] + [
    plt.Line2D([0], [0], marker='o', linestyle='',
               color=dark_colors.get(pf, "gray"),
               markeredgecolor='white', markeredgewidth=0.5, markersize=7,
               label=change_name(pf))
    for pf in pose_features
] + [
    mpatches.Patch(facecolor=nc_face, edgecolor="none", label="Split-half reliability"),
]
ax.legend(handles=legend_elements, bbox_to_anchor=(1, 0.5), fancybox=True)

# -- Significance brackets --
def _add_sig_bracket(ax, x1, x2, y, h=0.01, text="*", lw=1.5, fontsize=12):
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], linewidth=lw, color="#555555")
    ax.text((x1 + x2) / 2, y + h, text, ha="center", va="bottom",
            fontsize=fontsize, color="#555555")


bracket_gap    = 0.07
bracket_tick   = 0.01
base_bracket_y = 1.0
stack          = {t: 0 for t in targets}

for pose_feat in pose_features:
    for target in targets:
        info  = sig_info.get((pose_feat, target), {})
        stars = info.get("stars", "")
        if not stars:
            continue
        x1 = bar_center.get((SOTA_PLOT_NAME, target))
        x2 = bar_center.get((pose_feat, target))
        if x1 is None or x2 is None:
            continue
        y0 = base_bracket_y + bracket_gap * stack[target]
        _add_sig_bracket(ax, x1, x2, y=y0, h=bracket_tick, text=stars)
        stack[target] += 1

if stack and max(stack.values()) > 0:
    need_hi = base_bracket_y + bracket_gap * max(stack.values()) + 0.05
    lo, hi = ax.get_ylim()
    if need_hi > hi:
        ax.set_ylim(lo, need_hi)

out_dir = "./results/fig2.png"
os.makedirs(os.path.dirname(out_dir), exist_ok=True)
plt.tight_layout()
plt.savefig(out_dir, bbox_inches="tight")
plt.show()
print(f"Saved → {out_dir}")
