"""
Figure 5: DNN + 3D social pose features vs. DNN alone.

Layout : single bar chart, cols = 5 rating targets
Each target: two bars (Vision DNN embeddings | DNN + 3D social pose features)
             + individual model dots (light) + family mean diamonds (dark).

Statistical test: one-sample sign-flip permutation test on family-level
    paired differences (one-tailed, H₀: mean_diff ≤ 0).

    For each family f:
        diff_f = mean_z(ridge)_f − mean_z(dnn)_f   (Fisher z space)

    H₀: mean(diff_f) ≤ 0   (pose adds nothing)
    H₁: mean(diff_f) > 0   (pose improves prediction)

    Monte Carlo sampling of n_perm = 10,000 random sign-flip patterns.
    BH-FDR correction over 5 targets.
"""

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import textwrap
from matplotlib import colors as mcolors
from statsmodels.stats.multitest import multipletests

from src.encoding import get_top_sota_scores
from src.config import SOTA_PLOT_NAME, RATING_OF_INTEREST
from src.plottings import default_color_dict, change_name
from src.stats import stars_from_p


targets     = RATING_OF_INTEREST
RIDGE_LABEL         = f"{SOTA_PLOT_NAME} + \n3D social pose features"
RIDGE_LABEL_DISPLAY = f"{SOTA_PLOT_NAME} + \n3D positions and facing directions"
RIDGE_DIR   = "experiments/ridge_results/3D_social_pose_features"

# ── helpers ────────────────────────────────────────────────────────────────────

def r_to_z(r):
    return np.arctanh(np.clip(np.asarray(r, float), -1 + 1e-7, 1 - 1e-7))

def z_to_r(z):
    return np.tanh(z)


# ── 1. DNN-alone scores ────────────────────────────────────────────────────────

sota_scores = get_top_sota_scores(targets=targets, collect="target")
grouped     = pd.read_csv("data/processed/grouped_models.csv")[["Model UID", "model_family"]]

df_dnn = (
    sota_scores
    .merge(grouped, left_on="model_name", right_on="Model UID", how="left")
    .query("model_family != 'Other' and model_family.notna()")
    .copy()
)


# ── 2. Ridge (DNN + 3D social pose) scores ────────────────────────────────────

ridge_rows = []
for target in targets:
    folder = os.path.join(RIDGE_DIR, target.replace(" ", "_"))
    if not os.path.isdir(folder):
        continue
    for fname in os.listdir(folder):
        if not fname.endswith(".pkl"):
            continue
        with open(os.path.join(folder, fname), "rb") as fh:
            data = pickle.load(fh)
        ridge_rows.append({
            "model_name": fname.replace(".pkl", ""),
            "y":          target,
            "score_r":    float(data["score_r"]),
        })

df_ridge = (
    pd.DataFrame(ridge_rows)
    .merge(grouped, left_on="model_name", right_on="Model UID", how="left")
    .query("model_family != 'Other' and model_family.notna()")
    .copy()
)


# ── 3. Family-level means (Fisher z averaged, back to r for plotting) ──────────

families     = sorted(df_dnn["model_family"].unique())
n_families   = len(families)

print(f"\n{'='*60}")
print(f"Model families (F = {n_families}):  Monte Carlo n_perm=10000")
for fam in families:
    n = df_dnn[df_dnn["model_family"] == fam]["model_name"].nunique()
    print(f"  {fam:<30s}  n = {n}")
print(f"{'='*60}\n")


def compute_family_means(df):
    rows = []
    for fam in families:
        for tgt in targets:
            g = df[(df["model_family"] == fam) & (df["y"] == tgt)]
            if g.empty:
                continue
            mean_z = float(r_to_z(g["score_r"].values).mean())
            rows.append({
                "family":  fam,
                "y":       tgt,
                "score_z": mean_z,
                "score_r": float(z_to_r(mean_z)),
            })
    return pd.DataFrame(rows)


fam_dnn   = compute_family_means(df_dnn)
fam_ridge = compute_family_means(df_ridge)


# ── 4. Sign-flip permutation test (one-tailed, H₀: mean_diff ≤ 0) ─────────────

N_PERM = 10_000

def sign_flip_test_one_tailed(diffs, n_perm: int = N_PERM, seed: int = 42):
    """
    Monte Carlo one-tailed sign-flip test.
    p = fraction of null means ≥ observed mean.
    """
    d     = np.asarray(diffs, float)
    obs   = d.mean()
    F     = len(d)
    rng   = np.random.default_rng(seed)
    signs = rng.choice([-1.0, 1.0], size=(n_perm, F))
    null  = (signs * d).mean(axis=1)
    p     = float((np.sum(null >= obs) + 1) / (n_perm + 1))
    return float(obs), p


sig_info = {}
keys, raw_ps = [], []

print("="*85)
print(f"{'Target':<25}  {'F':>3}  {'delta_r':>10}  {'%fam↑':>8}  {'%model↑':>9}  {'p_raw':>10}")
print("-"*85)
for target in targets:
    dz = fam_dnn[fam_dnn["y"] == target].set_index("family")["score_z"]
    rz = fam_ridge[fam_ridge["y"] == target].set_index("family")["score_z"]
    common = sorted(set(dz.index) & set(rz.index))
    diffs  = np.array([rz[f] - dz[f] for f in common])
    obs_mean_z, p_raw = sign_flip_test_one_tailed(diffs)
    mean_dnn_r   = float(z_to_r(dz[common].values.mean()))
    mean_ridge_r = float(z_to_r(rz[common].values.mean()))
    delta_r      = mean_ridge_r - mean_dnn_r
    pct_fam      = float(np.mean([rz[f] > dz[f] for f in common]) * 100)
    dnn_indiv    = df_dnn[df_dnn["y"] == target].set_index("model_name")["score_r"]
    rdg_indiv    = df_ridge[df_ridge["y"] == target].set_index("model_name")["score_r"]
    common_m     = sorted(set(dnn_indiv.index) & set(rdg_indiv.index))
    pct_indiv    = float(np.mean([rdg_indiv[m] > dnn_indiv[m] for m in common_m]) * 100)
    keys.append(target)
    raw_ps.append(p_raw)
    print(f"{target:<25}  {len(common):>3}  {delta_r:>10.4f}  {pct_fam:>7.1f}%  {pct_indiv:>8.1f}%  {p_raw:>10.4g}")

print("="*85)

_, p_bh, _, _ = multipletests(raw_ps, alpha=0.05, method="fdr_bh")
print(f"\n[FDR] BH correction over {len(targets)} targets:")
for tgt, p_r, p_c in zip(keys, raw_ps, p_bh):
    s = stars_from_p(float(p_c))
    sig_info[tgt] = {"p_raw": float(p_r), "p_corrected": float(p_c), "stars": s}
    print(f"  [{tgt}]  p_raw={p_r:.4g}  p_BH={p_c:.4g}  {s}")
print()


# ── 5. Bar heights (Fisher-z mean across families → r) ────────────────────────

bar_heights = {}
for tgt in targets:
    bar_heights[(SOTA_PLOT_NAME, tgt)] = float(
        z_to_r(fam_dnn[fam_dnn["y"] == tgt]["score_z"].values.mean())
    )
    bar_heights[(RIDGE_LABEL, tgt)] = float(
        z_to_r(fam_ridge[fam_ridge["y"] == tgt]["score_z"].values.mean())
    )


# ── 6. Plot ───────────────────────────────────────────────────────────────────

noise_ceiling = {
    "spatial expanse":      0.7194827819823470,
    "interagent distance":  0.8854506352255450,
    "agents facing":        0.9574809908008190,
    "communication":        0.7621707694002240,
    "joint action":         0.7688673458027810,
}

x_to_plot   = [SOTA_PLOT_NAME, RIDGE_LABEL]
n_target    = len(targets)
n_hues      = len(x_to_plot)
total_width = 0.5
bar_width   = total_width / n_hues
x_idx       = np.arange(n_target)

light_colors = {k: mcolors.to_rgba(v, alpha=0.25) for k, v in default_color_dict.items()}
mid_colors   = {k: mcolors.to_rgba(v, alpha=0.55) for k, v in default_color_dict.items()}
dark_colors  = {k: mcolors.to_rgba(v, alpha=0.90) for k, v in default_color_dict.items()}

fig, ax = plt.subplots(figsize=(12, 5), dpi=300)
ax.grid(axis="y", linestyle="-", linewidth=1.0)
ax.grid(axis="x", visible=False)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

bar_center = {}

# -- Bars --
for i, predictor in enumerate(x_to_plot):
    color     = mid_colors.get(predictor, mcolors.to_rgba("gray", alpha=0.55))
    edgecolor = default_color_dict.get(predictor, "gray")
    means     = [bar_heights.get((predictor, t), np.nan) for t in targets]
    centers   = x_idx - total_width / 2 + (i + 0.5) * bar_width
    ax.bar(centers, means, width=bar_width, color=color, edgecolor=edgecolor)
    for t, cx in zip(targets, centers):
        bar_center[(predictor, t)] = float(cx)

# -- Individual model dots (light, jittered) --
rng = np.random.default_rng(42)
for i, (predictor, df_pts) in enumerate([(SOTA_PLOT_NAME, df_dnn), (RIDGE_LABEL, df_ridge)]):
    dot_color = light_colors.get(predictor, mcolors.to_rgba("gray", alpha=0.25))
    for _, row in df_pts.iterrows():
        if str(row["y"]) not in targets:
            continue
        t_idx  = targets.index(str(row["y"]))
        center = t_idx - total_width / 2 + (i + 0.5) * bar_width
        jitter = rng.uniform(-bar_width * 0.35, bar_width * 0.35)
        ax.scatter(center + jitter, row["score_r"], s=8,
                   color=dot_color, zorder=2, linewidths=0)

# -- Family mean diamonds (dark) --
dnn_label_added = False
for i, (predictor, fam_df_) in enumerate([(SOTA_PLOT_NAME, fam_dnn), (RIDGE_LABEL, fam_ridge)]):
    fam_color = dark_colors.get(predictor, mcolors.to_rgba("gray", alpha=0.90))
    for _, row in fam_df_.iterrows():
        if str(row["y"]) not in targets:
            continue
        t_idx  = targets.index(str(row["y"]))
        center = t_idx - total_width / 2 + (i + 0.5) * bar_width
        if predictor == SOTA_PLOT_NAME and not dnn_label_added:
            label = "DNN family mean"
            dnn_label_added = True
        else:
            label = None
        ax.scatter(center, row["score_r"], s=30, marker="D",
                   color=fam_color, edgecolors="white", linewidths=0.5,
                   zorder=4, label=label)

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
dnn_light  = light_colors.get(SOTA_PLOT_NAME, mcolors.to_rgba("gray", 0.25))
dnn_dark   = dark_colors.get(SOTA_PLOT_NAME,  mcolors.to_rgba("gray", 0.90))
rdg_light  = light_colors.get(RIDGE_LABEL,    mcolors.to_rgba("gray", 0.25))
rdg_dark   = dark_colors.get(RIDGE_LABEL,     mcolors.to_rgba("gray", 0.90))

legend_elements = [
    plt.Line2D([0], [0], marker='o', linestyle='', color=dnn_light,
               markersize=6, label=change_name(SOTA_PLOT_NAME)),
    plt.Line2D([0], [0], marker='D', linestyle='', color=dnn_dark,
               markeredgecolor='white', markeredgewidth=0.5, markersize=7,
               label="DNN family mean"),
    plt.Line2D([0], [0], marker='o', linestyle='', color=rdg_light,
               markersize=6, label=RIDGE_LABEL_DISPLAY),
    plt.Line2D([0], [0], marker='D', linestyle='', color=rdg_dark,
               markeredgecolor='white', markeredgewidth=0.5, markersize=7,
               label="DNN+3D features family mean"),
    mpatches.Patch(facecolor=nc_face, edgecolor="none", label="Split-half reliability"),
]
ax.legend(handles=legend_elements, bbox_to_anchor=(1, 0.5), fancybox=True, fontsize=13)

# -- Significance brackets --
def _add_sig_bracket(ax, x1, x2, y, h=0.01, text="*", lw=1.5, fontsize=12):
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], linewidth=lw, color="#555555")
    ax.text((x1 + x2) / 2, y + h, text, ha="center", va="bottom",
            fontsize=fontsize, color="#555555")

base_bracket_y = 1.0
bracket_tick   = 0.01

for target in targets:
    info  = sig_info.get(target, {})
    stars = info.get("stars", "")
    if not stars:
        continue
    x1 = bar_center.get((SOTA_PLOT_NAME, target))
    x2 = bar_center.get((RIDGE_LABEL, target))
    if x1 is None or x2 is None:
        continue
    _add_sig_bracket(ax, x1, x2, y=base_bracket_y, h=bracket_tick, text=stars)

if any(sig_info.get(t, {}).get("stars") for t in targets):
    ax.set_ylim(-0.25, base_bracket_y + 0.07)

out_dir = "./results/fig5.png"
os.makedirs(os.path.dirname(out_dir), exist_ok=True)
plt.tight_layout()
plt.savefig(out_dir, bbox_inches="tight", dpi=300)
plt.show()
print(f"Saved → {out_dir}")
