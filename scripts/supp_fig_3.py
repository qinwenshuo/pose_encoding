"""
Supplemental Figure 3: DNN encoding performance on the 500-video set (original 250 +
additional 250), each model's best layer plotted as a dot, grouped by model family,
compared against 3D body joints.

Nested-CV methodology (see scripts/06_dnn_nested_cv500_encoding.py): an outer 5-fold
rotation over the 500 videos (data/raw/dyad_videos/folds_500.csv, fixed/shared across
every model so results are comparable) -- each outer fold trains on the other 400 and
tests on its 100. Within those 400, every layer is scored via encode()'s own
eval_mode='cv' (unmodified, figure_2's exact inner-CV call) to select the best layer
per target, which is then evaluated honestly via encode(eval_mode='test') on the 100
held out, never touched during selection. Repeating over all 5 outer folds covers all
500 videos as test exactly once; this script pools those out-of-fold z-scored
predictions into a single Pearson r per model/target (loaded here via
_load_nested_500_scores, which only reads the already-computed .pkl files -- no
recompute). 3D body joints is scored the same way (section 3 below: same 5 outer
folds, same pooling, just no inner-CV layer-selection step since there's only one
feature) for an apples-to-apples comparison. Visual style (individual DNN dots +
family-mean diamonds, Video family highlighted) mirrors figure_2.py.

Statistical test (mirrors figure_2.py): one-sample sign-flip permutation test on
family-level differences (two-tailed), in Fisher z space, BH-FDR corrected across
the 4 targets.
    z_f    = mean( arctanh(model_r) ) within family f   (Fisher z average)
    pose_z = arctanh(pose_r)
    d_f    = pose_z - z_f,   f = 1…F families
    H0: mean(d_f) = 0   H1: mean(d_f) != 0
"""

import os
import json
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.figstyle import apply_style, save_figure
FIG_WIDTH_MM = 170
apply_style()   # Arial, 7 pt text, editable PDF glyphs
import matplotlib.patches as mpatches
from matplotlib import colors as mcolors
from scipy.stats import pearsonr
from statsmodels.stats.multitest import multipletests

from src.encoding import encode
from src.data_utils import load_pickle, zscore_fit_apply
from src.config import SOTA_PLOT_NAME, ALPHAS
from src.plottings import default_color_dict, change_name
from src.stats import stars_from_p

# video_ratings.csv (500 videos) has no 'agents facing' column.
targets = ['spatial expanse', 'interagent distance', 'communication', 'joint action']
VIDEO_FAMILY = "Video"
N_PERM = 10_000

NESTED_BEH_PATH = 'experiments/SOTA_beh_500_nested'
FOLDS_500_PATH = 'data/raw/dyad_videos/folds_500.csv'
N_OUTER = 5

POSE_FEATURE = '3D body joints'
POSE_VIDEO_LEVEL_PATH = 'data/processed/dyad_videos_500/mesh_video_level_features/'
POSE_RATINGS_PATH = 'data/raw/dyad_videos/video_ratings.csv'
POSE_TARGET_COLS = {
    'spatial expanse':     'spatial_expanse',
    'interagent distance': 'agent_distance',
    'communication':       'communication',
    'joint action':        'joint_action',
}


def _load_nested_500_scores(targets, model_type='both', top_n=1):
    """
    Reads experiments/SOTA_beh_500_nested pooled results (see
    scripts/06_dnn_nested_cv500_encoding.py) and computes each model's final pooled
    Pearson r per target directly from the saved z-scored (true, predicted) pairs --
    no recompute. Local to supp_fig_3.py; not a general-purpose loader.
    """
    model_types = ['image_models', 'video_models'] if model_type == 'both' else [model_type]
    rows = []
    for m_type in model_types:
        pooled_dir = Path(NESTED_BEH_PATH) / m_type / "pooled"
        if not pooled_dir.exists():
            print(f"[Warning] No nested-CV results for {m_type} at {pooled_dir}. Skipping.")
            continue
        for pkl_path in sorted(pooled_dir.glob("*_pooled.pkl")):
            model = pkl_path.stem[: -len("_pooled")]
            d = load_pickle(str(pkl_path))
            pooled_df = d['pooled']
            for t in targets:
                sub = pooled_df[pooled_df['y'] == t]
                if sub.empty:
                    continue
                r, _ = pearsonr(sub['y_true_z'], sub['y_pred_z'])
                rows.append({'model_name': model, 'x': SOTA_PLOT_NAME, 'y': t,
                             'score_r': float(r), 'model_type': m_type})
    return pd.DataFrame(rows)


def r_to_z(r):
    return np.arctanh(np.clip(r, -1 + 1e-7, 1 - 1e-7))


def z_to_r(z):
    return np.tanh(z)


# ── 1. Each DNN's nested-CV pooled score (best layer per outer fold, selected via
#      inner CV on that fold's 400 train, honestly scored on that fold's 100 test,
#      pooled across all 5 outer folds) ────────────────────────────────────────

sota_scores = _load_nested_500_scores(targets=targets)
if sota_scores.empty:
    raise SystemExit(
        "No 500-video nested-CV results found under experiments/SOTA_beh_500_nested/. "
        "Run scripts/06_dnn_nested_cv500_encoding.py first."
    )

grouped = pd.read_csv("data/processed/grouped_models.csv")[["Model UID", "model_family"]]
df_dnn = sota_scores.merge(grouped, left_on="model_name", right_on="Model UID", how="left")
df_dnn = df_dnn[df_dnn["model_family"] != "Other"]

# ── 2. Family-level means (Fisher z averaging) ────────────────────────────────

families = sorted(df_dnn["model_family"].dropna().unique())
print(f"\n{'='*60}")
print(f"Model families (F = {len(families)}), 500-video nested 5-fold CV")
for fam in families:
    n_models = df_dnn[df_dnn["model_family"] == fam]["model_name"].nunique()
    print(f"  {fam:<30s}  n = {n_models}")
print(f"{'='*60}\n")

fam_rows = []
for family in families:
    for target in targets:
        g = df_dnn[(df_dnn["model_family"] == family) & (df_dnn["y"] == target)]
        if g.empty:
            continue
        z_vals = r_to_z(g["score_r"].values)
        mean_z = float(z_vals.mean())
        fam_rows.append({
            "family":  family,
            "y":       target,
            "score_z": mean_z,
            "score_r": float(z_to_r(mean_z)),
        })
family_df = pd.DataFrame(fam_rows)

# ── 3. 3D body joints: same 5-outer-fold nested-CV pooling as the DNNs ────────
# Loads from the mesh video-level features json + video_ratings.csv (same source
# supp_fig_6.py reads, though that script now uses a different, plain 10-fold CV
# methodology -- not this nested/pooled one), scored on the same folds_500.csv outer
# rotation as the DNNs (encode(eval_mode='test') per fold, pooled across all 5) so
# the comparison here is apples-to-apples with the DNNs above. No inner-CV layer
# selection needed -- there's only one feature, nothing to select between.

def _load_pose_xy_named(pose_feature, target_col, video_level_path, ratings_path):
    ratings = pd.read_csv(ratings_path).set_index('video_name')
    X_list, y_list, names = [], [], []
    for fname in sorted(os.listdir(video_level_path)):
        if not fname.endswith('.json'):
            continue
        video_name = fname.replace('.json', '.mp4')
        if video_name not in ratings.index:
            continue
        with open(os.path.join(video_level_path, fname)) as f:
            feats = json.load(f)
        row = feats.get(pose_feature)
        if row is None or None in row:
            continue
        X_list.append(row)
        y_list.append(ratings.at[video_name, target_col])
        names.append(video_name)
    return np.array(X_list, dtype=float), np.array(y_list, dtype=float), names


folds_df = pd.read_csv(FOLDS_500_PATH).set_index('video_name')

pose_rows = []
for target in targets:
    X, y, names = _load_pose_xy_named(POSE_FEATURE, POSE_TARGET_COLS[target], POSE_VIDEO_LEVEL_PATH, POSE_RATINGS_PATH)
    names = np.array(names)
    fold_of = folds_df.loc[names, 'fold'].to_numpy()

    pooled_true = np.full(len(names), np.nan)
    pooled_pred = np.full(len(names), np.nan)
    for outer_fold in range(N_OUTER):
        train_mask = fold_of != outer_fold
        test_mask = fold_of == outer_fold
        y_train, y_test = y[train_mask], y[test_mask]
        result = encode(
            X_train=X[train_mask], X_test=X[test_mask], y_train=y_train, y_test=y_test,
            x_type=POSE_FEATURE, layer_name=None, feature=target,
            eval_mode='test', alphas=ALPHAS,
        )
        _, y_test_n = zscore_fit_apply(y_train[:, None], y_test[:, None], axis=0)
        test_idx_positions = np.where(test_mask)[0]
        pooled_pred[test_idx_positions] = result['y_hat']
        pooled_true[test_idx_positions] = y_test_n.ravel()

    assert not np.isnan(pooled_pred).any(), "every video should be predicted exactly once"
    r, _ = pearsonr(pooled_true, pooled_pred)
    pose_rows.append({'x': POSE_FEATURE, 'y': target, 'score_r': float(r), 'n': len(names)})
pose_df = pd.DataFrame(pose_rows)
print(f"\n[{POSE_FEATURE}] nested 5-fold pooled scores:")
print(pose_df[['y', 'score_r', 'n']].to_string(index=False))

# ── 4. Sign-flip permutation test: 3D body joints vs. DNN family means ───────

def sign_flip_test(diffs, n_perm: int = N_PERM, seed: int = 42):
    d = np.asarray(diffs, float)
    obs = d.mean()
    F = len(d)
    rng = np.random.default_rng(seed)
    signs = rng.choice([-1.0, 1.0], size=(n_perm, F))
    null = (signs * d).mean(axis=1)
    p = float((np.sum(np.abs(null) >= np.abs(obs)) + 1) / (n_perm + 1))
    return float(obs), p


sig_info = {}
keys, raw_ps = [], []
print(f"\n{'='*60}")
print(f"[{POSE_FEATURE}] vs. DNN family means (Monte Carlo n_perm={N_PERM} sign-flip permutations)")
print(f"{'='*60}")
for target in targets:
    p_row = pose_df.loc[pose_df["y"] == target, "score_r"]
    fam_z = family_df[family_df["y"] == target]["score_z"].values
    fam_r = family_df[family_df["y"] == target]["score_r"].values
    if p_row.empty or len(fam_z) == 0:
        continue
    pose_r = float(p_row.iloc[0])
    pose_z = float(r_to_z(pose_r))
    diffs = pose_z - fam_z
    obs_mean_z, p_raw = sign_flip_test(diffs)
    mean_fam_r = float(z_to_r(fam_z.mean()))
    keys.append(target)
    raw_ps.append(p_raw)

    indiv_r = df_dnn[df_dnn["y"] == target]["score_r"].values
    pct_indiv = float(np.mean(pose_r > indiv_r) * 100)
    pct_fam = float(np.mean(pose_r > fam_r) * 100)
    print(
        f"[{target}]  pose_r={pose_r:.4f}  mean_fam_r={mean_fam_r:.4f}  "
        f"delta_r={pose_r - mean_fam_r:+.4f}  p_raw={p_raw:.4g}\n"
        f"    exceeds {pct_indiv:.1f}% of individual DNN checkpoints (n={len(indiv_r)}), "
        f"{pct_fam:.1f}% of family means (F={len(fam_r)})"
    )

_, p_bh, _, _ = multipletests(raw_ps, alpha=0.05, method="fdr_bh")
print(f"\n[FDR] BH correction over {len(raw_ps)} targets:")
for t, p_r, p_c in zip(keys, raw_ps, p_bh):
    s = stars_from_p(float(p_c))
    sig_info[t] = {"p_raw": float(p_r), "p_corrected": float(p_c), "stars": s}
    print(f"  [{t}]  p_raw={p_r:.4g}  p_BH={p_c:.4g}  {s}")

# ── 5. Bar heights per (predictor, target) ────────────────────────────────────

x_to_plot = [SOTA_PLOT_NAME, POSE_FEATURE]
bar_heights = {}
for target in targets:
    fam_z_vals = family_df[family_df["y"] == target]["score_z"].values
    if len(fam_z_vals) == 0:
        continue
    bar_heights[(SOTA_PLOT_NAME, target)] = float(z_to_r(fam_z_vals.mean()))
    best = df_dnn[df_dnn["y"] == target].sort_values("score_r", ascending=False).iloc[0]
    print(f"[{target}]  DNN mean-of-families r={bar_heights[(SOTA_PLOT_NAME, target)]:.4f}  "
          f"best model={best['model_name']} ({best['model_type']}, r={best['score_r']:.4f})")
for _, row in pose_df.iterrows():
    bar_heights[(row["x"], row["y"])] = float(row["score_r"])

# -- Save plotted data --
data_dir = "./results/plot_data/supp_fig_3"
os.makedirs(data_dir, exist_ok=True)
pd.DataFrame(
    [{"predictor": p, "target": t, "mean_r": v} for (p, t), v in bar_heights.items()]
).to_csv(f"{data_dir}/bars.csv", index=False)
df_dnn.rename(columns={"y": "target"})[["model_name", "model_family", "model_type", "target", "score_r"]].to_csv(
    f"{data_dir}/individual_dnn_dots.csv", index=False)
family_df.rename(columns={"y": "target"}).to_csv(f"{data_dir}/family_means.csv", index=False)
pose_df.rename(columns={"x": "predictor", "y": "target"}).to_csv(f"{data_dir}/pose_feature_dots.csv", index=False)
print(f"Saved plotted data → {data_dir}/")

# ── 6. Plot ────────────────────────────────────────────────────────────────────

color_dict = default_color_dict
n_target   = len(targets)
n_hues     = len(x_to_plot)
total_width = 0.7
bar_width   = total_width / n_hues
x_idx       = np.arange(n_target)

light_colors = {k: mcolors.to_rgba(v, alpha=0.25) for k, v in color_dict.items()}
mid_colors   = {k: mcolors.to_rgba(v, alpha=0.55) for k, v in color_dict.items()}
dark_colors  = {k: mcolors.to_rgba(v, alpha=0.90) for k, v in color_dict.items()}
video_fam_color = mcolors.to_rgba(color_dict.get("Video DNNs", "#9B59B6"), alpha=0.90)

fig, ax = plt.subplots(figsize=(7.0, 2.8), dpi=300)
ax.set_xlim(-0.5, n_target - 0.5)
ax.grid(axis="y", linestyle="-", linewidth=0.5)
ax.grid(axis="x", visible=False)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

bar_center = {}

# -- Bars --
for i, predictor in enumerate(x_to_plot):
    color     = mid_colors.get(predictor, mcolors.to_rgba(color_dict.get(predictor, "gray"), alpha=0.55))
    edgecolor = color_dict.get(predictor, "gray")
    means     = [bar_heights.get((predictor, t), np.nan) for t in targets]
    centers   = x_idx - total_width / 2 + (i + 0.5) * bar_width
    ax.bar(centers, means, width=bar_width, color=color, edgecolor=edgecolor)
    for t, cx in zip(targets, centers):
        bar_center[(predictor, t)] = float(cx)

# -- Individual DNN dots (lighter, jittered) --
rng = np.random.default_rng(42)
dnn_idx = x_to_plot.index(SOTA_PLOT_NAME)
for _, row in df_dnn.iterrows():
    if row["y"] not in targets:
        continue
    t_idx  = targets.index(row["y"])
    center = t_idx - total_width / 2 + (dnn_idx + 0.5) * bar_width
    jitter = rng.uniform(-bar_width * 0.35, bar_width * 0.35)
    ax.scatter(center + jitter, row["score_r"], s=2.3,
               color=light_colors.get(SOTA_PLOT_NAME, "gray"), zorder=2, linewidths=0)

# -- Family mean diamonds (darker), Video family highlighted --
family_label_added = False
video_family_label_added = False
for _, row in family_df.iterrows():
    if row["y"] not in targets:
        continue
    t_idx  = targets.index(row["y"])
    center = t_idx - total_width / 2 + (dnn_idx + 0.5) * bar_width
    is_video = row["family"] == VIDEO_FAMILY
    if is_video:
        label = "Video DNN family mean" if not video_family_label_added else None
        ax.scatter(center, row["score_r"], s=14.5, marker="D",
                   color=video_fam_color, edgecolors="black", linewidths=0.4,
                   zorder=4, label=label)
        video_family_label_added = True
    else:
        label = "DNN family mean" if not family_label_added else None
        ax.scatter(center, row["score_r"], s=8.7, marker="D",
                   color=dark_colors.get(SOTA_PLOT_NAME, "gray"), edgecolors="white", linewidths=0.4,
                   zorder=4, label=label)
        family_label_added = True

# -- 3D body joints dot --
pose_idx = x_to_plot.index(POSE_FEATURE)
for _, row in pose_df.iterrows():
    t_idx  = targets.index(row["y"])
    center = t_idx - total_width / 2 + (pose_idx + 0.5) * bar_width
    ax.scatter(center, row["score_r"], s=10,
               color=dark_colors.get(POSE_FEATURE, "gray"),
               edgecolors="white", linewidths=0.4, zorder=4)

# -- Axis labels --
wrapped_labels = [textwrap.fill(change_name(t), width=12, break_long_words=False) for t in targets]
ax.set_xticks(x_idx)
ax.set_xticklabels(wrapped_labels, ha="center", fontsize=7)
ax.set_ylabel("Score ($r$)", fontsize=7, weight="bold")
ax.set_ylim(-0.25, 1.0)
ax.tick_params(axis="y", labelsize=7)

# -- Legend --
legend_elements = [
    mpatches.Patch(facecolor=mid_colors.get(SOTA_PLOT_NAME, "gray"),
                   edgecolor=color_dict.get(SOTA_PLOT_NAME, "gray"),
                   label="DNN (mean)"),
    plt.Line2D([0], [0], marker='o', linestyle='', color=light_colors.get(SOTA_PLOT_NAME, "gray"), markersize=3.2,
               markeredgewidth=0, label="DNN embeddings"),
    plt.Line2D([0], [0], marker='D', linestyle='', color=dark_colors.get(SOTA_PLOT_NAME, "gray"),
               markeredgecolor='white', markeredgewidth=0.4, markersize=3.8,
               label="DNN family mean"),
    plt.Line2D([0], [0], marker='D', linestyle='', color=video_fam_color,
               markeredgecolor='black', markeredgewidth=0.4, markersize=3.8,
               label="Video DNN family mean"),
]
legend_labels = [h.get_label() for h in legend_elements]
# The pose feature's key entry shows its bar with the dot on top (same value).
legend_elements.append((
    mpatches.Patch(facecolor=mid_colors.get(POSE_FEATURE, "gray"), edgecolor=color_dict.get(POSE_FEATURE, "gray")),
    plt.Line2D([0], [0], marker='o', linestyle='', color=dark_colors.get(POSE_FEATURE, "gray"),
               markeredgecolor='white', markeredgewidth=0.4, markersize=3.8),
))
legend_labels.append("3D body joints")
ax.legend(legend_elements, legend_labels, bbox_to_anchor=(1, 0.5), fancybox=True, fontsize=7)

# -- Significance brackets (3D body joints vs. DNN mean-of-families) --
def _add_sig_bracket(ax, x1, x2, y, h=0.01, text="*", lw=0.8, fontsize=7):
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], linewidth=lw, color="#555555")
    ax.text((x1 + x2) / 2, y + h, text, ha="center", va="bottom", fontsize=fontsize, color="#555555")


base_bracket_y = 1.0
bracket_tick   = 0.01
any_bracket    = False
for target in targets:
    stars = sig_info.get(target, {}).get("stars", "")
    if not stars:
        continue
    x1 = bar_center.get((SOTA_PLOT_NAME, target))
    x2 = bar_center.get((POSE_FEATURE, target))
    if x1 is None or x2 is None:
        continue
    _add_sig_bracket(ax, x1, x2, y=base_bracket_y, h=bracket_tick, text=stars)
    any_bracket = True

if any_bracket:
    lo, hi = ax.get_ylim()
    need_hi = base_bracket_y + 0.08
    if need_hi > hi:
        ax.set_ylim(lo, need_hi)

save_figure(fig, "./results/supp_fig_3.pdf", FIG_WIDTH_MM, relayout=fig.tight_layout)
