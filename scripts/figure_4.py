"""
Figure 4: DNN trends – behavioral rating score vs. pose encoding score.

Layout : rows = [3D social pose features, 2D social pose features]
         cols = 5 rating targets
Each cell: scatter of individual DNN checkpoints (light dots) + OLS trend line
           through family means (dark diamonds).

Statistical test: permutation test on family-level means (Fisher-z space),
    one-tailed (H₀: effect ≤ 0), BH-FDR corrected over 5 targets per family.
    n_perm = 10,000 with (1 + k) / (n_perm + 1) correction.

    Family 1 – 3D social pose → human rating  [permute pose_z_mean, test r]
    Family 2 – 2D social pose → human rating  [permute pose_z_mean, test r]
    Family 3 – 3D better than 2D              [family-level label-swap, test Δz = z(r(3D,human)) − z(r(2D,human))]
               Monte Carlo sampling of n_perm=10,000 random swap patterns
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import colors as mcolors
from matplotlib.ticker import FixedLocator
from scipy.stats import linregress, pearsonr
from statsmodels.stats.multitest import multipletests

from src.encoding import get_top_sota_scores
from src.plottings import default_color_dict, change_name
from src.stats import stars_from_p


# ── Configuration ─────────────────────────────────────────────────────────────

targets = [
    'spatial expanse', 'interagent distance', 'agents facing',
    'communication', 'joint action',
]
pose_features = ['3D social pose features', '2D social pose features']

INDIV_ALPHA = 0.20   # individual DNN dot opacity
FAM_ALPHA   = 0.85   # family mean diamond opacity
DOT_COLOR   = "#23a5f0"


def _wrap_pose_label(s):
    """Break pose-feature labels after 'and'."""
    return s.replace(" and ", " and\n", 1)


def r_to_z(r):
    return np.arctanh(np.clip(np.asarray(r, float), -1 + 1e-7, 1 - 1e-7))

def z_to_r(z):
    return np.tanh(z)

def _fmt_p(p):
    return "p < 0.001" if p < 0.001 else f"p = {p:.3f}"


# ── 1. Load data ──────────────────────────────────────────────────────────────

sota_df = get_top_sota_scores(
    targets=targets, pose_features=pose_features, top_n=1, collect="both",
)

grouped = pd.read_csv("data/processed/grouped_models.csv")[["Model UID", "model_family"]]
sota_df = sota_df.merge(grouped, left_on="model_name", right_on="Model UID", how="left")
sota_df = sota_df[
    sota_df["model_family"].notna() & (sota_df["model_family"] != "Other")
].copy()

sota_df["target_z"] = r_to_z(sota_df["target_r"].values)
sota_df["pose_z"]   = r_to_z(sota_df["pose_r"].values)

# Report family composition
families   = sorted(sota_df["model_family"].unique())
n_families = len(families)
print(f"\n{'='*60}")
print(f"Model families (F = {n_families}):")
for fam in families:
    n = sota_df[sota_df["model_family"] == fam]["model_name"].nunique()
    print(f"  {fam:<30s}  n = {n}")
print(f"{'='*60}\n")


# ── 2. Family means (Fisher z averaged, back to r for plotting) ───────────────

fam_rows = []
for pf in pose_features:
    for tgt in targets:
        g = sota_df[(sota_df["pose_feature"] == pf) & (sota_df["target"] == tgt)]
        for fam in g["model_family"].unique():
            gf = g[g["model_family"] == fam]
            fam_rows.append({
                "pose_feature":  pf,
                "target":        tgt,
                "model_family":  fam,
                "target_r_mean": float(z_to_r(r_to_z(gf["target_r"].values).mean())),
                "pose_r_mean":   float(z_to_r(r_to_z(gf["pose_r"].values).mean())),
            })
fam_df = pd.DataFrame(fam_rows)


# ── 3. Family-mean permutation tests + BH-FDR ────────────────────────────────

N_PERM = 10_000

def pearson_perm_test(x, y, n_perm=N_PERM, seed=0):
    """
    Two-tailed permutation test for Pearson r (H₀: r = 0), statistic in Fisher-z space.
    Permutes x, recomputes z(r). p = (1 + #{|perm_z| ≥ |obs_z|}) / (n_perm + 1).
    """
    rng    = np.random.default_rng(seed)
    x, y   = np.asarray(x, float), np.asarray(y, float)
    obs_z  = float(r_to_z(pearsonr(x, y)[0]))
    perm_z = np.array([r_to_z(pearsonr(rng.permutation(x), y)[0]) for _ in range(n_perm)])
    p_two  = (1.0 + float((np.abs(perm_z) >= abs(obs_z)).sum())) / (n_perm + 1)
    return obs_z, p_two


def perm_corr_diff_swap(wide, n_perm=N_PERM, seed=0):
    """
    One-tailed Monte Carlo family-level label-swap test: H₀: z(r(3D,human)) ≤ z(r(2D,human)).
    Samples n_perm random swap patterns; each pattern independently flips the
    3D/2D family means for a subset of families.
    Fully vectorised: all patterns evaluated in a single batch pearsonr computation.
    p = (1 + #{d_perm ≥ d_obs}) / (n_perm + 1).
    """
    p3         = wide["pose_z_3d"].values
    p2         = wide["pose_z_2d"].values
    tgt        = wide["target_z"].values
    _, fam_idx = np.unique(wide["model_family"].values, return_inverse=True)
    counts     = np.bincount(fam_idx)
    tgt_m      = np.bincount(fam_idx, weights=tgt) / counts   # (F,)
    p3_m       = np.bincount(fam_idx, weights=p3)  / counts   # (F,)
    p2_m       = np.bincount(fam_idx, weights=p2)  / counts   # (F,)

    n_fam = len(tgt_m)
    rng   = np.random.default_rng(seed)
    swaps = rng.integers(0, 2, size=(n_perm, n_fam)).astype(bool)  # (n_perm, F)

    v3 = np.where(swaps, p2_m[None, :], p3_m[None, :])  # (n_perm, F)
    v2 = np.where(swaps, p3_m[None, :], p2_m[None, :])  # (n_perm, F)

    def _batch_pearsonr(X, y):
        Xc = X - X.mean(axis=1, keepdims=True)
        yc = y - y.mean()
        num   = (Xc * yc).sum(axis=1)
        denom = np.sqrt((Xc ** 2).sum(axis=1)) * np.sqrt((yc ** 2).sum())
        return num / denom

    z3_perm = r_to_z(_batch_pearsonr(v3, tgt_m))
    z2_perm = r_to_z(_batch_pearsonr(v2, tgt_m))
    d_perm  = z3_perm - z2_perm

    z3_obs = float(r_to_z(pearsonr(p3_m, tgt_m)[0]))
    z2_obs = float(r_to_z(pearsonr(p2_m, tgt_m)[0]))
    d_obs  = z3_obs - z2_obs
    p_one  = float((1.0 + (d_perm >= d_obs).sum()) / (n_perm + 1))
    delta_r_obs = float(z_to_r(z3_obs)) - float(z_to_r(z2_obs))
    return d_obs, p_one, delta_r_obs


sig_info = {}   # (pf_id, tgt) → {r, p_raw, p_corrected, stars}

for pf_id, pf in [(1, "3D social pose features"), (2, "2D social pose features")]:
    ps   = []
    rows = []

    for tgt in targets:
        g = sota_df[
            (sota_df["pose_feature"] == pf) &
            (sota_df["target"] == tgt)
        ].copy()

        fam_mean = (
            g.groupby("model_family")
             .agg(target_z_mean=("target_z", "mean"), pose_z_mean=("pose_z", "mean"))
             .reset_index()
        )

        z_fam, p_two = pearson_perm_test(
            fam_mean["target_z_mean"], fam_mean["pose_z_mean"]
        )

        ps.append(p_two)
        rows.append((tgt, z_fam, p_two, len(fam_mean)))

    _, p_bh, _, _ = multipletests(ps, alpha=0.05, method="fdr_bh")

    print(f"\nFamily-mean permutation test: {pf}")
    for (tgt, z_fam, p_two, n_fam), p_corr in zip(rows, p_bh):
        s = stars_from_p(float(p_corr))
        sig_info[(pf_id, tgt)] = {
            "z":           z_fam,
            "r":           float(z_to_r(z_fam)),
            "p_raw":       p_two,
            "p_corrected": float(p_corr),
            "stars":       s,
        }
        print(
            f"  {tgt:<25}  F={n_fam:>2d}  "
            f"r={float(z_to_r(z_fam)):.3f}  p_perm={p_two:.4g}  p_BH={p_corr:.4g}  {s}"
        )


# ── 4. 3D > 2D family-mean OLS permutation + BH-FDR ─────────────────────────

diff_sig_info = {}
ps_diff   = []
diff_rows = []

for tgt in targets:
    df3 = sota_df[
        (sota_df["pose_feature"] == "3D social pose features") &
        (sota_df["target"] == tgt)
    ][["model_name", "model_family", "target_z", "pose_z"]].rename(
        columns={"pose_z": "pose_z_3d"}
    )
    df2 = sota_df[
        (sota_df["pose_feature"] == "2D social pose features") &
        (sota_df["target"] == tgt)
    ][["model_name", "pose_z"]].rename(columns={"pose_z": "pose_z_2d"})

    wide = df3.merge(df2, on="model_name", how="inner")

    fam_wide = (
        wide.groupby("model_family")
             .agg(
                 target_z_mean=("target_z", "mean"),
                 pose_z_3d_mean=("pose_z_3d", "mean"),
                 pose_z_2d_mean=("pose_z_2d", "mean"),
             )
             .reset_index()
    )

    d_obs, p_one, delta_r_obs = perm_corr_diff_swap(wide)

    ps_diff.append(p_one)
    diff_rows.append((tgt, d_obs, delta_r_obs, p_one, len(fam_wide)))

_, p_bh_diff, _, _ = multipletests(ps_diff, alpha=0.05, method="fdr_bh")

print(f"\nFamily-mean permutation test: 3D > 2D  (family-level label-swap, n_perm={N_PERM}, Δr = r3D − r2D)")
for (tgt, d_obs, delta_r_obs, p_one, n_fam), p_corr in zip(diff_rows, p_bh_diff):
    s = stars_from_p(float(p_corr))
    diff_sig_info[tgt] = {
        "d_obs":       d_obs,
        "delta_r":     delta_r_obs,
        "p_raw":       p_one,
        "p_corrected": float(p_corr),
        "stars":       s,
    }
    print(
        f"  {tgt:<25}  F={n_fam:>2d}  "
        f"Δr={delta_r_obs:.3f}  p_perm={p_one:.4g}  p_BH={p_corr:.4g}  {s}"
    )
print()


# ── 5. Plot ───────────────────────────────────────────────────────────────────

xlim    = (-0.2, 1.0)
ylim    = (-0.2, 1.0)
x_ticks = np.array([-0.1, 0.1, 0.3, 0.5, 0.7, 0.9])
y_ticks = np.array([-0.1, 0.1, 0.3, 0.5, 0.7, 0.9])

n_rows = len(pose_features)
n_cols = len(targets)
fig_w  = 3.0 * n_cols + 3.0   # +3 for outside legend
fig_h  = 3.3 * n_rows + 1.0

fig, axes = plt.subplots(
    n_rows, n_cols,
    figsize=(fig_w, fig_h),
    sharex=True, sharey=True,
    dpi=300,
)

fs_label = 17
fs_tick  = 13
fs_anno  = 15

indiv_color = mcolors.to_rgba(DOT_COLOR, alpha=INDIV_ALPHA)
fam_color   = mcolors.to_rgba(DOT_COLOR, alpha=FAM_ALPHA)

for r, pf in enumerate(pose_features):
    fam_id = r + 1   # 1 = 3D row, 2 = 2D row

    for c, tgt in enumerate(targets):
        ax = axes[r, c]

        df   = sota_df[(sota_df["pose_feature"] == pf) & (sota_df["target"] == tgt)]
        fsub = fam_df[(fam_df["pose_feature"] == pf) & (fam_df["target"] == tgt)]

        # Individual DNN dots (light): x = human rating, y = pose encoding
        ax.scatter(
            df["target_r"], df["pose_r"],
            s=18, color=indiv_color, zorder=2, linewidths=0,
        )

        # OLS line through family means
        if len(fsub) >= 3:
            sl, ic, _, _, _ = linregress(
                fsub["target_r_mean"].values, fsub["pose_r_mean"].values
            )
            x_fit = np.linspace(fsub["target_r_mean"].min() - 0.1, fsub["target_r_mean"].max() + 0.1, 200)
            ax.plot(x_fit, sl * x_fit + ic, color=DOT_COLOR, linewidth=1.5, zorder=6)

        # Family mean diamonds: x = human rating, y = pose encoding
        ax.scatter(
            fsub["target_r_mean"], fsub["pose_r_mean"],
            s=55, marker="D", color=fam_color,
            edgecolors="white", linewidths=0.6, zorder=5,
        )

        # Annotation: family-mean r (back-transformed from Fisher-z) + BH-corrected p
        info  = sig_info.get((fam_id, tgt), {})
        r_val = info.get("r", float("nan"))
        p_c   = info.get("p_corrected", float("nan"))
        stars = info.get("stars", "")
        p_str = "< 0.001" if p_c < 0.001 else f"= {p_c:.3f}"
        anno  = f"r = {r_val:.2f}\npBH {p_str}"
        if stars:
            anno += f"  {stars}"
        ax.text(
            0.04, 0.97, anno,
            transform=ax.transAxes, ha='left', va='top',
            fontsize=fs_anno, color='#000000',
        )

        # Axes
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.grid(True, linestyle=':', linewidth=0.8, alpha=0.5)
        ax.xaxis.set_major_locator(FixedLocator(x_ticks))
        ax.yaxis.set_major_locator(FixedLocator(y_ticks))

        show_x = (r == n_rows - 1)
        show_y = (c == 0)
        ax.tick_params(axis='x', labelbottom=show_x, labelsize=fs_tick)
        ax.tick_params(axis='y', left=show_y,
                       labelsize=fs_tick if show_y else 1, labelleft=show_y)

        if show_x:
            ax.set_xlabel(
                change_name(tgt) + "\nscores (r)",
                fontsize=fs_label,
            )
        if show_y:
            ax.set_ylabel(
                _wrap_pose_label(change_name(pf)) + "\nscores (r)",
                fontsize=fs_label,
            )

        # Gray subplot borders
        for spine in ax.spines.values():
            spine.set_edgecolor('gray')
            spine.set_linewidth(0.8)

# ── Legend ────────────────────────────────────────────────────────────────────

legend_elements = [
    plt.Line2D(
        [0], [0], marker='o', linestyle='',
        color=indiv_color, markersize=6, label="Vision DNN embeddings",
    ),
    plt.Line2D(
        [0], [0], marker='D', linestyle='',
        color=fam_color, markeredgecolor='white', markeredgewidth=0.6,
        markersize=8, label="DNN family mean",
    ),
]
fig.legend(
    handles=legend_elements,
    loc='center left', bbox_to_anchor=(1.01, 0.5),
    fontsize=fs_label, framealpha=0.9,
)

plt.tight_layout()
out_dir = "./results/fig4.png"
os.makedirs(os.path.dirname(out_dir), exist_ok=True)
plt.savefig(out_dir, dpi=300, bbox_inches='tight')
plt.show()
print(f"Saved → {out_dir}")
