import os
import re
import textwrap
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import TheilSenRegressor
from scipy.stats import pearsonr, zscore

from src.test_functions import df_descriptive
from src.config import SOTA_PLOT_NAME

from itertools import combinations
from matplotlib import colors as mcolors
from src.stats import (significance_testing, stars_from_p, _normalize_sig_pairs_dicts,
                        _perm_corr_pvalue, _perm_corr_diff_swap)


default_features =['spatial expanse', 'object directed','interagent distance',
                   'agents facing', 'communication',  'joint action']


### color

def expand_color_dict(base_colors):
    """Expand base color dict with '+' combinations."""
    new_colors = dict(base_colors)
    keys = list(base_colors.keys())
    for i in range(len(keys)):
        for j in range(i+1, len(keys)):
            combo = f"{keys[i]}+{keys[j]}"
            """Average two hex colors and return a new hex string."""
            rgb1 = tuple(int(base_colors[keys[i]][i:i+2], 16) for i in (1, 3, 5))
            rgb2 = tuple(int(base_colors[keys[j]][i:i+2], 16) for i in (1, 3, 5))
            avg  = tuple((a+b)//2 for a, b in zip(rgb1, rgb2))
            new_colors[combo] = '#{:02X}{:02X}{:02X}'.format(*avg)
    return new_colors

# Example usage:
base_color_dict = {
    'SocialGNN':       '#F35A62',
    'RNN Gaze':        '#F18282',
    '3D social pose features':"#fc1212",
    '2D social pose features':"#fa7946",
    # '3D relational':   '#fc7312',
    # '2D relational':   '#fa9046',
    '3D face arrows':  "#f44545",
    '2D face arrows':  "#fc7209",
    # 'RNN Entity':      '#1E90FF',
    '3D body joints':       "#56AA29",
    '2D body joints':       "#A8D070",
    # '3D vertices':     "#4D80E7",
    # '2D vertices':     "#76B2EE",
    'joints partial out \n3D social pose features': "#40342b",
    'joints partial out \n2D social pose features': "#7c6451",
    'joints partial out \n3D directions': "#326C6C",
    'joints partial out \n3D positions': "#526031",
    # 'Vision DNN (6 PCA)':        '#555555',
    'Vision DNNs':               "#2C95EBFF",
    SOTA_PLOT_NAME:              "#2C95EBFF",
    'Image DNNs':                "#2C95EBFF",
    'Video DNNs':                "#9B59B6FF",
    f'{SOTA_PLOT_NAME} + \n3D social pose features': "#C12A22",
    # 'Top 20 Vision DNNs (6 PCA)':'#555555',
    # 'Top 20 Vision DNNs':        '#888888',
    '4DHumans':                   '#25A2AE',
    '3D pose model embeddings':     '#25A2AE',

}

default_color_dict = expand_color_dict(base_color_dict)



def _fmt_p(p):
    """Format a p-value as 'p < 0.001' or 'p = 0.xxx'."""
    return "p < 0.001" if p < 0.001 else f"p = {p:.3f}"


def _pvalue_for_pair(results_df, y, class_A, class_B, testing_type, tails):
    return significance_testing(
        results_df=results_df,
        ys=[y],
        class_A=class_A,
        class_B=class_B,
        testing_type=testing_type,
        tails=tails
    )[0]

def _add_sig_bracket(ax, x1, x2, y, h=0.01, text='*', lw=1.5, fontsize=12):
    ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], linewidth=lw, color='#555555')
    ax.text((x1 + x2) / 2, y + h, text, ha='center', va='bottom', fontsize=fontsize, color='#555555')


def change_name(x):
    if x == '3D joints':
        return '3D joint positions'
    elif x == '3D social pose features':
        return '3D positions and facing directions'
    elif x == '2D social pose features':
        return '2D positions and facing directions'
    elif x == '4DHumans':
        return '3D pose model embeddings'
    elif x == 'communication':
        return 'communicative interaction'
    elif x == 'joint action':
        return 'physical interaction'
    elif x == '3D body joints (mean+diff+std)':
        return '3D body joints\n(position/motion/variability)'
    else:
        return x


# # def encoding_bar_with_points(
# #     results_df: pd.DataFrame,
# #     x_to_plot,
# #     y_to_plot,
# #     color_dict=None,
# #     out_dir=None,
# #     y_range=(-0.25, 1),
# #     size=(10, 5), dpi=300,
# #     # --- significance options ---
# #     show_significance=False,
# #     sig_pairs=None,
# #     sig_test_results=None,
# #     alpha_thresholds=(0.05, 0.01, 0.001),
# #     bracket_gap=0.07,
# #     bracket_tick=0.01,
# #     base_bracket_y=1.0,
# #     # --- new: star a specific model's raw points ---
# #     star_model: str | None = None,
# #     fs_tick: int = 16,
# #     # --- subplot support: pass an existing Axes to draw into ---
# #     ax=None,
# # ):
# #     """
# #     Plot mean bars with raw points and annotate significant differences between model classes per target.
# #     Adds a per-target Split-half reliability horizontal span (thick light gray band) and stacks significance brackets.

# #     sig_test_results : dict, optional
# #         Pre-computed test results from run_significance_tests() (recommended — includes FDR correction).
# #         Keys are (pair_idx, target_name); values have 'stars' and 'pair'.
# #         If None and show_significance=True, tests are run inline WITHOUT FDR correction (emits a warning).
# #     star_model : if provided, any raw-point rows with results_df['model_name'] == star_model
# #         are plotted as gold stars. A legend entry for this model is added automatically.
# #     """

# #     from matplotlib import colors as mcolors
# #     import matplotlib.patches as mpatches
# #     import numpy as np
# #     import matplotlib.pyplot as plt

# #     # ---------- Split-half reliability values ----------
# #     noise_ceiling = {
# #         "spatial expanse": 0.7194827819823470,
# #         "object directed": 0.928456388129668,
# #         "interagent distance": 0.8854506352255450,
# #         "agents facing": 0.9574809908008190,
# #         "joint action": 0.7688673458027810,
# #         "communication": 0.7621707694002240,
# #     }
# #     missing_nc = [t for t in y_to_plot if t not in noise_ceiling]

# #     plot_split_half = True
# #     if len(missing_nc) == len(y_to_plot):
# #         plot_split_half = False

# #     # 1) Mean & Max per (x,y)
# #     stats_df = results_df.groupby(["x", "y"], as_index=False)["score_r"].mean()
# #     print("-" * 40)
# #     for _, row in stats_df.iterrows():
# #         print(row.to_dict())
# #     print("-" * 40)

# #     # 2) Validate requested x/y
# #     available_x = stats_df["x"].unique().tolist()
# #     available_y = stats_df["y"].unique().tolist()
# #     missing_x = [x for x in x_to_plot if x not in available_x]
# #     missing_y = [y for y in y_to_plot if y not in available_y]
# #     if missing_x or missing_y:
# #         msg = "Missing values in data:\n"
# #         if missing_x: msg += f"  x_to_plot not found: {missing_x}\n"
# #         if missing_y: msg += f"  y_to_plot not found: {missing_y}\n"
# #         raise ValueError(msg)

# #     n_target = len(y_to_plot)
# #     n_hues = len(x_to_plot)
# #     if color_dict is None:
# #         color_dict = default_color_dict

# #     total_width = 0.7
# #     bar_width = total_width / n_hues
# #     x_idx = np.arange(n_target)

# #     # colors
# #     light_colors = {k: mcolors.to_rgba(v, alpha=0.55) for k, v in color_dict.items()}
# #     dark_colors  = {k: mcolors.to_rgba(v, alpha=0.90) for k, v in color_dict.items()}

# #     # --- NEW: precompute star-model availability among rows that will actually be plotted
# #     use_stars = False
# #     if star_model is not None:
# #         if "model_name" not in results_df.columns:
# #             print(f"[warning] 'model_name' column not found; cannot star '{star_model}'.")
# #         else:
# #             mask_draw = (
# #                 results_df["x"].isin(x_to_plot) &
# #                 results_df["y"].isin(y_to_plot)
# #             )
# #             star_rows = results_df.loc[mask_draw & (results_df["model_name"] == star_model)]
# #             n_unique = star_rows["model_name"].nunique()
# #             if n_unique == 1:
# #                 use_stars = True
# #             elif n_unique == 0:
# #                 print(f"[warning] star_model '{star_model}' not found among plotted rows; no stars will be drawn.")
# #             else:
# #                 print(f"[warning] multiple models matched star_model '{star_model}'; no stars will be drawn.")
# #                 use_stars = False

# #     # 3) Figure
# #     if ax is None:
# #         fig, ax = plt.subplots(figsize=size, dpi=dpi)
# #         _standalone = True
# #     else:
# #         fig = ax.get_figure()
# #         _standalone = False
# #     ax.grid(axis='x', visible=False)
# #     ax.grid(axis='y', linestyle='-', linewidth=1.0)
# #     ax.spines["top"].set_visible(False)
# #     ax.spines["right"].set_visible(False)

# #     # Track bar geometry
# #     bar_pos, bar_val = {}, {}

# #     # 4) Bars
# #     for i, predictor in enumerate(x_to_plot):
# #         base = color_dict.get(predictor, "gray")
# #         color = light_colors.get(predictor, mcolors.to_rgba(base, alpha=0.55))
# #         edgecolor = color_dict.get(predictor, "gray")

# #         means = [
# #             stats_df.loc[(stats_df.y == target) & (stats_df.x == predictor), "score_r"].values[0]
# #             for target in y_to_plot
# #         ]
# #         centers = x_idx - total_width / 2 + (i + 0.5) * bar_width
# #         ax.bar(centers, means, width=bar_width, color=color, edgecolor=edgecolor, label=predictor)

# #         for target, cx, m in zip(y_to_plot, centers, means):
# #             bar_pos[(predictor, target)]  = float(cx)
# #             bar_val[(predictor, target)]  = float(m)

# #     desired_order = x_to_plot

# #     # 5) Raw points (with optional gold stars)
# #     rng = np.random.default_rng(seed=42)
# #     gold_color = "#2C95EB9C"
# #     star_plotted = False

# #     for _, row in results_df.iterrows():
# #         if row.x in x_to_plot and row.y in y_to_plot:
# #             feat_idx = y_to_plot.index(row.y)
# #             model_idx = x_to_plot.index(row.x)
# #             center = feat_idx - total_width / 2 + (model_idx + 0.5) * bar_width
# #             jitter = rng.uniform(-bar_width * 0.1, bar_width * 0.1)

# #             is_star = (
# #                 use_stars
# #                 and ("model_name" in results_df.columns)
# #                 and (row.get("model_name", None) == star_model)
# #             )

# #             if is_star:
# #                 ax.scatter(
# #                     center + jitter, row.score_r,
# #                     s=80, marker="o", zorder=4,
# #                     color=gold_color,
# #                     edgecolors="black", linewidths=0.9,
# #                     label=change_name(star_model) if not star_plotted else None
# #                 )
# #                 star_plotted = True
# #             else:
# #                 ax.scatter(
# #                     center + jitter, row.score_r, s=20,
# #                     color=dark_colors.get(row.x, "gray"),
# #                     linewidths=0.5
# #                 )

# #     # === 5.5) Split-half reliability bars ===
# #     nc_band_height = 0.024
# #     if plot_split_half:
# #         nc_face = "#999999D0"
# #         for ti, target in enumerate(y_to_plot):
# #             nc_val = noise_ceiling[target]
# #             x_left  = (ti - total_width / 2)
# #             x_right = (ti + total_width / 2)
# #             rect = mpatches.Rectangle(
# #                 (x_left, nc_val - nc_band_height / 2),
# #                 width=(x_right - x_left) * 1.05,
# #                 height=nc_band_height,
# #                 facecolor=nc_face,
# #                 edgecolor='none'
# #             )
# #             ax.add_patch(rect)
# #         nc_patch = mpatches.Patch(facecolor=nc_face, edgecolor='none', label="Split-half reliability")

# #     # 6) Labels & legend
# #     wrapped_labels = []
# #     for target in y_to_plot:
# #         wrapped = textwrap.fill(change_name(target), width=10, break_long_words=False)
# #         wrapped_labels.append(wrapped)

# #     ax.set_xticks(x_idx)
# #     ax.set_xticklabels(wrapped_labels, ha="center", fontsize=fs_tick)
# #     ax.set_ylabel("Score ($r$)", fontsize=fs_tick, weight="bold")
# #     ax.set_ylim(y_range[0], y_range[1])
# #     ax.tick_params(axis="y", labelsize=fs_tick)

# #     handles, labels = ax.get_legend_handles_labels()
# #     ordered_handles = [handles[labels.index(name)] for name in desired_order if name in labels]
# #     ordered_labels = [change_name(name) for name in desired_order if name in labels]

# #     # add split-half reliability at the end (for now)
# #     if plot_split_half:
# #         ordered_handles.append(nc_patch)
# #         ordered_labels.append("Split-half reliability")

# #     if star_plotted:
# #         star_patch = mpatches.Patch(facecolor=gold_color, edgecolor="black", label=star_model)
# #         # insert as second item
# #         ordered_handles.insert(1, star_patch)
# #         ordered_labels.insert(1, star_model)

# #     ax.legend(ordered_handles, ordered_labels, bbox_to_anchor=(1, 0.5), fancybox=True)


# #     # 7) Significance brackets
# #     max_bracket_top = y_range[1]
# #     if show_significance and len(x_to_plot) >= 2:
# #         pair_specs = _normalize_sig_pairs_dicts(sig_pairs)

# #         if sig_test_results is None:
# #             import warnings
# #             warnings.warn(
# #                 "show_significance=True but sig_test_results is None — running tests inline "
# #                 "WITHOUT FDR correction. Call run_significance_tests() first and pass the result "
# #                 "as sig_test_results= to apply BH-FDR correction.",
# #                 UserWarning,
# #                 stacklevel=2,
# #             )

# #         stack_count = {t: 0 for t in y_to_plot}
# #         for y in y_to_plot:
# #             for pair_idx, spec in enumerate(pair_specs):
# #                 a, b = spec["a"], spec["b"]
# #                 if (a, y) not in bar_pos or (b, y) not in bar_pos:
# #                     continue

# #                 if sig_test_results is not None:
# #                     entry = sig_test_results.get((pair_idx, y))
# #                     stars = entry["stars"] if entry is not None else ""
# #                 else:
# #                     p = _pvalue_for_pair(
# #                         results_df, y,
# #                         class_A=a, class_B=b,
# #                         testing_type=spec["testing_type"],
# #                         tails=spec["tails"],
# #                     )
# #                     stars = stars_from_p(p, thresholds=alpha_thresholds)

# #                 if not stars:
# #                     continue

# #                 x1 = bar_pos[(a, y)]
# #                 x2 = bar_pos[(b, y)]
# #                 y0 = base_bracket_y + bracket_gap * (stack_count[y])
# #                 _add_sig_bracket(ax, x1, x2, y=y0, h=bracket_tick, text=stars, lw=1.5, fontsize=10)
# #                 stack_count[y] += 1
# #                 max_bracket_top = max(max_bracket_top, y0 + bracket_tick)

# #         ylim_lo, ylim_hi = ax.get_ylim()
# #         # max_nc = max(noise_ceiling[t] for t in y_to_plot) + nc_band_height / 2
# #         need_hi = max(max_bracket_top + 0.02, 1.0, y_range[1])
# #         if need_hi > ylim_hi:
# #             ax.set_ylim(ylim_lo, need_hi)
# #     else:
# #         ylim_lo, ylim_hi = ax.get_ylim()
# #         # max_nc = max(noise_ceiling[t] for t in y_to_plot) + nc_band_height / 2
# #         # if max_nc + min_headroom > ylim_hi:
# #         #     ax.set_ylim(ylim_lo, max(max_nc + min_headroom, y_range[1]))

# #     if _standalone:
# #         plt.tight_layout()
# #         if out_dir:
# #             plt.savefig(out_dir, bbox_inches="tight")
# #         plt.show()



# ### SOTA PLOTTING

# def _parse_layer_idx(name: str) -> int:
#     """
#     Robustly parse an integer index from a layer_name like '12-block1' or '003-something'.
#     Returns np.nan if no leading integer is found.
#     """
#     if not isinstance(name, str):
#         return np.nan
#     m = re.match(r"\s*(\d+)", name)
#     return int(m.group(1)) if m else np.nan

# def plot_score_by_layer(results: pd.DataFrame,
#                         save_dir: str,
#                         title: str = "Score by Layer Depth",
#                         score_name: str = "Score",
#                         y_lim: tuple = None,
#                         plot_regression: bool = True,
#                         group_col: str = "y"):
#     """
#     For each unique group in `results[group_col]`, plot metric scores against layer depth.

#     Expected columns in `results`:
#         - 'layer_name' (str): contains a leading integer like '12-block1'
#         - group_col (str): e.g., 'y' or 'feature' (the grouping variable)
#         - 'score' (float)

#     The function extracts a numeric 'layer_idx' from 'layer_name' and uses it for sorting
#     and regression, while keeping 'layer_name' as tick labels.
#     """
#     os.makedirs(save_dir, exist_ok=True)

#     if group_col not in results.columns:
#         raise ValueError(f"`group_col='{group_col}'` not found in results columns: {list(results.columns)}")

#     if 'layer_name' not in results.columns or 'score_r' not in results.columns:
#         raise ValueError("`results` must contain 'layer_name' and 'score_r' columns.")

#     for feat in results[group_col].dropna().unique():
#         df_feat = results[results[group_col] == feat].copy()
#         if df_feat.empty:
#             continue

#         # Extract numeric layer index safely
#         df_feat['layer_idx'] = df_feat['layer_name'].apply(_parse_layer_idx)

#         # Keep only rows with valid numeric index and score_r
#         df_feat = df_feat[np.isfinite(df_feat['layer_idx']) & np.isfinite(df_feat['score_r'])].copy()
#         if df_feat.empty:
#             continue

#         # Sort by layer index
#         df_sorted = df_feat.sort_values('layer_idx')

#         # Arrays for plotting/regression
#         x_idx   = df_sorted['layer_idx'].to_numpy(dtype=float)   # numeric
#         x_labels = df_sorted['layer_name'].astype(str).to_numpy() # string labels
#         scores  = df_sorted['score_r'].to_numpy(dtype=float)

#         # Guard against degenerate arrays
#         if scores.size == 0:
#             continue

#         plt.figure(figsize=(12, 6))
#         ax = plt.gca()

#         # Plot points/lines using numeric x
#         ax.plot(x_idx, scores, 'o-', markersize=6, linewidth=1.5,
#                 alpha=0.9, label=str(feat))

#         # Trend line (use numeric x only; no isnan on strings)
#         if plot_regression and len(x_idx) >= 2:
#             mask = np.isfinite(x_idx) & np.isfinite(scores)
#             X = x_idx[mask].reshape(-1, 1)
#             y = scores[mask]
#             if len(y) >= 2 and np.ptp(X) > 0:  # ensure variance in X
#                 reg = TheilSenRegressor().fit(X, y)
#                 xr = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
#                 yr = reg.predict(xr)

#                 sign = '-' if reg.intercept_ < 0 else '+'
#                 eq = f"y = {reg.coef_[0]:.3f}x {sign} {abs(reg.intercept_):.3f}"
#                 ax.plot(xr.ravel(), yr, '--', linewidth=2, label=f"Trend\n({eq})")

#         # Labels & limits
#         ax.set_title(f"{feat}: {title}", fontsize=16, pad=20)
#         ax.set_xlabel("Layer Index", fontsize=14)
#         ax.set_ylabel(score_name, fontsize=14)

#         if y_lim is not None:
#             ax.set_ylim(*y_lim)
#         else:
#             ymin, ymax = np.nanmin(scores), np.nanmax(scores)
#             if np.isfinite(ymin) and np.isfinite(ymax):
#                 if np.isclose(ymin, ymax):
#                     # Flat line case: add a small buffer
#                     delta = max(1e-3, abs(ymin) * 0.05)
#                     ax.set_ylim(ymin - delta, ymax + delta)
#                 else:
#                     buffer = 0.1 * (ymax - ymin)
#                     ax.set_ylim(ymin - buffer, ymax + buffer)

#         # X ticks: show numeric positions, label with layer_name
#         ax.set_xticks(x_idx)
#         ax.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=10)

#         ax.axhline(0, color='gray', linestyle='--', linewidth=0.8)
#         ax.grid(True, alpha=0.3)
#         ax.legend(loc='best', fontsize=11)
#         plt.tight_layout()

#         # Safe filename
#         safe_feat = re.sub(r'[^-\w]+', '_', str(feat)).strip('_')
#         fname = f"{safe_feat}.png"
#         plt.savefig(os.path.join(save_dir, fname), dpi=300, bbox_inches='tight')
#         plt.close()




# def plot_sota_data(results_df, save_path=None):
#     # draw lmplot with one column per target, hue by pose_feature
#     g = sns.lmplot(
#         data=results_df,
#         x='pose_r',
#         y="target_r",
#         col="target",
#         hue="pose_feature",
#         col_wrap=3,
#         height=4,
#         aspect=1,
#         scatter_kws={"alpha": 0.1},
#         # ci=95,
#         markers="o",
#         palette="tab10",
#         facet_kws={"sharex": False, "sharey": False, "legend_out": True}
#     )

#     g.set_axis_labels('Pose Encoding Score', "Behavioral Encoding Score")
#     g.set_titles("Target: {col_name}")
#     g._legend.set_title("Pose Feature")

#     # annotate each facet with r for each pose_feature
#     pose_feats = results_df['pose_feature'].unique()
#     axes = g.axes.flatten()
#     for ax in axes:
#         title = ax.get_title()
#         target_name = title.split(": ")[1]

#         xlim = ax.get_xlim()
#         ylim = ax.get_ylim()
#         x_text = xlim[0] + 0.05 * (xlim[1] - xlim[0])
#         y_top = ylim[1] - 0.05 * (ylim[1] - ylim[0])

#         for i, feat in enumerate(pose_feats):
#             sub = results_df[
#                 (results_df['target'] == target_name) &
#                 (results_df['pose_feature'] == feat)
#             ]
#             if len(sub) >= 2:
#                 r_val, _ = pearsonr(sub['pose_r'], sub['target_r'])
#                 txt = f"{feat}: r={r_val:.2f}"
#                 ax.text(x_text,
#                         y_top - i * 0.05 * (ylim[1] - ylim[0]),
#                         txt,
#                         fontsize='small')

#     plt.tight_layout(rect=[0, 0, 0.9, 1])
#     if save_path:
#         plt.savefig(save_path, dpi=300, bbox_inches='tight')
#         plt.close()
#     else:
#         plt.show()



# import numpy as np
# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt
# from matplotlib.ticker import FixedLocator
# from scipy.stats import pearsonr


# def plot_data_grid(
#     sota_df: pd.DataFrame,
#     pose_features: list,
#     targets: list,
#     pose_model_df: pd.DataFrame | None = None,
#     save_path: str | None = None,
#     figsize_per_cell: tuple[float, float] = (3.6, 3.2),
#     xlim: tuple[float, float] = (-0.2, 1.0),
#     ylim: tuple[float, float] = (-0.2, 1.0),
#     show_corr_text: bool = True,
#     grid_sig_results: dict | None = None,
#     n_perm: int = 5000,
#     random_state: int = 0,
#     fs_label: int = 22,
#     fs_tick: int = 16,
#     fs_anno: int = 19,
# ):
#     """
#     Plot pose_r vs target_r in a grid: rows=pose_features, cols=targets.

#     Required
#     --------
#     pose_features : list[str]
#     targets       : list[str]

#     Data requirements
#     -----------------
#     sota_df must have columns: ['target', 'pose_feature', 'pose_r', 'target_r'].
#     pose_model_df (optional) has the same columns and is plotted as secondary points.

#     Visual spec
#     -----------
#     - Each row = one pose feature.
#     - Each column = one target.
#     - Only the LEFT of each row has the pose feature label (as the y-axis label).
#     - Only the BOTTOM of each column has the target label (as the x-axis label).
#     """

#     # ---- Validate inputs
#     required_cols = {'target', 'pose_feature', 'pose_r', 'target_r'}
#     for name, df in [('sota_df', sota_df), ('pose_model_df', pose_model_df)]:
#         if df is not None and not required_cols.issubset(df.columns):
#             missing = required_cols - set(df.columns)
#             raise ValueError(f"{name} missing required columns: {missing}")

#     if not pose_features or not targets:
#         raise ValueError("Both pose_features and targets must be non-empty lists.")

#     available_pose_feats = set(sota_df['pose_feature'].unique()) if not sota_df.empty else set()
#     if pose_model_df is not None and not pose_model_df.empty:
#         available_pose_feats |= set(pose_model_df['pose_feature'].unique())
#     missing_pose_feats = [p for p in pose_features if p not in available_pose_feats]
#     if missing_pose_feats:
#         raise ValueError(f"pose_features not found in data: {missing_pose_feats}")

#     available_targets = set(sota_df['target'].unique()) if not sota_df.empty else set()
#     if pose_model_df is not None and not pose_model_df.empty:
#         available_targets |= set(pose_model_df['target'].unique())
#     missing_targets = [t for t in targets if t not in available_targets]
#     if missing_targets:
#         raise ValueError(f"targets not found in data: {missing_targets}")

    
#     counts = sota_df["model_name"].value_counts()
#     assert counts.nunique() == 1, f"Unequal counts of model names in the score collection. The models should be matched for all target and pose pair:\n{counts}"

#     # ---- Colors
#     sota_color = "#23a5f0"
#     # secondary_color = "#07a245"

#     # ---- Fixed ticks
#     x_ticks = np.array([-0.1, 0.1, 0.3, 0.5, 0.7, 0.9], dtype=float)
#     y_ticks = np.array([-0.1, 0.1, 0.3, 0.5, 0.7, 0.9], dtype=float)

#     # ---- Figure setup
#     # ---- Resolve significance results ----
#     if show_corr_text and grid_sig_results is None:
#         import warnings
#         warnings.warn(
#             "show_corr_text=True but grid_sig_results is None — running tests inline "
#             "WITHOUT FDR correction. Call run_grid_significance_tests() first and pass "
#             "the result as grid_sig_results= to apply BH-FDR correction.",
#             UserWarning,
#             stacklevel=2,
#         )
#         from src.stats import run_grid_significance_tests
#         grid_sig_results = run_grid_significance_tests(
#             sota_df, targets, n_perm=n_perm, random_state=random_state
#         )

#     _corr_pvals = grid_sig_results["corr"] if grid_sig_results else {}
#     _diff_pvals = grid_sig_results["diff"] if grid_sig_results else {}

#     n_rows = len(pose_features)
#     n_cols = len(targets)
#     fig_w = max(2.0, figsize_per_cell[0] * n_cols + 1.0)
#     fig_h = max(2.0, figsize_per_cell[1] * n_rows + 0.8)

#     fig, axes = plt.subplots(
#         nrows=n_rows,
#         ncols=n_cols,
#         figsize=(fig_w, fig_h),
#         sharex=True, sharey=True
#     )

#     # Normalize axes to 2D array
#     if n_rows == 1 and n_cols == 1:
#         axes = np.array([[axes]])
#     elif n_rows == 1:
#         axes = axes[np.newaxis, :]
#     elif n_cols == 1:
#         axes = axes[:, np.newaxis]

#     # ---- Legend handles
#     # legend_handles = [
#     #     plt.Line2D([0], [0], marker='o', linestyle='',
#     #                markerfacecolor=sota_color, markeredgecolor='none',
#     #                markersize=7, label='Vision DNN\nembeddings')
#     # ]
#     # if pose_model_df is not None:
#     #     legend_handles.append(
#     #         plt.Line2D([0], [0], marker='o', linestyle='',
#     #                    markerfacecolor=secondary_color, markeredgecolor='none',
#     #                    markersize=7, label='Pose model embeddings')
#     #     )

#     # ---- Plot each cell
#     for c, tgt in enumerate(targets):

#         df_2d = sota_df[
#             (sota_df['pose_feature'] == '2D social pose features') &
#             (sota_df['target'] == tgt)
#         ].sort_values("model_name").reset_index(drop=True)

#         df_3d = sota_df[
#             (sota_df['pose_feature'] == '3D social pose features') &
#             (sota_df['target'] == tgt)
#         ].sort_values("model_name").reset_index(drop=True)


#         # Check uniqueness of model_name within each table
#         dup_2d = df_2d["model_name"].duplicated().any()
#         dup_3d = df_3d["model_name"].duplicated().any()

#         if dup_2d or dup_3d:
#             raise ValueError(
#                 f"[{tgt}] Duplicate model_name detected. "
#                 f"2D duplicates: {dup_2d}, 3D duplicates: {dup_3d}"
#             )

#         # (c) Check that model_name and layer_name match exactly (same order)
#         if not df_2d["model_name"].equals(df_3d["model_name"]):
#             diff = set(df_2d["model_name"]) ^ set(df_3d["model_name"])
#             raise ValueError(
#                 f"[{tgt}] model_name mismatch between 2D and 3D. "
#                 f"Symmetric difference: {sorted(diff)}"
#             )

#         if not df_2d["layer_name"].equals(df_3d["layer_name"]):
#             mism = df_2d[["model_name", "layer_name"]].merge(
#                 df_3d[["model_name", "layer_name"]],
#                 on="model_name",
#                 suffixes=("_2d", "_3d"),
#                 how="inner"
#             )
#             bad = mism[mism["layer_name_2d"] != mism["layer_name_3d"]]
#             raise ValueError(
#                 f"[{tgt}] layer_name mismatch for models:\n{bad}"
#             )
        
#         # Main regplot
#         for r, data_df in enumerate([df_3d, df_2d]):
#             pf = data_df['pose_feature'].iloc[0]
#             ax = axes[r, c]
#             sns.regplot(
#                 data=data_df,
#                 x='target_r', y='pose_r', ax=ax,
#                 scatter_kws={"alpha": 0.3, "s": 25, "color": sota_color},
#                 line_kws={"color": sota_color, "linewidth": 2},
#                 ci=None
#             )
#             if show_corr_text:
#                 if len(data_df) < 2:
#                     raise ValueError(
#                         f"[{tgt}] Not enough data points to plot (need at least 2 each). "
#                     )
#                 entry = _corr_pvals[(pf, tgt)]
#                 r_rounded = round(entry["r"], 2)
#                 if r_rounded == 0:
#                     r_rounded = 0.00
#                 ax.text(0.03, 0.95, f"r = {r_rounded:.2f}\n{_fmt_p(entry['p_corrected'])}",
#                         transform=ax.transAxes, ha='left', va='top',
#                         fontsize=fs_anno, color='#000000')

#             # Pose model scatter
#             # if pose_model_df is not None:
#             #     sec_sub = pose_model_df[
#             #         (pose_model_df['pose_feature'] == pf) &
#             #         (pose_model_df['target'] == tgt)
#             #     ]
#             #     if not sec_sub.empty:
#             #         ax.scatter(sec_sub['target_r'], sec_sub['pose_r'],
#             #                    s=25, alpha=0.6, color=secondary_color)

#             # Limits, grid, ticks
#             ax.set_xlim(*xlim)
#             ax.set_ylim(*ylim)
#             ax.grid(True, linestyle=':', linewidth=0.8, alpha=0.5)
#             ax.xaxis.set_major_locator(FixedLocator(x_ticks))
#             ax.yaxis.set_major_locator(FixedLocator(y_ticks))

#             show_x = (r == n_rows - 1)
#             show_y = (c == 0)

#             ax.tick_params(axis='x', bottom=show_x)
#             ax.tick_params(axis='y', left=show_y)

#             if show_x:
#                 ax.tick_params(axis='x', labelsize=fs_tick)
#             if show_y:
#                 ax.tick_params(axis='y', labelsize=fs_tick)

#             ax.set_xlabel("")
#             ax.set_ylabel("")

#             # Left column: pose feature labels
#             if c == 0:
#                 wrapped = textwrap.fill(change_name(str(pf))+ ' scores(r)', width=20)
#                 ax.set_ylabel(wrapped, fontsize=fs_label, labelpad=8)

#             # Bottom row: target labels
#             if r == n_rows - 1:
#                 wrapped = textwrap.fill(change_name(str(tgt))+ ' scores(r)', width=21)
#                 ax.set_xlabel(wrapped, fontsize=fs_label, labelpad=10)
#                 ax.xaxis.set_label_position('bottom')
        
#         if show_corr_text:
#             diff_entry = _diff_pvals[tgt]
#             print(f'[LOGGING] Target: {change_name(tgt)}, 3D vs 2D difference: p_raw={diff_entry["p_raw"]:.3g}, p_BH={diff_entry["p_corrected"]:.3g} (one-sided, BH-corrected)')


#     # ---- Shared legend
#     # fig.legend(
#     #     handles=legend_handles, loc="center left",
#     #     bbox_to_anchor=(0.98, 0.5),
#     #     borderaxespad=0.0,
#     #     fontsize=fs_legend
#     # )

#     plt.tight_layout(rect=[0, 0.03, 0.98, 1])
#     if save_path:
#         plt.savefig(save_path, dpi=300, bbox_inches='tight')
#         plt.close(fig)
#     else:
#         plt.show()
