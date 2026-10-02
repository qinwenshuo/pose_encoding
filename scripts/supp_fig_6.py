"""
Supplemental Figure 6: perceptual validation of the two-dot stimuli, full
2x2 manipulation grid.

Loads participant communication ratings (position.csv, facing.csv) for four
2x2 manipulation conditions (same-2D/diff-3D position, same-3D/diff-2D
position, same-2D/diff-3D facing, same-3D/diff-2D facing), runs paired
sign-flip permutation tests (BH-FDR corrected within condition) on
participant-level means, and plots a 2x2 grid of bar charts + mini scene
illustrations + significance brackets, followed by a printed table of all
pairwise test results.

See figure_6.py for the companion close-up comparison (the 1a-vs-1b pair
from condition 1, highlighted in the main text), and plot_responses.ipynb
for the original notebook these two scripts were split from.
"""

import os
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.figstyle import apply_style, save_figure, DOUBLE_COL_MM
apply_style()   # Arial, 7 pt text, editable PDF glyphs
import matplotlib.patches as mpatches
from matplotlib.patches import Circle
from matplotlib.lines import Line2D
from statsmodels.stats.multitest import multipletests

DATA_DIR = 'data/raw/synthetic_videos'


# ── 1. Load ratings ────────────────────────────────────────────────────────

def load_csv(path):
    d = pd.read_csv(path)
    d['video_name'] = d['video'].apply(
        lambda url: re.sub(r'_r\d+$', '', url.split('/')[-1].replace('.mp4', ''))
    )
    return d

df_pos = load_csv(f'{DATA_DIR}/position.csv')   # C1, C2
df_fac = load_csv(f'{DATA_DIR}/facing.csv')     # C3, C4

# Each bar is (label, video_name key, scene key matching two_dots_angles.ipynb)
conditions = [
    {
        'title': '1: Same 2D, Diff 3D position',
        'df': df_pos,
        'bars': [
            ('1a: Near\nin-plane',     'c11_c21_c31_c41_near_in_plane', 'near_in_plane'),
            ('1b: Near\nout-of-plane', 'c12_near_out_of_plane',         'near_out_of_plane'),
            ('1c: Far\nin-plane',      'c13_c23_far_in_plane',          'far_in_plane'),
            ('1d: Far\nout-of-plane',  'c14_far_out_of_plane',          'far_out_of_plane'),
        ],
    },
    {
        'title': '2: Same 3D, Diff 2D position',
        'df': df_pos,
        'bars': [
            ('2a: Near\ncam 0°',  'c11_c21_c31_c41_near_in_plane', 'near_cam0'),
            ('2b: Near\ncam 60°', 'c22_c42_near_cam60',            'near_cam60'),
            ('2c: Far\ncam 0°',   'c13_c23_far_in_plane',          'far_cam0'),
            ('2d: Far\ncam 60°',  'c24_far_cam60',                 'far_cam60'),
        ],
    },
    {
        'title': '3: Same 2D, Diff 3D facing',
        'df': df_fac,
        'bars': [
            ('3a: Facing\nflat',       'c11_c21_c31_c41_near_in_plane', 'facing_flat'),
            ('3b: Facing\ntilted',     'c32_facing_tilted',             'facing_tilted'),
            ('3c: Not-facing\nflat',   'c33_c43_notfacing_flat',        'notfacing_flat'),
            ('3d: Not-facing\ntilted', 'c34_notfacing_tilted',          'notfacing_tilted'),
        ],
    },
    {
        'title': '4: Same 3D, Diff 2D facing',
        'df': df_fac,
        'bars': [
            ('4a: Facing\ncam 0°',      'c11_c21_c31_c41_near_in_plane', 'facing_cam0'),
            ('4b: Facing\ncam 60°',     'c22_c42_near_cam60',            'facing_cam60'),
            ('4c: Not-facing\ncam 0°',  'c33_c43_notfacing_flat',        'notfacing_cam0'),
            ('4d: Not-facing\ncam 60°', 'c44_notfacing_cam60',           'notfacing_cam60'),
        ],
    },
]

print(f"position.csv: {len(df_pos)} rows, {df_pos['pid'].nunique()} participants")
print(f"facing.csv:   {len(df_fac)} rows, {df_fac['pid'].nunique()} participants")


# ── 2. Paired permutation tests ────────────────────────────────────────────

def paired_permutation_test(a, b, n_perm=10000, rng=None):
    """Paired sign-flip permutation test on participant-level means.

    a, b must be paired arrays (same length, same participant order) -
    e.g. one mean rating per participant per condition, since each
    participant rated every condition (3 reps each) in this design.
    """
    if rng is None:
        rng = np.random.default_rng(42)
    diffs = np.array(a) - np.array(b)
    observed = np.abs(diffs.mean())
    n = len(diffs)
    signs = rng.choice([-1, 1], size=(n_perm, n))
    perm_means = np.abs((diffs * signs).mean(axis=1))
    count = np.sum(perm_means >= observed)
    return observed, count / n_perm

PAIRS = [(0, 1), (0, 2), (1, 3), (2, 3)]

# Pre-compute all permutation tests
rng_perm = np.random.default_rng(0)
perm_results = {}
for cond in conditions:
    df = cond['df']
    bar_keys = [key for _, key, _ in cond['bars']]

    # raw per-rating values (all 3 reps), used only for plotting
    groups = [df[df['video_name'] == key]['q_communication'].values
              for _, key, _ in cond['bars']]

    # participant-level means (avg over 3 reps), one row per pid,
    # aligned across conditions since every participant saw every video
    pivot = (df.groupby(['pid', 'video_name'])['q_communication']
               .mean().unstack()[bar_keys])

    cond_res = {}
    pairs_done = []
    for i, j in PAIRS:
        a = pivot.iloc[:, i].values
        b = pivot.iloc[:, j].values
        diff, p = paired_permutation_test(a, b, n_perm=10000, rng=rng_perm)
        cond_res[(i, j)] = {'diff': diff, 'p': p}
        pairs_done.append((i, j))
    # BH FDR correction within condition
    if pairs_done:
        pvals = [cond_res[(i, j)]['p'] for i, j in pairs_done]
        _, pvals_fdr, _, _ = multipletests(pvals, method='fdr_bh')
        for (i, j), p_fdr in zip(pairs_done, pvals_fdr):
            cond_res[(i, j)]['p_fdr'] = p_fdr
    perm_results[cond['title']] = cond_res


def sig_stars(p):
    if p < 0.001: return '***'
    if p < 0.01:  return '**'
    if p < 0.05:  return '*'
    return None


# Tight bracket-stacking geometry (data units on the 0-100 rating scale) so
# the significance annotations sit close above the bars, paper-figure style.
BR_GAP, BR_TICK, BR_LPAD = 2.2, 2, 8.5


def bracket_height(pairs_with_p):
    """Compute max y height all brackets would occupy, without drawing."""
    if not pairs_with_p:
        return 100
    pwp = sorted(pairs_with_p, key=lambda x: (x[1] - x[0], x[0]))
    occupied = [100] * 4
    for i, j, _ in pwp:
        h = max(occupied[i:j + 1]) + BR_GAP
        for k in range(i, j + 1):
            occupied[k] = h + BR_TICK + BR_LPAD
    return max(occupied)


def add_sig_brackets(ax, pairs_with_p):
    """Draw brackets for all tested pairs; significant ones get stars, others n.s.
    Drawn with clip_on=False so they float above the axes' y=100 frame."""
    if not pairs_with_p:
        return 100
    pwp = sorted(pairs_with_p, key=lambda x: (x[1] - x[0], x[0]))

    occupied = [100] * 4

    for i, j, p in pwp:
        label = sig_stars(p) or 'n.s.'
        h = max(occupied[i:j + 1]) + BR_GAP
        ax.plot([i, i, j, j], [h, h + BR_TICK, h + BR_TICK, h],
                lw=0.63, color='black', clip_on=False)
        ax.text((i + j) / 2, h + BR_TICK + 0.6, label,
                ha='center', va='bottom', fontsize=7, clip_on=False)
        for k in range(i, j + 1):
            occupied[k] = h + BR_TICK + BR_LPAD

    return max(occupied)


# ── 3. Two-dot scene illustrations (geometry copied from two_dots_angles.ipynb) ──

CAM_STD = np.array([0., 0.])
CAM_60  = np.array([2 * np.sqrt(3), 2.])

SCENES = {
    'near_in_plane':          dict(p1=[-1., 4.],  a1=0.,   p2=[1., 4.],  a2=180., camera=CAM_STD),
    'near_out_of_plane':      dict(p1=[-1., 3.],  a1=0.,   p2=[1., 5.],  a2=180., camera=CAM_STD),
    'far_in_plane':           dict(p1=[-2.5, 4.], a1=0.,   p2=[2.5, 4.], a2=180., camera=CAM_STD),
    'far_out_of_plane':       dict(p1=[-2.5, 3.], a1=0.,   p2=[2.5, 5.], a2=180., camera=CAM_STD),
    'near_cam60':             dict(p1=[-1., 4.],  a1=0.,   p2=[1., 4.],  a2=180., camera=CAM_60),
    'far_cam60':               dict(p1=[-2.5, 4.], a1=0.,   p2=[2.5, 4.], a2=180., camera=CAM_60),
    'facing_tilted':          dict(p1=[-1., 4.],  a1=60.,  p2=[1., 4.],  a2=240., camera=CAM_STD),
    'notfacing_flat':         dict(p1=[-1., 4.],  a1=180., p2=[1., 4.],  a2=0.,   camera=CAM_STD),
    'notfacing_tilted':       dict(p1=[-1., 4.],  a1=240., p2=[1., 4.],  a2=60.,  camera=CAM_STD),
    'notfacing_cam60':        dict(p1=[-1., 4.],  a1=180., p2=[1., 4.],  a2=0.,   camera=CAM_60),
}
SCENES['near_cam0']      = SCENES['near_in_plane']
SCENES['far_cam0']       = SCENES['far_in_plane']
SCENES['facing_flat']    = SCENES['near_in_plane']
SCENES['facing_cam0']    = SCENES['near_in_plane']
SCENES['facing_cam60']   = SCENES['near_cam60']
SCENES['notfacing_cam0'] = SCENES['notfacing_flat']


def draw_mini_scene(ax, scene_key, color, xlim=(-4, 4), ylim=(0, 8)):
    """Compact two-dot illustration (agent A=red, B=blue, camera=black triangle)."""
    s = SCENES[scene_key]
    p1, p2 = np.array(s['p1']), np.array(s['p2'])
    cx, cy = s['camera']
    ax.plot(cx, cy, 'k^', markersize=4.1, zorder=6, clip_on=False)
    ax.add_patch(Circle(p1, 0.22, color='#C44E52', zorder=3))
    ax.add_patch(Circle(p2, 0.22, color='#4C72B0', zorder=3))
    for pos, angle, c in [(p1, s['a1'], '#C44E52'), (p2, s['a2'], '#4C72B0')]:
        d = np.array([np.cos(np.radians(angle)), np.sin(np.radians(angle))])
        ax.annotate('', xy=pos + d * 1.2, xytext=pos,
                    arrowprops=dict(arrowstyle='->', color=c, lw=0.82, mutation_scale=5.9),
                    zorder=4)
    ax.set_aspect('equal')
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xticks(np.arange(xlim[0], xlim[1] + 1, 1))
    ax.set_yticks(np.arange(ylim[0], ylim[1] + 1, 1))
    ax.tick_params(labelbottom=False, labelleft=False, length=0)
    ax.grid(True, alpha=0.35, linewidth=0.27)
    ax.set_axisbelow(True)
    for sp in ax.spines.values():
        sp.set_edgecolor(color)
        sp.set_linewidth(0.73)


# -- Save plotted data --
data_dir = "./results/plot_data/supp_fig_6"
os.makedirs(data_dir, exist_ok=True)
raw_rows, pid_rows = [], []
for cond in conditions:
    df = cond['df']
    for label, video_key, scene_key in cond['bars']:
        subset = df[df['video_name'] == video_key]
        for _, row in subset.iterrows():
            raw_rows.append({
                'condition': cond['title'], 'bar_label': label.replace('\n', ' '),
                'video_name': video_key, 'pid': row['pid'], 'q_communication': row['q_communication'],
            })
        pid_means = subset.groupby('pid')['q_communication'].mean()
        for pid, val in pid_means.items():
            pid_rows.append({
                'condition': cond['title'], 'bar_label': label.replace('\n', ' '),
                'video_name': video_key, 'pid': pid, 'q_communication_mean': val,
            })
pd.DataFrame(raw_rows).to_csv(f"{data_dir}/individual_responses.csv", index=False)
pd.DataFrame(pid_rows).to_csv(f"{data_dir}/participant_averages.csv", index=False)
print(f"Saved plotted data → {data_dir}/")

# ── 4. Plot: 2x2 grid of condition blocks ─────────────────────────────────
# 180 mm wide (2-column) and ≤ 185 mm tall. Every axes is placed explicitly
# in inches so the bracket stacks, tick labels and scene boxes never collide.
# One shared key sits under the grid.

colors = ['#4C72B0', '#DD8452', '#55A868', '#C44E52']
AGENT_A, AGENT_B, NEUTRAL = '#C44E52', '#4C72B0', '#555555'
rng = np.random.default_rng(42)

# Uniform title y across all conditions (based on tallest bracket stack)
all_pairs_with_p = [
    [(i, j, perm_results[cond['title']][(i, j)]['p_fdr'])
     for (i, j) in PAIRS if (i, j) in perm_results[cond['title']]]
    for cond in conditions
]
title_y = max(bracket_height(pwp) for pwp in all_pairs_with_p) + 5

# Layout, in inches. Vertical sizes are per condition row, top to bottom.
FIG_W     = 7.0
BRACKET_H = 0.80   # above the bars: bracket stack + condition title
BAR_H     = 1.50
LABEL_H   = 0.32   # two-line x tick labels
ILLUS_H   = 0.55   # square scene boxes
ROW_GAP   = 0.14
KEY_H     = 0.50
LEFT, RIGHT, MID_GAP = 0.50, 0.05, 0.20
ROW_H = BRACKET_H + BAR_H + LABEL_H + ILLUS_H + ROW_GAP
FIG_H = 2 * ROW_H + KEY_H
COL_W = (FIG_W - LEFT - RIGHT - MID_GAP) / 2
XLIM  = (-0.7, 3.7)

fig = plt.figure(figsize=(FIG_W, FIG_H), dpi=300)


def rect(x, y_top, w, h):
    """Inches (x from left edge, y from top edge) → figure-fraction rect."""
    return [x / FIG_W, 1 - (y_top + h) / FIG_H, w / FIG_W, h / FIG_H]


for k, (cond, color, pairs_with_p) in enumerate(zip(conditions, colors, all_pairs_with_p)):
    row, col = divmod(k, 2)
    x0    = LEFT + col * (COL_W + MID_GAP)
    y_bar = row * ROW_H + BRACKET_H

    df = cond['df']
    groups = [df[df['video_name'] == key]['q_communication'].values
              for _, key, _ in cond['bars']]

    ax = fig.add_axes(rect(x0, y_bar, COL_W, BAR_H))

    for i, (label, video_key, scene_key) in enumerate(cond['bars']):
        subset = groups[i]
        mean_val = subset.mean() if len(subset) > 0 else 0

        ax.bar(i, mean_val, color=color, alpha=0.65, width=0.6,
               edgecolor=color, linewidth=0.54, zorder=2)
        if len(subset) > 0:
            # individual ratings (all 3 reps): light, jittered open circles
            jitter = rng.uniform(-0.15, 0.15, len(subset))
            ax.scatter(i + jitter, subset,
                       facecolors='none', edgecolors=color, alpha=0.45,
                       s=6.6, linewidths=0.36, zorder=3, clip_on=False)

            # participant-level averages (3 reps collapsed to 1 point per
            # participant): solid diamonds, not jittered
            pid_means = (df[df['video_name'] == video_key]
                         .groupby('pid')['q_communication'].mean().values)
            ax.scatter(np.full(len(pid_means), i), pid_means,
                       marker='D', color=color, s=8.6, zorder=4,
                       edgecolors='white', linewidths=0.32, clip_on=False)

    add_sig_brackets(ax, pairs_with_p)

    ax.text(1.5, title_y, cond['title'],
            ha='center', va='bottom', fontsize=7, fontweight='bold', clip_on=False)

    ax.set_xticks(np.arange(4))
    ax.set_xticklabels([lbl for lbl, _, _ in cond['bars']], fontsize=7)
    ax.spines[['top', 'right']].set_visible(False)
    ax.spines[['left', 'bottom']].set_linewidth(0.54)
    ax.set_xlim(*XLIM)
    ax.set_ylim(0, 100)
    ax.set_yticks(range(0, 101, 20))
    ax.tick_params(axis='y', labelsize=7)
    ax.tick_params(axis='x', length=0, pad=1.5)
    if col == 0:
        ax.set_ylabel('Communication rating', fontsize=7, fontweight='bold')
    else:
        ax.tick_params(labelleft=False)

    # Square mini scene illustration centred under each bar
    y_illus = y_bar + BAR_H + LABEL_H
    for i, (label, video_key, scene_key) in enumerate(cond['bars']):
        x_c = x0 + (i - XLIM[0]) / (XLIM[1] - XLIM[0]) * COL_W
        iax = fig.add_axes(rect(x_c - ILLUS_H / 2, y_illus, ILLUS_H, ILLUS_H))
        draw_mini_scene(iax, scene_key, color)

# Shared key: scene symbols and data markers.
key_handles = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor=AGENT_A, markersize=5, label='Agent A'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor=AGENT_B, markersize=5, label='Agent B'),
    Line2D([0], [0], marker=r'$\rightarrow$', color=NEUTRAL, linestyle='None', markersize=8,
           label='Facing direction'),
    Line2D([0], [0], marker='^', color='k', linestyle='None', markersize=4.6, label='Camera'),
    mpatches.Patch(facecolor='#999999', alpha=0.65, edgecolor='#999999', label='Mean rating'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='none',
           markeredgecolor=NEUTRAL, markersize=3.9, label='Individual response'),
    Line2D([0], [0], marker='D', color='w', markerfacecolor=NEUTRAL,
           markeredgecolor='white', markersize=3.5, label='Participant average'),
]
fig.legend(handles=key_handles, loc='lower center',
           bbox_to_anchor=(LEFT / FIG_W + (FIG_W - LEFT - RIGHT) / FIG_W / 2, 0.0),
           ncol=4, frameon=False, fontsize=7, handletextpad=0.4, columnspacing=1.2)

save_figure(fig, "./results/supp_fig_6.pdf", DOUBLE_COL_MM)
plt.close(fig)


# ── 5. Permutation test results table ─────────────────────────────────────

rows = []
for cond in conditions:
    df = cond['df']
    bar_labels = [lbl.replace('\n', ' ') for lbl, _, _ in cond['bars']]
    groups = [df[df['video_name'] == key]['q_communication'].values
              for _, key, _ in cond['bars']]
    cond_perm = perm_results[cond['title']]
    for i, j in PAIRS:
        if (i, j) not in cond_perm:
            continue
        a, b = groups[i], groups[j]
        r = cond_perm[(i, j)]
        rows.append({
            'Condition':  cond['title'],
            'Group A':    bar_labels[i],
            'Group B':    bar_labels[j],
            'n (participants)': df['pid'].nunique(),
            'Mean A': round(a.mean(), 2),
            'Mean B': round(b.mean(), 2),
            'Diff':   round(r['diff'], 2),
            'p-value':    round(r['p'], 4),
            'p (BH FDR)': round(r['p_fdr'], 4),
        })

results_df = pd.DataFrame(rows)
results_df['sig (raw)']    = results_df['p-value'].apply(
    lambda v: '***' if v < .001 else ('**' if v < .01 else ('*' if v < .05 else 'ns')))
results_df['sig (BH FDR)'] = results_df['p (BH FDR)'].apply(
    lambda v: '***' if v < .001 else ('**' if v < .01 else ('*' if v < .05 else 'ns')))

print("\nPaired sign-flip permutation test on participant-level means (3 reps averaged per participant)")
print("10,000 permutations, two-sided | BH FDR × 4 within condition | n=20 participants")
print("* p<.05  ** p<.01  *** p<.001\n")
print(results_df.to_string(index=False))

results_df.to_csv(f"{data_dir}/pairwise_test_results.csv", index=False)
print(f"\nSaved test results → {data_dir}/pairwise_test_results.csv")
