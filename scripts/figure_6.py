"""
Figure 6: close-up comparison, near in-plane vs. near out-of-plane
(condition 1a vs. 1b), with example video frames.

Outputs (all in results/fig6/) are separate so they can be assembled in
Illustrator/PowerPoint with labels and arrows laid over the renders:
  • chart.svg                 vector bar chart, exactly 84 × 54 mm, Arial 7 pt (key 5 pt),
                              editable text, nothing clipped at 0 or 100
  • frame_<condition>.png     rendered frames for 32 × 21 mm: cropped to 32:21 and
                              downscaled (never upscaled) to 756 × 496 px (600 dpi),
                              sRGB PNG

Duplicates the data loading and permutation-test setup from supp_fig_7.py
(needed here only to recover the BH-FDR-corrected p-value for the specific
1a-vs-1b pair, which depends on the other three pairs tested within
condition 1) and adds the two example-frame illustrations.

See supp_fig_7.py for the full 2x2 grid this pair is drawn from, and
plot_responses.ipynb for the original notebook these two scripts were
split from.
"""

import os
import re

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from PIL import ImageCms
from src.figstyle import apply_style, save_exact
apply_style()   # Arial, 7 pt text, editable PDF glyphs
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


# ── 2. Paired permutation tests (needed for the 1a-vs-1b BH-FDR p-value) ───

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

rng_perm = np.random.default_rng(0)
perm_results = {}
for cond in conditions:
    df = cond['df']
    bar_keys = [key for _, key, _ in cond['bars']]
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


# ── 3. Plot: close-up near in-plane vs. near out-of-plane ─────────────────

def get_middle_frame(video_path):
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, total // 2)
    ret, frame = cap.read()
    cap.release()
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if ret else None

frame_in  = get_middle_frame(f'{DATA_DIR}/videos/c11_c21_c31_c41_near_in_plane_r1.mp4')
frame_out = get_middle_frame(f'{DATA_DIR}/videos/c12_near_out_of_plane_r1.mp4')

# Data for the two bars
df = conditions[0]['df']
keys   = ['c11_c21_c31_c41_near_in_plane', 'c12_near_out_of_plane']
labels = ['Near in-plane', 'Near out-of-plane']
groups = [df[df['video_name'] == k]['q_communication'].values for k in keys]
xpos   = [0, 2]

p_fdr  = perm_results['1: Same 2D, Diff 3D position'][(0, 1)]['p_fdr']
bracket_label = sig_stars(p_fdr) or 'n.s.'

# -- Save plotted data --
data_dir = "./results/plot_data/fig6"
os.makedirs(data_dir, exist_ok=True)
label_of = dict(zip(keys, labels))
raw = df[df['video_name'].isin(keys)][['pid', 'video_name', 'q_communication']].copy()
raw['label'] = raw['video_name'].map(label_of)
raw.to_csv(f"{data_dir}/individual_responses.csv", index=False)
pid_means_df = (
    df[df['video_name'].isin(keys)]
    .groupby(['pid', 'video_name'])['q_communication'].mean()
    .reset_index()
)
pid_means_df['label'] = pid_means_df['video_name'].map(label_of)
pid_means_df.to_csv(f"{data_dir}/participant_averages.csv", index=False)
print(f"Saved plotted data → {data_dir}/")

OUT_DIR = "./results/fig6"
os.makedirs(OUT_DIR, exist_ok=True)

CHART_MM = (84, 54)
FRAME_MM = (32, 21)
FRAME_DPI = 600                                    # render above 300 dpi, then downscale
FRAME_PX = tuple(round(v / 25.4 * FRAME_DPI) for v in FRAME_MM)   # (756, 496)
color = '#4C72B0'
rng2  = np.random.default_rng(42)

# -- Rendered frames: crop to 32:21, downscale with Lanczos (never upscale), sRGB PNG --
srgb_icc = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
for slug, frame in zip(['near_in_plane', 'near_out_of_plane'], [frame_in, frame_out]):
    if frame is None:
        continue
    img = Image.fromarray(frame).convert("RGB")
    w, h = img.size
    aspect = FRAME_MM[0] / FRAME_MM[1]
    if w / h > aspect:                             # too wide: trim the sides equally
        cw, ch = round(h * aspect), h
    else:                                          # too tall: trim top and bottom equally
        cw, ch = w, round(w / aspect)
    left, top = (w - cw) // 2, (h - ch) // 2
    img = img.crop((left, top, left + cw, top + ch))
    assert cw >= FRAME_PX[0] and ch >= FRAME_PX[1], "source too small: would need upscaling"
    img = img.resize(FRAME_PX, Image.LANCZOS)
    path = f"{OUT_DIR}/frame_{slug}.png"
    img.save(path, dpi=(FRAME_DPI, FRAME_DPI), icc_profile=srgb_icc)
    print(f"Saved → {path}  ({w}×{h} px source → crop {cw}×{ch} → {FRAME_PX[0]}×{FRAME_PX[1]} px, "
          f"{FRAME_PX[0] / (FRAME_MM[0] / 25.4):.0f} dpi at {FRAME_MM[0]} × {FRAME_MM[1]} mm, sRGB)")

# -- Vector bar chart --
fig, ax = plt.subplots()

for xi, k, subset, lbl in zip(xpos, keys, groups, labels):
    ax.bar(xi, subset.mean(), color=color, alpha=0.6, width=0.6, zorder=2, clip_on=False)

    # individual ratings (all 3 reps): light, jittered open circles
    jitter = rng2.uniform(-0.15, 0.15, len(subset))
    ax.scatter(xi + jitter, subset,
               facecolors='none', edgecolors=color, alpha=0.45,
               s=5, linewidths=0.35, zorder=3, clip_on=False)

    # participant-level averages (3 reps collapsed to 1 point per
    # participant): solid diamonds, not jittered
    pid_means = df[df['video_name'] == k].groupby('pid')['q_communication'].mean().values
    ax.scatter(np.full(len(pid_means), xi), pid_means,
               marker='D', color=color, s=7, zorder=4,
               edgecolors='white', linewidths=0.3, clip_on=False)

# Significance bracket
h, tick = 104, 3
x0, x1 = xpos
ax.plot([x0, x0, x1, x1], [h, h+tick, h+tick, h], lw=0.5, color='black', clip_on=False)
ax.text((x0+x1)/2, h+tick+0.5, bracket_label,
        ha='center', va='bottom', fontsize=7, clip_on=False)

ax.set_ylim(0, 100)
ax.set_yticks(range(0, 101, 20))
ax.set_xlim(-1, 3)
ax.set_xticks(xpos)
ax.set_xticklabels(labels, fontsize=7)
ax.tick_params(axis='y', labelsize=7)
ax.set_ylabel('Communication rating', fontsize=7, fontweight='bold')
ax.spines[['top', 'right']].set_visible(False)

# Key, outside the axes to the right, vertically centered.
key_handles = [
    plt.Rectangle((0, 0), 1, 1, facecolor=color, alpha=0.6, label='Mean rating'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='none',
           markeredgecolor=color, markersize=3.4, label='Individual response'),
    Line2D([0], [0], marker='D', color='w', markerfacecolor=color,
           markeredgecolor='white', markersize=3, label='Participant average'),
]
ax.legend(handles=key_handles, loc='center left', bbox_to_anchor=(1.01, 0.5),
          frameon=False, fontsize=5, handlelength=1.2, handletextpad=0.4,
          borderaxespad=0.2)   # 5 pt key leaves more width for the plot

save_exact(fig, f"{OUT_DIR}/chart.svg", *CHART_MM,
           relayout=lambda: fig.tight_layout(pad=0.3))
plt.close(fig)
