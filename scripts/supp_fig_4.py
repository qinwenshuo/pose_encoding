"""
Supplemental Figure 4: CV encoding across feature variants and datasets.
"""

import argparse
import json
import os
import textwrap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import colors as mcolors
from joblib import Parallel, delayed
from tqdm import tqdm
from src.config import ALPHAS, JOBS
from src.encoding import encode
from src.plottings import change_name

POSE_FEATURE_COLORS = {
    '3D social pose features': "#FC1212",
    '2D social pose features': "#FA7946",
    '3D body joints':          "#56AA29",
    '2D body joints':          "#A8D070",
}

_TARGET_DISPLAY_NAME = {
    'communication': 'communicative interaction',
    'joint action':  'physical interaction',
}

# Split-half reliability values (same as in encoding_bar_with_points)
_NOISE_CEILING = {
    "spatial expanse":     0.7194827819823470,
    "interagent distance": 0.8854506352255450,
    "agents facing":       0.9574809908008190,
    "communication":       0.7621707694002240,
    "joint action":        0.7688673458027810,
}

POSE_FEATURES = [
    '3D body joints',
    '2D body joints',
    '3D social pose features',
    '2D social pose features',
]

# Five aggregation variants; keys must match what 02_video_level_features.py writes
VARIANTS = {
    'averaged':      POSE_FEATURES,
    'concat':        [f'{f} (concat)'        for f in POSE_FEATURES],
    'diff':          [f'{f} (diff)'          for f in POSE_FEATURES],
    'mean+diff':     [f'{f} (mean+diff)'     for f in POSE_FEATURES],
    'mean+diff+std': [f'{f} (mean+diff+std)' for f in POSE_FEATURES],
}

DATASETS = {
    '500 videos': {
        'path':    'data/processed/dyad_videos_500/mesh_video_level_features/',
        'ratings': 'data/raw/dyad_videos/video_ratings.csv',
        'targets': {
            'spatial_expanse': 'spatial expanse',
            'agent_distance':  'interagent distance',
            'communication':   'communication',
            'joint_action':    'joint action',
        },
    },
    '250 videos': {
        'path':    'data/processed/dyad_videos/mesh_video_level_features/',
        'ratings': 'data/raw/dyad_videos/behavioral_ratings.csv',
        'targets': {
            'spatial expanse':     'spatial expanse',
            'interagent distance': 'interagent distance',
            'agents facing':       'agents facing',
            'communication':       'communication',
            'joint action':        'joint action',
        },
    },
}


def load_all(pose_feature, target_col, video_level_path, ratings_path):
    """Return (X, y, names) aligned across videos in video_level_path, dropping missing."""
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


def run_one(pose_feature, target_col, target_display, video_level_path, ratings_path):
    X, y, _ = load_all(pose_feature, target_col, video_level_path, ratings_path)
    if len(X) == 0:
        print(f'No data: {pose_feature} × {target_display}')
        return None
    result = encode(
        X_train=X, X_test=None,
        y_train=y, y_test=None,
        x_type=pose_feature, layer_name=None, feature=target_display,
        eval_mode='cv',
        alphas=ALPHAS,
        cv_splits=10,
    )
    result['n'] = len(X)
    return result


def run_cv(video_level_path, ratings_path, targets, pose_features, label):
    """Run CV encoding; return per-fold DataFrame with columns [x, y, score_r]."""
    jobs = [
        (pf, tc, td)
        for pf in pose_features
        for tc, td in targets.items()
    ]
    results = Parallel(n_jobs=JOBS)(
        delayed(run_one)(pf, tc, td, video_level_path, ratings_path)
        for pf, tc, td in tqdm(jobs, desc=label)
    )

    valid = [r for r in results if r is not None]
    if valid:
        summary = pd.DataFrame(valid)
        print(f'\n=== {label} ===')
        print(summary.pivot(index='x', columns='y', values='score_r').to_string())

    rows = []
    for r in valid:
        for fold_r in r['score_r_per_fold']:
            rows.append({'x': r['x'], 'y': r['y'], 'score_r': fold_r})
    return pd.DataFrame(rows)


def _base(name):
    """Strip the aggregation-variant suffix to recover the underlying pose feature name."""
    for suffix in (' (mean+diff+std)', ' (mean+diff)', ' (concat)', ' (diff)'):
        name = name.replace(suffix, '')
    return name


def report_feature_differences(df, x_to_plot):
    """
    Print, for a given variant's feature set, the mean prediction score (r) per
    pose feature averaged across all targets, plus the averaged differences
    between 3D vs 2D and between social-pose-features vs body-joints.
    """
    means = df.groupby('x')['score_r'].mean().reindex(x_to_plot)

    print('\n--- Mean score (r), averaged across targets ---')
    print(means.round(3).to_string())

    def get(base_name):
        matches = [x for x in x_to_plot if _base(x) == base_name]
        return means[matches[0]] if matches else None

    pairs_3d_2d = [
        ('3D body joints', '2D body joints'),
        ('3D social pose features', '2D social pose features'),
    ]
    pairs_social_joints = [
        ('3D social pose features', '3D body joints'),
        ('2D social pose features', '2D body joints'),
    ]

    print('\n--- Averaged difference: 3D minus 2D (same feature type) ---')
    diffs = []
    for hi, lo in pairs_3d_2d:
        v_hi, v_lo = get(hi), get(lo)
        if v_hi is None or v_lo is None:
            continue
        d = v_hi - v_lo
        diffs.append(d)
        print(f'{hi} - {lo} = {d:.3f}')
    if diffs:
        print(f'overall average 3D - 2D = {np.mean(diffs):.3f}')

    print('\n--- Averaged difference: social pose features minus body joints (same dimensionality) ---')
    diffs = []
    for hi, lo in pairs_social_joints:
        v_hi, v_lo = get(hi), get(lo)
        if v_hi is None or v_lo is None:
            continue
        d = v_hi - v_lo
        diffs.append(d)
        print(f'{hi} - {lo} = {d:.3f}')
    if diffs:
        print(f'overall average social - joints = {np.mean(diffs):.3f}')


def _plot_panel(ax, df, x_to_plot, y_to_plot, color_dict,
                fs_tick=16, y_range=(-0.25, 1.0)):
    """
    Publication-quality bar + dot plot mirroring encoding_bar_with_points:
    same grid, font sizes, dot size, noise ceiling band, and bracket aesthetics.
    """
    stats_grp = df.groupby(['x', 'y'], as_index=False)['score_r'].mean()

    n_targets   = len(y_to_plot)
    n_hues      = len(x_to_plot)
    total_width = 0.7
    bar_width   = total_width / n_hues
    x_idx       = np.arange(n_targets)

    light = {n: mcolors.to_rgba(color_dict.get(_base(n), '#888888'), alpha=0.55) for n in x_to_plot}

    # Grid & spines — matches encoding_bar_with_points
    ax.grid(axis='x', visible=False)
    ax.grid(axis='y', linestyle='-', linewidth=1.0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Bars
    for i, predictor in enumerate(x_to_plot):
        centers = x_idx - total_width / 2 + (i + 0.5) * bar_width
        means = [
            stats_grp.loc[(stats_grp.x == predictor) & (stats_grp.y == t), 'score_r'].values[0]
            if ((stats_grp.x == predictor) & (stats_grp.y == t)).any() else np.nan
            for t in y_to_plot
        ]
        ax.bar(centers, means, width=bar_width,
               color=light[predictor],
               edgecolor=color_dict.get(_base(predictor), '#888888'),
               label=change_name(_base(predictor)))

    # Split-half reliability band (matches encoding_bar_with_points)
    nc_band_h = 0.024
    nc_face   = '#999999D0'
    nc_plotted = False
    for fi, target in enumerate(y_to_plot):
        if target not in _NOISE_CEILING:
            continue
        nc_val  = _NOISE_CEILING[target]
        x_left  = fi - total_width / 2
        x_right = fi + total_width / 2
        ax.add_patch(mpatches.Rectangle(
            (x_left, nc_val - nc_band_h / 2),
            width=(x_right - x_left) * 1.05, height=nc_band_h,
            facecolor=nc_face, edgecolor='none',
        ))
        nc_plotted = True

    ax.set_ylim(y_range[0], y_range[1])
    ax.axhline(0, color='gray', linestyle='--', linewidth=0.8)

    # x-axis: maps 'communication'→'communicative interaction', etc.
    ax.set_xticks(x_idx)
    ax.set_xticklabels(
        [textwrap.fill(_TARGET_DISPLAY_NAME.get(t, t), width=10, break_long_words=False) for t in y_to_plot],
        ha='center', fontsize=fs_tick,
    )
    ax.tick_params(axis='y', labelsize=fs_tick)
    ax.set_ylabel('Score ($r$)', fontsize=fs_tick, weight='bold')

    # Legend — right-side, fancybox, matches encoding_bar_with_points
    handles, labels = ax.get_legend_handles_labels()
    seen, unique_h, unique_l = set(), [], []
    for h, l in zip(handles, labels):
        if l not in seen:
            seen.add(l); unique_h.append(h); unique_l.append(l)
    if nc_plotted:
        unique_h.append(mpatches.Patch(facecolor=nc_face, edgecolor='none'))
        unique_l.append('Split-half reliability')
    ax.legend(unique_h, unique_l, bbox_to_anchor=(1, 0.5), fancybox=True,
              fontsize=fs_tick - 2)


def main():
    parser = argparse.ArgumentParser(description='CV encoding for a single variant × dataset panel.')
    parser.add_argument('--variant',  choices=list(VARIANTS.keys()),  default='averaged',
                        help='Feature aggregation variant (default: averaged)')
    parser.add_argument('--dataset',  choices=list(DATASETS.keys()),  default='500 videos',
                        help='Dataset (default: 500 videos)')
    args = parser.parse_args()

    cfg        = DATASETS[args.dataset]
    pose_feats = VARIANTS[args.variant]
    y_to_plot  = list(cfg['targets'].values())
    title      = f'{args.variant}  |  {args.dataset}'

    fold_df = run_cv(cfg['path'], cfg['ratings'], cfg['targets'], pose_feats, label=title)
    report_feature_differences(fold_df, pose_feats)

    _, ax = plt.subplots(figsize=(12, 6), dpi=300)
    _plot_panel(
        ax,
        df=fold_df,
        x_to_plot=pose_feats,
        y_to_plot=y_to_plot,
        color_dict=POSE_FEATURE_COLORS,
        fs_tick=16,
        y_range=(-0.25, 1.0),
    )

    plt.tight_layout()
    os.makedirs('results', exist_ok=True)
    safe_v = args.variant.replace('+', '_').replace(' ', '_')
    safe_d = args.dataset.replace(' ', '_')
    out    = f'results/supp_fig_4.png'
    plt.savefig(out, bbox_inches='tight', dpi=300)
    plt.show()
    print(f'\nSaved → {out}')


if __name__ == '__main__':
    main()
