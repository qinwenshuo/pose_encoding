"""
Supplemental Figure 1: Dataset composition summary (in-plane / baby counts).
"""

import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt

VIDEO_NAMES_PATH = 'data/raw/dyad_videos/video_names.csv'
ORIGINAL_VIDEO_DIR = 'data/raw/dyad_videos/dyad_videos_3000ms_250/'
ADDITIONAL_VIDEO_DIR = 'data/raw/dyad_videos/additional_250/'
OUT_DIR = 'results/'

IN_PLANE_LABELS = {0: 'Out of plane', 1: 'In plane'}
BABY_COUNT_LABELS = {0: 'No baby', 1: '1 baby', 2: '2 babies'}

plt.rcParams.update({
    'font.size': 16,
    'axes.titlesize': 19,
    'axes.titleweight': 'bold',
    'axes.labelsize': 17,
    'axes.labelweight': 'bold',
    'xtick.labelsize': 15,
    'ytick.labelsize': 15,
})


def _bar_panel(ax, counts, label_map, title, color):
    labels = [label_map.get(i, str(i)) for i in counts.index]
    total = counts.sum()
    bars = ax.bar(labels, counts.values, color=color, edgecolor='black', linewidth=0.8)
    ax.set_title(title, pad=12)
    ax.set_ylabel('Number of videos')
    ax.grid(axis='y', linestyle='-', linewidth=0.6, alpha=0.35)
    ax.set_axisbelow(True)
    for spine in ('top', 'right'):
        ax.spines[spine].set_visible(False)
    ymax = counts.values.max()
    ax.set_ylim(0, ymax * 1.18)
    for bar, val in zip(bars, counts.values):
        pct = 100 * val / total
        ax.text(bar.get_x() + bar.get_width() / 2, val + ymax * 0.02,
                 f'{val}\n({pct:.0f}%)', ha='center', va='bottom',
                 fontsize=14, fontweight='bold')


def plot_subset(df, title_suffix, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.5))

    counts = df['in_plane'].value_counts().sort_index()
    _bar_panel(axes[0], counts, IN_PLANE_LABELS, 'Standing Positions', '#4C72B0')

    counts = df['baby_count'].value_counts().sort_index()
    _bar_panel(axes[1], counts, BABY_COUNT_LABELS, 'Number of babies in frame', '#55A868')

    fig.suptitle(f'{title_suffix} (n = {len(df)})', fontsize=20, fontweight='bold', y=1.02)
    plt.tight_layout()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved summary plot to {out_path}')

    print(f'\n{title_suffix} (n = {len(df)})')
    print('\nStanding position counts:')
    print(df['in_plane'].value_counts().sort_index().to_string())
    print('\nNumber of babies counts:')
    print(df['baby_count'].value_counts().sort_index().to_string())


def plot_summary(csv_path=VIDEO_NAMES_PATH, original_dir=ORIGINAL_VIDEO_DIR,
                  additional_dir=ADDITIONAL_VIDEO_DIR, out_dir=OUT_DIR):
    df = pd.read_csv(csv_path)

    original_names = set(os.listdir(original_dir))
    is_original = df['video_name'].isin(original_names)
    n_unmatched = (~df['video_name'].isin(original_names) &
                   ~df['video_name'].isin(set(os.listdir(additional_dir)))).sum()
    if n_unmatched:
        print(f'Warning: {n_unmatched} video(s) in {csv_path} not found in '
              f'{original_dir} or {additional_dir}')

    df_original = df[is_original]

    plot_subset(df_original, 'Original videos',
                os.path.join(out_dir, 'video_names_summary_original_250.png'))
    plot_subset(df, 'Full dataset (original + additional videos)',
                os.path.join(out_dir, 'video_names_summary_full_500.png'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot summary statistics of video_names.csv')
    parser.add_argument('--csv_path', type=str, default=VIDEO_NAMES_PATH)
    parser.add_argument('--original_dir', type=str, default=ORIGINAL_VIDEO_DIR)
    parser.add_argument('--additional_dir', type=str, default=ADDITIONAL_VIDEO_DIR)
    parser.add_argument('--out_dir', type=str, default=OUT_DIR)
    args = parser.parse_args()

    plot_summary(csv_path=args.csv_path, original_dir=args.original_dir,
                 additional_dir=args.additional_dir, out_dir=args.out_dir)
