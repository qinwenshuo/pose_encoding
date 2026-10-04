"""
Build a plain random 5-way fold assignment of the 500 videos (original 250 +
additional 250), for the nested-CV methodology used by supp_fig_3.py: an outer
5-fold split (each fold = 400 train / 100 test) with an inner 5-fold CV (on the
400) for layer selection within each outer fold.

No distributional balancing -- a plain random partition, per request.

Writes: data/raw/dyad_videos/folds_500.csv  (columns: video_name, fold [0-4])
(--seed/--out let you generate an alternate-seed file without overwriting the
default, e.g. to sanity-check robustness to the specific fold partition chosen.)
"""
import argparse
import numpy as np
import pandas as pd

from src.config import VIDEO_RATINGS_500_PATH, RANDOM

N_FOLDS = 5

parser = argparse.ArgumentParser(description="Build a random 5-fold assignment of the 500 videos.")
parser.add_argument("--seed", type=int, default=RANDOM, help=f"Random seed (default: RANDOM={RANDOM})")
parser.add_argument("--out", type=str, default="data/raw/dyad_videos/folds_500.csv", help="Output CSV path")
args = parser.parse_args()

ratings = pd.read_csv(VIDEO_RATINGS_500_PATH)
video_names = ratings['video_name'].to_numpy()
n = len(video_names)
assert n == 500, f"expected 500 videos, got {n}"

rng = np.random.default_rng(args.seed)
perm = rng.permutation(n)
fold = np.empty(n, dtype=int)
# n=500, N_FOLDS=5 divides evenly into 5 groups of 100
group_size = n // N_FOLDS
for f in range(N_FOLDS):
    fold[perm[f * group_size:(f + 1) * group_size]] = f

out = pd.DataFrame({'video_name': video_names, 'fold': fold}).sort_values('video_name').reset_index(drop=True)
print(out['fold'].value_counts().sort_index())

out.to_csv(args.out, index=False)
print(f"\nSaved -> {args.out} ({len(out)} videos, {N_FOLDS} folds, seed={args.seed})")
