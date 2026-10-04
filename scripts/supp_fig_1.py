"""
Supplemental Figure 1: Variance decomposition of the z (depth) coordinates of the
3D body joints.

The z coordinate of each joint (45 SMPL joints per agent, averaged across frames)
is split into three additive components, for video v, agent p, joint j:
    (1) dyad depth from camera       d[v]     = mean over both agents' joints of z
    (2) depth difference btw agents  a[v,p]   = c[v,p] - d[v],  c[v,p] = mean_j z[v,p,j]
    (3) within-body depth            r[v,p,j] = z[v,p,j] - c[v,p]
so z = (1) + (2) + (3).

Main analysis (plotted): for each feature column (one joint of one agent), take the
variance across videos of z and of each component, sum over all columns, and express
each component's total as a share of the total variance in z. (2) is equal and
opposite for the two agents and (3) sums to zero over each agent's joints, so the
covariance terms cancel after summing and the shares add to 100%. The right bar
repeats this with (1) removed: (2) / ((2) + (3)) and (3) / ((2) + (3)).

Robustness (CSV only): the 500-video set, pooled variance over all video x agent x
joint entries instead of per-feature, the pelvis instead of the joint mean as the
body centre (cross terms are then nonzero and reported), and the 24 core SMPL joints.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.figstyle import apply_style, save_figure, MM, SINGLE_COL_MM
apply_style()   # Arial, 7 pt text, editable PDF glyphs

DATASETS = {
    '250': 'data/processed/dyad_videos/mesh_video_level_features/',
    '500': 'data/processed/dyad_videos_500/mesh_video_level_features/',
}
PLOT_DATASET = '250'
FEATURE = '3D body joints'
N_AGENT, N_JOINT = 2, 45

COLORS = {'camera': '#c3c2b7', 'inter': '#2a78d6', 'body': '#eb6834'}
LABELS = {'camera': 'Dyad depth from camera',
          'inter': 'Depth difference between agents',
          'body': 'Within-body depth'}


def load_z(folder):
    """z coordinates of the 3D body joints, (n_videos, 2 agents, 45 joints)."""
    zs = []
    for f in sorted(os.listdir(folder)):
        if not f.endswith('.json'):
            continue
        with open(os.path.join(folder, f)) as fh:
            row = json.load(fh)[FEATURE]
        if row is None or None in row:
            continue
        zs.append(np.asarray(row, dtype=float).reshape(N_AGENT, N_JOINT, 3)[..., 2])
    return np.stack(zs)


def components(z, centre):
    """Split z (V, 2, J) into dyad depth, between-agent offset and within-body offset."""
    c = centre[..., None]                       # (V, 2, 1) agent centre
    d = centre.mean(1, keepdims=True)[..., None]  # (V, 1, 1) dyad centre
    shape = z.shape
    return {'camera': np.broadcast_to(d, shape),
            'inter': np.broadcast_to(c - d, shape),
            'body': z - c}


def decompose(z, centre, method):
    """Share (%) of z variance for each component; 'cross' collects all covariance terms."""
    comps = components(z, centre)
    if method == 'featurewise':
        # variance across videos per feature column, summed over columns
        var = lambda a: a.reshape(a.shape[0], -1).var(0).sum()
    else:
        # pooled over all video x agent x joint entries
        var = lambda a: a.var() * a.size
    total = var(z)
    out = {k: 100 * var(v) / total for k, v in comps.items()}
    out['cross'] = 100 - sum(out.values())
    within = out['inter'] + out['body']
    out['inter_nocam'] = 100 * out['inter'] / within
    out['body_nocam'] = 100 * out['body'] / within
    out['n_videos'] = z.shape[0]
    return out


rows = []
for ds, folder in DATASETS.items():
    z_all = load_z(folder)
    for joints, n in [('all45', 45), ('smpl24', 24)]:
        z = z_all[:, :, :n]
        for centre_name, centre in [('joint-mean', z.mean(-1)), ('pelvis', z[..., 0])]:
            for method in ['featurewise', 'pooled']:
                rows.append({'dataset': ds, 'joints': joints, 'centre': centre_name,
                             'method': method, **decompose(z, centre, method)})
    z = z_all
    cz = z.mean(-1)
    print(f'{ds} videos: n = {z.shape[0]} with complete {FEATURE}; '
          f'agent depth {cz.mean():.2f} ± {cz.std():.2f} m; '
          f'median |depth difference between agents| {np.median(np.abs(cz[:, 0] - cz[:, 1])):.2f} m; '
          f'mean within-body z SD {(z - cz[..., None]).std(-1).mean():.3f} m')

table = pd.DataFrame(rows)
data_dir = "./results/plot_data/supp_fig_1"
os.makedirs(data_dir, exist_ok=True)
table.to_csv(f"{data_dir}/decomposition_all.csv", index=False)

main = table[(table.joints == 'all45') & (table.centre == 'joint-mean') & (table.method == 'featurewise')]
pd.set_option('display.width', 200)
print('\nShares of z variance (%), 45 joints, joint-mean centre, feature-wise:')
print(main.round(2).to_string(index=False))
print('\nAll variants:')
print(table.round(2).to_string(index=False))

r = main[main.dataset == PLOT_DATASET].iloc[0]
bars = [
    ('All variance', [('camera', r.camera), ('inter', r.inter), ('body', r.body)]),
    ('Dyad depth\nremoved', [('inter', r.inter_nocam), ('body', r.body_nocam)]),
]
pd.DataFrame([{'bar': b.replace('\n', ' '), 'component': LABELS[k], 'share_pct': v}
              for b, parts in bars for k, v in parts]).to_csv(f"{data_dir}/bars.csv", index=False)
print(f"Saved plotted data → {data_dir}/")

# ── Plot: vertical 100% stacked bars ──────────────────────────────────────────
fig, ax = plt.subplots(figsize=(88 * MM, 55 * MM))
xs, width = [0, 1.25], 0.55
for x, (_, parts) in zip(xs, bars):
    bottom = 0.0
    for key, v in parts:
        ax.bar(x, v, bottom=bottom, width=width, color=COLORS[key], edgecolor='white',
               linewidth=1.0, label=LABELS[key] if x == xs[0] else None)
        if v >= 8:   # label only segments tall enough to hold text
            ax.text(x, bottom + v / 2, f'{v:.1f}%', ha='center', va='center',
                    color='white' if key != 'camera' else '#3d3d3a')
        bottom += v

# the two thin segments of the first bar: labels to the right, spread apart, with leaders
mids = [r.camera + r.inter / 2, r.camera + r.inter + r.body / 2]
for v, mid, ty in zip([r.inter, r.body], mids, [86, 100]):
    ax.annotate(f'{v:.1f}%', xy=(xs[0] + width / 2, mid), xytext=(xs[0] + width / 2 + 0.12, ty),
                ha='left', va='center', color='#3d3d3a',
                arrowprops=dict(arrowstyle='-', color='#8a8980', lw=0.5, shrinkA=0, shrinkB=0))

ax.set_xticks(xs, [b[0] for b in bars])
ax.set_xlim(-0.45, xs[-1] + 0.45)
ax.set_ylim(0, 100)
ax.set_yticks([0, 25, 50, 75, 100])
ax.set_ylabel('Share of across-video variance in z (%)')
ax.tick_params(axis='x', length=0)
for side in ('top', 'right'):
    ax.spines[side].set_visible(False)
# legend top-to-bottom in the same order as the stack (top segment first)
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles[::-1], labels[::-1], loc='center left', bbox_to_anchor=(1.02, 0.5),
          frameon=False, handlelength=1, labelspacing=0.8)

save_figure(fig, "./results/supp_fig_1.pdf", SINGLE_COL_MM, relayout=fig.tight_layout)
plt.close(fig)
