import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from joblib import Parallel, delayed
from scipy.stats import permutation_test, pearsonr
from src.data_utils import get_target_ratings, get_pose_features, align_multiple_vars, RANDOM
from src.config import RATING_OF_INTEREST

def _nan_pearsonr_1d(x, y):
    """NaN-safe Pearson r for 1D arrays; returns NaN if <2 valid points or zero variance."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 2:
        return np.nan
    xm, ym = x[mask] - x[mask].mean(), y[mask] - y[mask].mean()
    denom = np.sqrt((xm ** 2).sum() * (ym ** 2).sum())
    return float(np.dot(xm, ym) / denom) if denom > 0 else np.nan


def paired_class_diff_perm(class_A_r, class_B_r, tails):
    # score_a, score_b: shape (n,) — paired correlation scores per unit
    # Hypothesis (one-tailed): B > A  <=>  mean(A - B) < 0
    def stat(a, b):
        return np.mean(a - b)

    res = permutation_test(
        data=(np.asarray(class_A_r, float), np.asarray(class_B_r, float)),
        statistic=stat,
        permutation_type='samples',   # swap labels within each pair
        alternative=tails,            # mean(A-B) less than 0 ⇒ B better
        n_resamples=5000,             # or np.inf for exact if n is tiny
        rng=RANDOM                    # reproducible
    )

    return {
        "diff_obs": float(res.statistic),     # mean(A) - mean(B)
        "p_value": float(res.pvalue),
        "n_perm": int(res.null_distribution.size)
    }



# ---------------------------------------------------------------------------
# Plotting helper
# ---------------------------------------------------------------------------

def plot_null_diff(result: dict, save_dir: str = "results") -> str:
    """
    Plot the permutation null distribution of (mean_r_A - mean_r_B) and
    mark the observed difference.

    Parameters
    ----------
    result   : dict returned by two_classes_perm()
    save_dir : directory in which to save the figure

    Returns
    -------
    str  – full path of the saved figure
    """
    os.makedirs(save_dir, exist_ok=True)

    # ---- resolve output path (temp.png → temp1.png → temp2.png …) ----
    base = os.path.join(save_dir, "temp.png")
    if not os.path.exists(base):
        out_path = base
    else:
        k = 1
        while True:
            candidate = os.path.join(save_dir, f"temp{k}.png")
            if not os.path.exists(candidate):
                out_path = candidate
                break
            k += 1

    # ---- unpack ----
    diff_null = np.asarray(result["diff_null"], float)
    diff_obs  = float(result["diff_obs"])
    mean_r_A  = float(result["mean_r_A"])
    mean_r_B  = float(result["mean_r_B"])
    p_value   = float(result["p_value"])
    tails     = result.get("tails", "two-sided")

    # ---- figure ----
    fig, ax = plt.subplots(figsize=(8, 5))

    # histogram of null distribution
    n_bins = min(60, max(20, len(diff_null) // 80))
    ax.hist(
        diff_null, bins=n_bins,
        color="#4C72B0", edgecolor="white", linewidth=0.4,
        alpha=0.85, label="Null distribution",
        zorder=2,
    )

    # shading: which tail(s) count toward p-value
    xlim_lo, xlim_hi = ax.get_xlim()
    if tails == "two-sided":
        thresh = np.abs(diff_obs)
        ax.axvspan(thresh,  max(xlim_hi, thresh * 1.1),
                   color="#DD4444", alpha=0.18, label="|diff| ≥ |obs|", zorder=1)
        ax.axvspan(min(xlim_lo, -thresh * 1.1), -thresh,
                   color="#DD4444", alpha=0.18, zorder=1)
    elif tails == "greater":
        ax.axvspan(diff_obs, max(xlim_hi, diff_obs * 1.1),
                   color="#DD4444", alpha=0.18, label="diff ≥ obs", zorder=1)
    else:  # less
        ax.axvspan(min(xlim_lo, diff_obs * 1.1), diff_obs,
                   color="#DD4444", alpha=0.18, label="diff ≤ obs", zorder=1)

    # observed statistic line
    ax.axvline(diff_obs, color="#DD4444", linewidth=2.0,
               linestyle="--", label=f"Observed diff = {diff_obs:.4f}", zorder=4)

    # null mean reference
    ax.axvline(np.nanmean(diff_null), color="#555555", linewidth=1.2,
               linestyle=":", alpha=0.7, label="Null mean", zorder=3)

    # labels & aesthetics
    ax.set_xlabel("mean r(A, y) − mean r(B, y)", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title(
        f"Permutation null distribution\n"
        f"mean r(A) = {mean_r_A:.4f},  mean r(B) = {mean_r_B:.4f}  |  "
        f"p = {p_value:.4f}  ({tails})",
        fontsize=11,
    )
    ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.legend(fontsize=9, framealpha=0.9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()

    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[plot_null_diff] saved → {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# Main permutation test
# ---------------------------------------------------------------------------

def two_classes_perm(
    y, y_hat_A, y_hat_B,
    n_perm=5000, n_jobs=-1, random_state=RANDOM, tails='two-sided',
    plot=False, plot_dir="results",
):
    """
    Permutation test: is mean(corr(A_i, y)) - mean(corr(B_j, y)) significantly != 0?

    Null distribution built by permuting y (shared index across all models),
    which breaks any real signal while preserving inter-model correlation structure
    within and across groups (A and B).

    Parameters
    ----------
    y        : (n,)
    y_hat_A  : (nA, n)
    y_hat_B  : (nB, n) or (n,) or (1, n)
    n_perm   : int
    n_jobs   : int
    random_state : None | int | np.random.Generator
    tails    : {'less', 'greater', 'two-sided'}
    plot     : bool
        If True, call plot_null_diff() and save the figure automatically.
    plot_dir : str
        Directory passed to plot_null_diff() (default 'results').

    Returns
    -------
    dict with keys:
        mean_r_A, mean_r_B, diff_obs, p_value, diff_null, rA, rB, tails,
        [plot_path]   ← added only when plot=True
    """
    def _corrs_against_y(preds, y_vec):
        return np.array([_nan_pearsonr_1d(row, y_vec) for row in np.asarray(preds, float)])

    # ---- inputs & shapes ----
    y = np.asarray(y, float).ravel()
    A = np.asarray(y_hat_A, float)
    B = np.asarray(y_hat_B, float)
    if B.ndim == 1:
        B = B[None, :]
    n = y.shape[0]
    assert A.ndim == 2 and B.ndim == 2, "y_hat_A and y_hat_B must be 2D after coercion."
    assert A.shape[1] == n and B.shape[1] == n, "Shape mismatch: second dim must match y."
    if tails not in ('less', 'greater', 'two-sided'):
        raise ValueError("tails must be 'less', 'greater', or 'two-sided'.")

    # ---- observed statistic ----
    rA = _corrs_against_y(A, y)   # (nA,)
    rB = _corrs_against_y(B, y)   # (nB,)
    diff_obs = float(np.nanmean(rA) - np.nanmean(rB))

    # ---- reproducible seeds ----
    if isinstance(random_state, np.random.Generator):
        base_ss = np.random.SeedSequence(random_state.bit_generator.state["state"]["state"])
    elif random_state is None:
        base_ss = np.random.SeedSequence()
    else:
        base_ss = np.random.SeedSequence(int(random_state))
    seeds = base_ss.spawn(n_perm)

    # ---- single permutation: shuffle y once, reuse for all models ----
    def _perm_once(seed):
        rng = np.random.default_rng(seed)
        y_perm = y[rng.permutation(n)]
        rA_p = _corrs_against_y(A, y_perm)
        rB_p = _corrs_against_y(B, y_perm)
        return float(np.nanmean(rA_p) - np.nanmean(rB_p))

    diff_null = np.asarray(
        Parallel(n_jobs=n_jobs, backend="loky")(delayed(_perm_once)(s) for s in seeds),
        dtype=float,
    )

    # ---- p-value ----
    m = diff_null.size
    if tails == 'less':
        p_value = (np.sum(diff_null <= diff_obs) + 1.0) / (m + 1.0)
    elif tails == 'greater':
        p_value = (np.sum(diff_null >= diff_obs) + 1.0) / (m + 1.0)
    else:  # two-sided: |null| >= |observed|
        p_value = (np.sum(np.abs(diff_null) >= np.abs(diff_obs)) + 1.0) / (m + 1.0)

    result = {
        "mean_r_A": float(np.nanmean(rA)),
        "mean_r_B": float(np.nanmean(rB)),
        "diff_obs": diff_obs,
        "p_value":  float(p_value),
        "diff_null": diff_null,
        "rA": rA,
        "rB": rB,
        "tails": tails,
    }

    # ---- optional plot ----
    if plot:
        result["plot_path"] = plot_null_diff(result, save_dir=plot_dir)

    return result


def stars_from_p(p, thresholds=(0.05, 0.01, 0.001)):
    if p <= thresholds[2]: return '***'
    if p <= thresholds[1]: return '**'
    if p <= thresholds[0]: return '*'
    return ''


def _normalize_sig_pairs_dicts(sig_pairs):
    """
    Validate and normalize sig_pairs (list of dicts). Each dict must have:
      'a', 'b' (str class names), 'testing_type', 'tails'.
    Returns a list of normalized dicts.
    """
    if sig_pairs is None:
        raise TypeError("sig_pairs is empty but must be a list of dicts")

    norm = []
    for i, d in enumerate(sig_pairs):
        if not isinstance(d, dict):
            raise TypeError(
                f"sig_pairs must be a list of dicts; entry {i} has type {type(d)} with value {d}"
            )
        required_keys = ["a", "b", "testing_type", "tails"]
        missing = [k for k in required_keys if k not in d]
        if missing:
            raise ValueError(f"sig_pairs[{i}] missing required keys: {missing}; got {d}")
        a, b = d["a"], d["b"]
        if not isinstance(a, str) or not isinstance(b, str):
            raise ValueError(f"sig_pairs[{i}] must have string 'a' and 'b'; got {d}")
        norm.append({"a": a, "b": b, "testing_type": d["testing_type"], "tails": d["tails"]})

    return norm


def bh_fdr_correction(p_values):
    """
    Benjamini-Hochberg FDR correction.

    Parameters
    ----------
    p_values : list or array of raw p-values (length m)

    Returns
    -------
    p_adjusted : np.ndarray of BH-adjusted p-values, same order as input
    """
    p = np.asarray(p_values, dtype=float)
    m = len(p)
    if m == 0:
        return np.array([])

    order = np.argsort(p)
    ranked = np.empty(m, dtype=float)
    # BH adjusted: p_adj[k] = p[k] * m / (k+1) for rank k (0-indexed)
    ranked[order] = p[order] * m / (np.arange(m) + 1)
    # enforce monotonicity from largest rank downward
    for i in range(m - 2, -1, -1):
        ranked[order[i]] = min(ranked[order[i]], ranked[order[i + 1]])
    return np.clip(ranked, 0.0, 1.0)


def run_significance_tests(
    results_df,
    y_to_plot,
    sig_pairs,
    fdr_correction=True,
    alpha_thresholds=(0.05, 0.01, 0.001),
):
    """
    Run all (pair × target) permutation tests, optionally apply BH-FDR correction,
    and return a ready-to-use result dict for encoding_bar_with_points.

    Parameters
    ----------
    results_df       : DataFrame used for plotting
    y_to_plot        : list of target names (defines the FDR family)
    sig_pairs        : list of dicts, each with keys 'a', 'b', 'testing_type', 'tails'
    fdr_correction   : if True, apply Benjamini-Hochberg FDR across all tests
    alpha_thresholds : thresholds for *, **, *** stars (applied to corrected p-values)

    Returns
    -------
    dict mapping (pair_idx, target_name) -> {
        'p_raw'       : float,
        'p_corrected' : float,
        'stars'       : str,   # '', '*', '**', or '***'
        'pair'        : dict,  # the normalized pair spec
    }
    """
    pair_specs = _normalize_sig_pairs_dicts(sig_pairs)

    test_keys, p_raws = [], []
    for pair_idx, spec in enumerate(pair_specs):
        for y in y_to_plot:
            p = significance_testing(
                results_df=results_df,
                ys=[y],
                class_A=spec["a"],
                class_B=spec["b"],
                testing_type=spec["testing_type"],
                tails=spec["tails"],
            )[0]
            test_keys.append((pair_idx, y))
            p_raws.append(p)

    if fdr_correction:
        # BH correction applied per pair (each pair is its own family of len(y_to_plot) tests)
        p_corrected = np.asarray(p_raws, dtype=float)
        from collections import defaultdict
        pair_to_indices = defaultdict(list)
        for i, (pair_idx, _) in enumerate(test_keys):
            pair_to_indices[pair_idx].append(i)
        for pair_idx, indices in pair_to_indices.items():
            spec = pair_specs[pair_idx]
            raw_family = [p_raws[i] for i in indices]
            corrected_family = bh_fdr_correction(raw_family)
            print(f"[FDR] BH correction for '{spec['a']} vs {spec['b']}' over {len(raw_family)} targets:")
            for i, p_r, p_c in zip(indices, raw_family, corrected_family):
                _, y = test_keys[i]
                p_corrected[i] = p_c
                print(f"  [{y}]  p_raw={p_r:.4g}  p_BH={p_c:.4g}")
    else:
        p_corrected = np.asarray(p_raws, dtype=float)

    test_results = {}
    for (pair_idx, y), p_raw, p_corr in zip(test_keys, p_raws, p_corrected):
        stars = stars_from_p(float(p_corr), thresholds=alpha_thresholds)
        test_results[(pair_idx, y)] = {
            "p_raw":       float(p_raw),
            "p_corrected": float(p_corr),
            "stars":       stars,
            "pair":        pair_specs[pair_idx],
        }

    return test_results


# ---------------------------------------------------------------------------
# Grid significance helpers (used by plot_data_grid)
# ---------------------------------------------------------------------------

def _perm_corr_pvalue(x, y, n_perm=5000, two_sided=True, random_state=None):
    """Permutation test for Pearson r; returns (r_obs, p_value)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if x.size < 3:
        return np.nan, np.nan
    rng = np.random.default_rng(random_state)
    x0, y0 = x - x.mean(), y - y.mean()
    denom = np.sqrt(np.sum(x0 * x0) * np.sum(y0 * y0))
    if denom == 0:
        return np.nan, np.nan
    r_obs = float(np.sum(x0 * y0) / denom)
    perm_rs = np.array([np.sum(x0 * rng.permutation(y0)) / denom for _ in range(n_perm)])
    if two_sided:
        p_val = (np.sum(np.abs(perm_rs) >= abs(r_obs)) + 1) / (n_perm + 1)
    else:
        p_val = (np.sum(perm_rs >= r_obs) + 1) / (n_perm + 1)
    return r_obs, float(p_val)


def _perm_corr_diff_swap(human, feat2d, feat3d, n_perm=5000, seed=0):
    """Permutation test for r(3D,human) > r(2D,human) via label-swap."""
    rng = np.random.default_rng(seed)
    human  = np.asarray(human)
    feat2d = np.asarray(feat2d)
    feat3d = np.asarray(feat3d)
    r3d_obs = pearsonr(feat3d, human)[0]
    r2d_obs = pearsonr(feat2d, human)[0]
    d_obs = r3d_obs - r2d_obs
    d_perm = np.empty(n_perm)
    for i in range(n_perm):
        swap = rng.random(len(human)) < 0.5
        f2 = feat2d.copy(); f3 = feat3d.copy()
        f2[swap] = feat3d[swap]; f3[swap] = feat2d[swap]
        d_perm[i] = pearsonr(f3, human)[0] - pearsonr(f2, human)[0]
    p_one_sided = (np.sum(d_perm >= d_obs) + 1) / (n_perm + 1)
    p_two_sided = (np.sum(np.abs(d_perm) >= abs(d_obs)) + 1) / (n_perm + 1)
    return {
        "r3d_obs": r3d_obs, "r2d_obs": r2d_obs, "d_obs": d_obs,
        "p_one_sided": p_one_sided, "p_two_sided": p_two_sided, "d_perm": d_perm,
    }


def run_grid_significance_tests(sota_df, targets, n_perm=5000, random_state=0):
    """
    Run and BH-FDR-correct all significance tests for plot_data_grid.

    Three separate families (5 tests each):
      - 3D social pose features r-tests (one per target)
      - 2D social pose features r-tests (one per target)
      - 3D vs 2D difference tests      (one per target)

    Returns
    -------
    dict with two keys:
        'corr': {(pose_feature, target): {'r': float, 'p_raw': float, 'p_corrected': float}}
        'diff': {target:                 {'p_raw': float, 'p_corrected': float}}
    """
    raw_corr = {}   # (pose_feature, target) -> (r_val, p_raw)
    raw_diff = {}   # target -> p_raw

    for tgt in targets:
        df2 = sota_df[
            (sota_df['pose_feature'] == '2D social pose features') &
            (sota_df['target'] == tgt)
        ].sort_values("model_name").reset_index(drop=True)
        df3 = sota_df[
            (sota_df['pose_feature'] == '3D social pose features') &
            (sota_df['target'] == tgt)
        ].sort_values("model_name").reset_index(drop=True)

        for df in [df3, df2]:
            pf = df['pose_feature'].iloc[0]
            r, p = _perm_corr_pvalue(
                df["pose_r"].to_numpy(), df["target_r"].to_numpy(),
                n_perm=n_perm, random_state=random_state,
            )
            raw_corr[(pf, tgt)] = (r, p)

        diff = _perm_corr_diff_swap(
            df2["target_r"].to_numpy(),
            df2["pose_r"].to_numpy(),
            df3["pose_r"].to_numpy(),
            n_perm=n_perm, seed=random_state,
        )
        raw_diff[tgt] = diff["p_one_sided"]

    # BH per pose feature
    corr_results = {}
    for pf in ['3D social pose features', '2D social pose features']:
        p_raw_list = [raw_corr[(pf, t)][1] for t in targets]
        p_bh_list  = bh_fdr_correction(p_raw_list)
        print(f"[FDR] BH correction for '{pf}' r-tests over {len(targets)} targets:")
        for t, p_r, p_c in zip(targets, p_raw_list, p_bh_list):
            r_val = raw_corr[(pf, t)][0]
            corr_results[(pf, t)] = {"r": r_val, "p_raw": float(p_r), "p_corrected": float(p_c)}
            print(f"  [{t}]  p_raw={p_r:.4g}  p_BH={p_c:.4g}")

    # BH for difference tests
    diff_raw  = [raw_diff[t] for t in targets]
    diff_bh   = bh_fdr_correction(diff_raw)
    print(f"[FDR] BH correction for 3D vs 2D difference tests over {len(targets)} targets:")
    diff_results = {}
    for t, p_r, p_c in zip(targets, diff_raw, diff_bh):
        diff_results[t] = {"p_raw": float(p_r), "p_corrected": float(p_c)}
        print(f"  [{t}]  p_raw={p_r:.4g}  p_BH={p_c:.4g}")

    return {"corr": corr_results, "diff": diff_results}


def significance_testing(results_df, ys, class_A, class_B, testing_type, tails):
    def collect_r_score(x_name, y_name, df_results):
        mask = (df_results["x"] == x_name) & (df_results["y"] == y_name)
        filtered = df_results.loc[mask]
        r_array = np.ravel(np.array([np.ravel(y) for y in filtered["score_r"]]))
        return r_array
    def collect_y_hat(x_name, y_name, df_results):
        mask = (df_results["x"] == x_name) & (df_results["y"] == y_name)
        filtered = df_results.loc[mask]
        y_hat_list = [np.ravel(y) for y in filtered["y_hat"]]
        # print('===========')
        # print(x_name, y_name)
        # print(filtered.head())
        # print(f'y hat list {y_hat_list}')
        y_hat_2d = np.vstack(y_hat_list)
        return y_hat_2d

    p_values = []
    # print(ys)
    for y in ys:
        class_A_r = collect_r_score(class_A, y, results_df)
        class_B_r = collect_r_score(class_B, y, results_df)
        if testing_type == 'paired_multi':
            stat = paired_class_diff_perm(class_A_r, class_B_r, tails=tails)
            # Compute the percentage of units where model B > model A
            n_total = len(class_A_r)
            n_B_higher = np.sum(class_B_r > class_A_r)
            pct_B_higher = (n_B_higher / n_total) * 100
            print(f"[LOGGING] {class_B} > {class_A} in {n_B_higher}/{n_total} cases ({pct_B_higher:.2f}%)")
            print(f"[LOGGING] Mean difference ({class_A} - {class_B}): {stat['diff_obs']:.4f}")

        elif testing_type == 'single' or testing_type == 'non_paired_multi':
            class_A_y_hat = collect_y_hat(class_A, y, results_df)
            class_B_y_hat = collect_y_hat(class_B, y, results_df)
            if y in RATING_OF_INTEREST:
                true_y = get_target_ratings(y)
            else: 
                true_y = get_pose_features(y)
            true_y = align_multiple_vars(true_y)[0]['test']
            stat = two_classes_perm(y=true_y, y_hat_A=class_A_y_hat, y_hat_B=class_B_y_hat, tails=tails)
            if class_B_r.size == 1:
                b_value = class_B_r.item()
                n_total = len(class_A_r)
                n_B_higher = np.sum(b_value > class_A_r)
                pct_B_higher = (n_B_higher / n_total) * 100
                print(f"[LOGGING] {class_B} > {class_A} in {n_B_higher}/{n_total} cases ({pct_B_higher:.2f}%)")
            print(f"[LOGGING] Mean difference ({class_A} - {class_B}): {stat['diff_obs']:.4f}")

        else:
            raise ValueError(f'Unrecognized testing type: {testing_type}')
        # print class means + p-value for this comparison
        print(f'[LOGGING] Hypothesis: if {class_A} is {tails} than {class_B} on predicting {y}')
        print(f'[LOGGING] p_value: {round(stat["p_value"], 5)}   {stars_from_p(stat["p_value"])}')
        print(f'[LOGGING] testing type: {testing_type}')
        print(f'[LOGGING] {class_A} number : {len(class_A_r)}, {class_B} number: {len(class_B_r)}')
        p_values.append(stat['p_value'])
        mean_a = np.mean(class_A_r)
        mean_b = np.mean(class_B_r)
        print(f"[LOGGING] {class_A} mean={mean_a:.4f} vs {class_B} mean={mean_b:.4f}; delta r: a-b = {mean_a - mean_b:.4f} b-a = {mean_b - mean_a:.4f}")
        print('='*50)

    return p_values