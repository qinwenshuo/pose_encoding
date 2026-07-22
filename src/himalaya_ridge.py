from __future__ import annotations
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from math import ceil
from pathlib import Path
from scipy.stats import pearsonr
from sklearn.metrics import r2_score
from himalaya.ridge import GroupRidgeCV
from src.config import ALPHAS, CV_SPLITS
# from himalaya.scoring import r2_score as hima_r2
from himalaya.scoring import correlation_score 
from joblib import Parallel, delayed, cpu_count

from src.encoding import get_top_sota_layers, get_target_ratings, get_pose_features, align_multiple_vars
from src.data_utils import save_pickle, load_pickle
from src.config import SOTA_PLOT_NAME

def _is_empty_block(Z):
    return (Z is None) or (np.asarray(Z).size == 0) or (np.asarray(Z).ndim == 2 and np.asarray(Z).shape[1] == 0)

def _ensure_2d(X):
    X = np.asarray(X)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    return X

def _clean_array(X, name="array"):
    """Remove inf/nan values from array and warn if found."""
    X = np.asarray(X, dtype=np.float64)
    
    # Check for inf/nan
    has_inf = np.isinf(X).any()
    has_nan = np.isnan(X).any()
    
    if has_inf or has_nan:
        inf_count = np.isinf(X).sum()
        nan_count = np.isnan(X).sum()
        print(f"[WARNING] {name} contains {nan_count} NaNs and {inf_count} Infs - replacing with 0")
        
        # Replace inf with large finite values
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    
    return X

def _standardize_train_apply(X_tr, X_te):
    """Standardize with robust handling of zero variance and inf/nan."""
    X_tr = _clean_array(X_tr, "X_train")
    X_te = _clean_array(X_te, "X_test")
    
    mu = np.mean(X_tr, axis=0, keepdims=True)
    sd = np.std(X_tr, axis=0, keepdims=True)
    
    # Avoid division by zero
    sd[sd == 0] = 1.0
    
    X_tr_std = (X_tr - mu) / sd
    X_te_std = (X_te - mu) / sd
    
    # Final safety check after standardization
    X_tr_std = _clean_array(X_tr_std, "X_train_standardized")
    X_te_std = _clean_array(X_te_std, "X_test_standardized")
    
    return X_tr_std, X_te_std, mu, sd

def _center_train_apply(y_tr, y_te):
    """Center and scale with robust handling."""
    y_tr = _clean_array(y_tr, "y_train")
    y_te = _clean_array(y_te, "y_test")
    
    y_mu = float(np.mean(y_tr))
    y_sd = float(np.std(y_tr))
    if y_sd == 0:
        y_sd = 1.0
    
    y_tr_scaled = (y_tr - y_mu) / y_sd
    y_te_scaled = (y_te - y_mu) / y_sd
    
    # Final safety check
    y_tr_scaled = _clean_array(y_tr_scaled, "y_train_scaled")
    y_te_scaled = _clean_array(y_te_scaled, "y_test_scaled")
    
    return y_tr_scaled, y_te_scaled, y_mu, y_sd


def group_ridge(
    predictor1, predictor2, target,
    alphas=ALPHAS,
    n_splits=CV_SPLITS,
    verbose=True,
):
    """
    Himalaya-based rewrite with robust NaN/Inf handling.
    Now strictly requires Z to have >= 1 feature column.
    Raises:
        ValueError: if predictor2['train'] or predictor2['test'] is empty (shape (*, 0)).
    """

    # Coerce arrays
    X1_tr = _ensure_2d(predictor1['train'])
    X1_te = _ensure_2d(predictor1['test'])
    Z_tr  = _ensure_2d(predictor2['train'])
    Z_te  = _ensure_2d(predictor2['test'])
    y_tr  = np.asarray(target['train']).ravel()
    y_te  = np.asarray(target['test']).ravel()

    # Enforce non-empty Z
    if _is_empty_block(Z_tr) or _is_empty_block(Z_te):
        raise ValueError(
            f"`predictor2` (Z) has no features: "
            f"train shape={Z_tr.shape}, test shape={Z_te.shape}. "
            f"Provide at least one column in Z."
        )
    
    if verbose:
        print(f'[LOGGING] Shapes X1: {X1_tr.shape}/{X1_te.shape}  Z: {Z_tr.shape}/{Z_te.shape}')
        
        # Diagnostic info about data quality
        for name, arr in [("X1_tr", X1_tr), ("X1_te", X1_te), 
                          ("Z_tr", Z_tr), ("Z_te", Z_te),
                          ("y_tr", y_tr), ("y_te", y_te)]:
            n_nan = np.isnan(arr).sum()
            n_inf = np.isinf(arr).sum()
            if n_nan > 0 or n_inf > 0:
                print(f"[LOGGING] {name}: {n_nan} NaNs, {n_inf} Infs")

    # Standardize on TRAIN, apply to TEST (now with cleaning)
    X1_tr_s, X1_te_s, X1_mu, X1_sd = _standardize_train_apply(X1_tr, X1_te)
    Z_tr_s,  Z_te_s,  Z_mu,  Z_sd  = _standardize_train_apply(Z_tr, Z_te)
    y_tr_s,  y_te_s,  y_mu,  y_sd  = _center_train_apply(y_tr, y_te)

    Y_tr = y_tr_s[:, None]
    Y_te = y_te_s[:, None]

    # Final verification before fitting
    for name, arr in [("X1_tr_s", X1_tr_s), ("Z_tr_s", Z_tr_s), ("Y_tr", Y_tr)]:
        if not np.isfinite(arr).all():
            raise ValueError(f"{name} still contains non-finite values after cleaning!")

    # GroupRidgeCV with two groups [X1, Z]
    if verbose:
        print('[LOGGING] Tuning BOTH with Himalaya.GroupRidgeCV (random_search)')

    group_model = GroupRidgeCV(
        groups="input",
        solver="random_search",
        fit_intercept=False,
        solver_params=dict(
            n_iter=200, 
            alphas=alphas,
            concentration=[0.1, 1.0],
            score_func=correlation_score),
    )
    group_model.fit([X1_tr_s, Z_tr_s], Y_tr)
    yhat_both_te = group_model.predict([X1_te_s, Z_te_s]).ravel()
    r2_both = r2_score(y_te_s, yhat_both_te)
    try:
        r_both = float(pearsonr(y_te_s, yhat_both_te)[0])
    except Exception:
        r_both = float('nan')

    # Recover per-group effective lambdas
    deltas = np.asarray(group_model.deltas_).reshape(-1)      # length 2
    alpha_global = float(np.atleast_1d(group_model.best_alphas_)[0])
    lam_groups = alpha_global * np.exp(-deltas)               # [lam_X, lam_Z]
    lam1_both, lam2_both = float(lam_groups[0]), float(lam_groups[1])

    return {
        "score_r2":   float(r2_both),
        "score_r":    float(r_both),

        "y_hat":    yhat_both_te,
        "lambda1_both":   float(lam1_both),
        "lambda2_both":   float(lam2_both),

        "standardization": {
            "X1_mu": X1_mu, "X1_sd": X1_sd,
            "Z_mu":  Z_mu,  "Z_sd":  Z_sd,
            "y_mu":  y_mu,  "y_sd":  y_sd,
        },
    }



# Optional: tqdm is fine inside a worker, but keep it lightweight
from tqdm import tqdm

def _process_one_pair(target, pose_feature, task_id, max_tasks, overwrite):
    """Worker: run the original inner logic for a single (target, pose_feature)."""
    print(f"\n🎯 Running target: '{target}' | pose_feature: '{pose_feature}'")

    output_dir = f"experiments/ridge_results/{pose_feature.replace(' ', '_')}/{target.replace(' ', '_')}"
    os.makedirs(output_dir, exist_ok=True)

    # Load data
    target_data = get_target_ratings(target)
    print(f"📦 Loading pose feature: '{pose_feature}'...")
    arrow_data = get_pose_features(pose_feature)

    print("📦 Loading SOTA model representations...")
    all_sota_data = get_top_sota_layers(
        target=target,
        top_n_models=-1,
        top_k_layers=1,
        n_jobs=-1,
        show_progress=False
    )

    # Divide SOTA models by (task_id, max_tasks); DO NOT parallelize here.
    total_models = len(all_sota_data)
    chunk_size = ceil(total_models / max_tasks) if max_tasks > 0 else total_models
    start_idx = max(0, (task_id - 1) * chunk_size)
    end_idx = min(task_id * chunk_size, total_models)
    subset_sota_data = all_sota_data[start_idx:end_idx]

    print(f"🧩 Task {task_id}: processing models {start_idx}-{max(0, end_idx - 1)} "
          f"({len(subset_sota_data)} total) for '{target}' × '{pose_feature}'.")

    for sota_data in tqdm(subset_sota_data, desc=f"Task {task_id} | {target} × {pose_feature}", leave=False):
        model_name = sota_data['model_name']
        output_path = os.path.join(output_dir, f"{model_name}.pkl")

        # Skip if already computed
        if not overwrite and os.path.exists(output_path):
            print(f"[SKIP] {model_name}, already processed: {output_path}")
            continue

        # Align and run grouped ridge
        sota_aligned, arrow_aligned, target_aligned = align_multiple_vars(
            sota_data, arrow_data, target_data
        )
        results = group_ridge(sota_aligned, arrow_aligned, target_aligned)

        save_pickle(results, output_path)
        print(f"[DONE] {model_name}")

    print(f"✅ Done: target '{target}' | pose_feature '{pose_feature}'")


def get_grouped_ridge_scores(
    targets,
    pose_features,
    task_id,
    max_tasks,
    overwrite: bool = False,
    n_jobs: int | None = None,
    prefer: str = "processes",   # "threads" also works; ridge is usually CPU-bound -> processes
    verbose: int = 0,
):
    """
    Parallelizes ONLY over (target, pose_feature) pairs.
    SOTA models remain sequential inside each worker.
    Also prints a parallelism report.
    """
    # Prepare all (target, pose_feature) pairs
    pairs = [(t, p) for t in targets for p in pose_features]
    total_pairs = len(pairs)

    # Decide n_jobs
    available = cpu_count()
    if n_jobs is None or n_jobs <= 0:
        n_jobs = min(available, max(1, total_pairs))

    # Report parallelism
    print("====== Parallelism Report ======")
    print(f"Total (target × pose_feature) pairs to run: {total_pairs}")
    print(f"System parallel capacity (cpu_count):       {available}")
    print(f"Using n_jobs (parallel workers):            {n_jobs}")
    print("================================")

    # Run
    Parallel(n_jobs=n_jobs, backend="loky" if prefer == "processes" else "threading", verbose=verbose)(
        delayed(_process_one_pair)(t, p, task_id, max_tasks, overwrite)
        for (t, p) in pairs
    )



import os
import pandas as pd
from joblib import Parallel, delayed

def collect_grouped_ridge_scores(pose_features, targets, n_jobs=-1, verbose=0):
    """
    Collect grouped ridge scores in parallel using joblib.
    Each .pkl file is loaded and processed independently.
    
    Parameters
    ----------
    pose_features : list[str]
    targets : list[str]
    n_jobs : int, default=-1
        Number of parallel workers. -1 uses all available cores.
    verbose : int, default=0
        Joblib verbosity.
    """
    # Gather all (file_path, target) pairs
    print(f'[LOGGING] Loading the grouped ridge scores on targets({targets}) and pose_features({pose_features})')
    tasks = []
    for pose_feature in pose_features:
        for target in targets:
            folder = f"experiments/ridge_results/{pose_feature.replace(' ', '_')}/{target.replace(' ', '_')}"
            if not os.path.isdir(folder):
                continue
            for fname in os.listdir(folder):
                if fname.endswith(".pkl"):
                    file_path = os.path.join(folder, fname)
                    tasks.append((file_path, target, pose_feature))

    def process_file(file_path, target, pose_feature):
        """Load and extract ridge results from one pickle."""
        data = load_pickle(file_path)
        return {
            'x': f"{SOTA_PLOT_NAME} + \n3D social pose features",
            'y': target,
            'pose_feature': pose_feature,
            'score_r': data['score_r'],
            'y_hat': data['y_hat']
        }

    # Parallel processing 
    results = Parallel(n_jobs=n_jobs, verbose=verbose)(
        delayed(process_file)(file_path, target, pose_feature)
        for file_path, target, pose_feature in tasks
    )

    return pd.DataFrame(results)