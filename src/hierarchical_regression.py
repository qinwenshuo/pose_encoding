import os
from math import ceil
from multiprocessing import cpu_count
from pathlib import Path

import numpy as np
from tqdm import tqdm
from itertools import product
from scipy.stats import pearsonr
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from joblib import Parallel, delayed
from src.encoding import get_top_sota_layers, get_target_ratings, get_pose_features, align_multiple_vars
from src.data_utils import save_pickle, load_pickle, get_sota_model_layers
from src.config import SOTA_PLOT_NAME, SOTA_MODEL_PATH
from src.config import ALPHAS, CV_SPLITS, RANDOM

ALPHA_CV_SPLITS = 4

# ==========================================================
# Global (target × pose × model) partition + grid-parallel
# ==========================================================

# ----------------------------
# Ridge helpers (Z never empty)
# ----------------------------
# --- unchanged helpers ---
def _fit_block_ridge_closed_form(X1, Z, y, lam1, lam2, jitter=0.0):
    n, p1 = X1.shape
    p2 = Z.shape[1]
    X = np.concatenate([X1, Z], axis=1)

    D = np.zeros((p1 + p2, p1 + p2), dtype=float)
    if lam1 > 0: D[:p1, :p1] = lam1 * np.eye(p1)
    if lam2 > 0: D[p1:, p1:] = lam2 * np.eye(p2)

    XtX = X.T @ X
    if jitter and jitter > 0:
        XtX = XtX + jitter * np.eye(X.shape[1])
    A = XtX + D
    Xty = X.T @ y
    try:
        beta = np.linalg.solve(A, Xty)
    except np.linalg.LinAlgError:
        beta = np.linalg.pinv(A) @ Xty
    return beta[:p1], beta[p1:]


def _predict_block(X1, Z, b1, b2):
    return X1 @ b1 + Z @ b2


# --- parallelize over CV splits (not grid) ---

def _cv_score_block_ridge_raw_with_splits(
    X1_raw, Z_raw, y_raw, lam1, lam2, splits, jitter=0.0,
    n_jobs_cv=-1, cv_backend="threads"
):
    """Mean R^2 across folds; each fold standardized on its train split only."""
    def _z(X, mu=None, sd=None):
        if mu is None:
            mu = np.mean(X, axis=0, keepdims=True)
        if sd is None:
            sd = np.std(X, axis=0, keepdims=True)
            sd[sd == 0] = 1.0
        return (X - mu) / sd, mu, sd

    def _score_one_fold(tr_idx, va_idx):
        X1_tr, X1_va = X1_raw[tr_idx], X1_raw[va_idx]
        Z_tr,  Z_va  = Z_raw[tr_idx],  Z_raw[va_idx]
        y_tr,  y_va  = y_raw[tr_idx],  y_raw[va_idx]

        # Per-fold standardization (fit on TRAIN only)
        X1_tr_s, x1_mu, x1_sd = _z(X1_tr);  X1_va_s, _, _ = _z(X1_va, x1_mu, x1_sd)
        Z_tr_s,  z_mu, z_sd  = _z(Z_tr);    Z_va_s,  _, _ = _z(Z_va,  z_mu,  z_sd)
        y_tr_s,  y_mu, y_sd  = _z(y_tr.reshape(-1, 1))
        y_va_s,  _, _        = _z(y_va.reshape(-1, 1), y_mu, y_sd)
        y_tr_s = y_tr_s.ravel(); y_va_s = y_va_s.ravel()

        b1, b2 = _fit_block_ridge_closed_form(X1_tr_s, Z_tr_s, y_tr_s, lam1, lam2, jitter)
        yhat_va = _predict_block(X1_va_s, Z_va_s, b1, b2)
        return r2_score(y_va_s, yhat_va)

    fold_scores = Parallel(n_jobs=n_jobs_cv, backend=("threading" if cv_backend=="threads" else "loky"))(
        delayed(_score_one_fold)(tr_idx, va_idx) for tr_idx, va_idx in splits
    )
    return float(np.mean(fold_scores))


# --- group_ridge with sequential grid & parallel CV ---
def group_ridge(
    predictor1, predictor2, target,
    lambdas1=ALPHAS,
    lambdas2=ALPHAS,
    n_splits=ALPHA_CV_SPLITS,
    random_state=RANDOM,
    jitter=0.0,
    n_jobs_cv=-1,          # <--- parallelism here
    cv_backend="threads",  # "threads" (default) or "loky"
    verbose=True,
):
    """
    Joint block ridge: y ~ [X1, Z].
    Parallelization is ONLY across CV folds; grid search runs sequentially.
    """
    # ----- Coerce arrays (RAW) -----
    X1_tr = np.asarray(predictor1['train']);  X1_te = np.asarray(predictor1['test'])
    Z_tr  = np.asarray(predictor2['train']);  Z_te  = np.asarray(predictor2['test'])
    y_tr  = np.asarray(target['train']).ravel(); y_te = np.asarray(target['test']).ravel()

    if Z_tr.ndim == 1:  Z_tr = Z_tr[:, None]
    if Z_te.ndim == 1:  Z_te = Z_te[:, None]
    if X1_tr.ndim == 1: X1_tr = X1_tr[:, None]
    if X1_te.ndim == 1: X1_te = X1_te[:, None]

    # ----- Final-fit standardization (fit on TRAIN) -----
    X1_mu = np.mean(X1_tr, axis=0, keepdims=True); X1_sd = np.std(X1_tr, axis=0, keepdims=True); X1_sd[X1_sd == 0] = 1.0
    Z_mu  = np.mean(Z_tr,  axis=0, keepdims=True); Z_sd  = np.std(Z_tr,  axis=0, keepdims=True); Z_sd[Z_sd == 0]  = 1.0
    y_mu  = float(np.mean(y_tr));                  y_sd  = float(np.std(y_tr));                   y_sd = 1.0 if y_sd == 0 else y_sd

    X1_tr_s = (X1_tr - X1_mu) / X1_sd
    X1_te_s = (X1_te - X1_mu) / X1_sd
    Z_tr_s  = (Z_tr  - Z_mu)  / Z_sd
    Z_te_s  = (Z_te  - Z_mu)  / Z_sd
    y_tr_s  = (y_tr  - y_mu)  / y_sd
    y_te_s  = (y_te  - y_mu)  / y_sd

    if verbose:
        print(f'[LOGGING] Shapes X1: {X1_tr.shape}/{X1_te.shape} | Z: {Z_tr.shape}/{Z_te.shape}')
        print('[LOGGING] Grid tuning (lam1 × lam2) with CV-parallel only')

    # ----- CV splits once -----
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    splits = list(kf.split(X1_tr))

    # ----- Sequential grid search; each pair runs CV in parallel -----
    best_cv = -np.inf
    best_pair = (None, None)
    for lam1, lam2 in tqdm(product(lambdas1, lambdas2), total=len(lambdas1)*len(lambdas2), desc='Searching lambda grid...'):
        cv = _cv_score_block_ridge_raw_with_splits(
            X1_tr, Z_tr, y_tr, lam1, lam2, splits, jitter=jitter,
            n_jobs_cv=n_jobs_cv, cv_backend=cv_backend
        )
        if cv > best_cv:
            best_cv = cv
            best_pair = (lam1, lam2)

    lam1_both, lam2_both = best_pair

    # ----- Final fit & test -----
    b1_both, b2_both = _fit_block_ridge_closed_form(X1_tr_s, Z_tr_s, y_tr_s, lam1_both, lam2_both, jitter)
    yhat_both_te = _predict_block(X1_te_s, Z_te_s, b1_both, b2_both)

    score_r2 = r2_score(y_te_s, yhat_both_te)
    try:
        score_r = float(pearsonr(y_te_s, yhat_both_te)[0])
    except Exception:
        score_r = float('nan')

    return {
        "score_r2": float(score_r2),
        "score_r":  float(score_r),
        "y_hat":    yhat_both_te,
        "lambda1_both": float(lam1_both),
        "lambda2_both": float(lam2_both),
        "standardization": {
            "X1_mu": X1_mu, "X1_sd": X1_sd,
            "Z_mu":  Z_mu,  "Z_sd":  Z_sd,
            "y_mu":  y_mu,  "y_sd":  y_sd,
        },
    }


# ==========================================================
# Global job list: (target × pose_feature × model) flatten
# ==========================================================



def _count_jobs(targets, pose_features, model_types=['image_models', 'video_models']):
    """
    Count total jobs without loading any data.
    Returns list of job specifications (target, pose_feature, model_name, model_type).
    """
    # Get all models once by listing directories
    all_models = []
    for m_type in model_types:
        models_dir = Path(SOTA_MODEL_PATH) / m_type
        if models_dir.exists():
            models = os.listdir(models_dir)
            all_models.extend([(model, m_type) for model in models])
    
    # Create job specs for every combination
    job_specs = []
    for target in targets:
        for pose_feature in pose_features:
            for model_name, model_type in all_models:
                job_specs.append({
                    'target': target,
                    'pose_feature': pose_feature,
                    'model_name': model_name,
                    'model_type': model_type
                })
    
    return job_specs


def _get_model_info_for_job(target, model_name, model_type, pca_dim=None, top_k_layers=1):
    """
    Get the model info for a specific model by directly loading only that model's pickle files.
    Avoids loading all models via get_top_sota_scores.
    """
    # Determine the score directory based on PCA
    if pca_dim is None:
        score_base_dir = Path(f"experiments/SOTA_beh/{model_type}")
    else:
        score_base_dir = Path(f"experiments/SOTA_beh/PCA{pca_dim}/{model_type}")
    
    # Load only this specific model's CV pickle to find best layer
    cv_path = score_base_dir / "cv" / f"{model_name}_target_encoding.pkl"
    
    if not cv_path.exists():
        return None
    
    cv_df = load_pickle(str(cv_path))
    
    # Filter to the specific target
    target_cv = cv_df[cv_df['y'] == target]
    
    if target_cv.empty:
        return None
    
    # Get top k layers for this target
    top_layers = (
        target_cv.sort_values('score_r', ascending=False)
        .head(top_k_layers)
    )
    
    if top_layers.empty:
        return None
    
    # Use the best layer
    best_layer = top_layers.iloc[0]
    layer_name = best_layer['layer_name']
    
    # Construct the layer path
    layer_path = os.path.join(
        SOTA_MODEL_PATH,
        model_type,
        model_name,
        f"{layer_name}.npz"
    )
    
    if not os.path.exists(layer_path):
        return None
    
    # Load only this specific model layer
    model_dict = get_sota_model_layers(layer_path, pca_dim)
    
    return model_dict


def _partition_jobs_by_task(targets, pose_features, task_id, max_tasks):
    """
    Optimized version: counts jobs first, partitions, then loads only needed data.
    """
    # Step 1: Count all jobs without loading data
    job_specs = _count_jobs(targets, pose_features)
    total = len(job_specs)
    
    # Step 2: Determine partition
    if max_tasks is None or max_tasks <= 0:
        start, end = 0, total
        chunk = total
    else:
        chunk = ceil(total / max_tasks)
        start = max(0, (task_id - 1) * chunk)
        end = min(task_id * chunk, total)
    
    # Step 3: Get only the job specs for this partition
    partition_specs = job_specs[start:end]
    
    return partition_specs, start, end, total, chunk


def get_grouped_ridge_scores(
    targets,
    pose_features,
    task_id: int,
    max_tasks: int,
    overwrite: bool = False,
    n_jobs_grid: int | None = None,
    jitter: float = 0.0,
    verbose: bool = True,
):
    """
    Optimized single-level parallelism with lazy loading.
    Only loads data for jobs assigned to this task.
    """
    if n_jobs_grid is None or n_jobs_grid <= 0:
        n_jobs_grid = ALPHA_CV_SPLITS

    # Partition across ALL (target × pose × model) - only specs, no data loaded yet
    partition_specs, start, end, total, chunk = _partition_jobs_by_task(
        targets, pose_features, task_id, max_tasks
    )

    print("====== Global Partition Report ======")
    print(f"Total jobs (target × pose × model): {total}")
    if max_tasks and max_tasks > 0:
        print(f"max_tasks: {max_tasks}  |  chunk size: {chunk}")
        print(f"task_id:   {task_id}    |  processing slice [{start}:{end})  → {len(partition_specs)} jobs")
    print(f"Grid CV n_jobs: {n_jobs_grid}")
    print("====================================")

    # Step 4: Process each job spec, loading data only when needed
    for job_spec in partition_specs:
        target = job_spec['target']
        pose_feature = job_spec['pose_feature']
        model_name = job_spec['model_name']
        model_type = job_spec['model_type']
        
        # NOW load the actual data (lazy loading)
        target_data = get_target_ratings(target)
        pose_data = get_pose_features(pose_feature)
        model_dict = _get_model_info_for_job(target, model_name, model_type)
        
        if model_dict is None:
            if verbose:
                print(f"[SKIP] {target} | {pose_feature} | {model_name} (not found)")
            continue
        
        # Setup output paths
        output_dir = f"experiments/ridge_results/{pose_feature.replace(' ', '_')}/{target.replace(' ', '_')}"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{model_name}.pkl")

        if (not overwrite) and os.path.exists(output_path):
            if verbose:
                print(f"[SKIP] {target} | {pose_feature} | {model_name} (exists)")
            continue

        # Align and run group ridge
        sota_aligned, arrow_aligned, target_aligned = align_multiple_vars(
            model_dict, pose_data, target_data
        )

        results = group_ridge(
            predictor1=sota_aligned,
            predictor2=arrow_aligned,
            target=target_aligned,
            jitter=jitter,
            n_jobs_cv=n_jobs_grid,
            verbose=True
        )

        save_pickle(results, output_path)
        if verbose:
            print(f"[DONE] {target} | {pose_feature} | {model_name} → {output_path}")