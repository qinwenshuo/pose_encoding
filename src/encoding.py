import os
import pickle
import torch
import numpy as np
import pandas as pd
from math import ceil
from tqdm import tqdm
from pathlib import Path
from scipy.stats import pearsonr
from himalaya.ridge import RidgeCV
from sklearn.metrics import r2_score
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import KFold
from sklearn.linear_model import RidgeCV as SKRidgeCV
from sklearn.linear_model import LinearRegression
from joblib import Parallel, delayed
from src.config import ALPHAS, CV_SPLITS, JOBS, SOTA_PLOT_NAME, RANDOM
from src.config import RATING_OF_INTEREST, SOTA_MODEL_PATH
from src.data_utils import apply_pca, order_files, save_pickle, load_pickle, zscore_fit_apply
from src.data_utils import get_pose_features, get_target_ratings, get_sota_model_layers, align_multiple_vars
# from src.plottings import plot_score_by_layer

def _assert_finite(name, arr):
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} contains non-finite values after preprocessing")

def encode(
    X_train, X_test, y_train, y_test,
    x_type, layer_name, feature,
    eval_mode: str = "test",
    alphas=ALPHAS,
    cv_splits: int = CV_SPLITS,
):
    """
    One-shot encoding using RidgeCV with leak-free normalization.

    Returns
    -------
    dict with keys:
      - "x", "layer_name", "y"                  : identifiers (names)
      - "X_train", "X_test", "y_train", "y_test": raw arrays as provided (optional with 'return_data')
      - "y_hat"                                 : predictions (OOF for 'cv', test-set for 'test')
      - "score_r"                               : mean Pearson r across targets
      - "score_r_per_target"                    : per-target Pearson r (1D array)
      - "score_r2_mean"                         : mean R² across targets
      - "score_r2_per_target"                   : per-target R² (1D array)
      - "alpha_selected"                        : float (test) or list[float] (cv, per fold)
      - "eval_mode"                             : bookkeeping
    """
    # --- helpers ---
    def _ensure_2d_y(y):
        y = np.asarray(y)
        if y.ndim == 1:
            y = y[:, None]
        return y

    def _pearson_per_target(y_true, y_pred):
        y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
        if y_true.ndim == 1:  # single target
            return np.array([pearsonr(y_true, y_pred)[0]], dtype=float)
        rs = []
        for j in range(y_true.shape[1]):
            yt = y_true[:, j]; yp = y_pred[:, j]
            mask = np.isfinite(yt) & np.isfinite(yp)
            if mask.sum() >= 2:
                r = pearsonr(yt[mask], yp[mask])[0]
            else:
                r = np.nan
            rs.append(r)
        return np.array(rs, dtype=float)

    def _r2_per_target(y_true, y_pred):
        y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
        if y_true.ndim == 1:
            return np.array([r2_score(y_true, y_pred)], dtype=float)
        r2s = []
        for j in range(y_true.shape[1]):
            yt = y_true[:, j]; yp = y_pred[:, j]
            mask = np.isfinite(yt) & np.isfinite(yp)
            if mask.sum() >= 2 and np.nanvar(yt[mask]) > 0:
                r2 = r2_score(yt[mask], yp[mask])
            else:
                r2 = np.nan
            r2s.append(r2)
        return np.array(r2s, dtype=float)

    def _nanmean_safe(arr):
        arr = np.asarray(arr, dtype=float)
        return float(np.nanmean(arr)) if arr.size else np.nan
    
    # --- core ---
    out = {
        "x": x_type,
        "layer_name": layer_name,
        "y": feature,
        "score_r": np.nan,
        "score_r_per_target": None,
        "score_r2_mean": np.nan,
        "score_r2_per_target": None,
        "alpha_selected": None,
        "eval_mode": eval_mode,
    }

    y_train = _ensure_2d_y(y_train)
    y_test  = _ensure_2d_y(y_test) if y_test is not None else None

    if eval_mode == "cv":
        # Build OOF predictions on the training set
        n = X_train.shape[0]
        n_targets = y_train.shape[1]
        y_oof = np.full((n, n_targets), np.nan, dtype=float)
        alpha_each_fold = []
        fold_scores = []

        kf = RepeatedKFold(n_splits=cv_splits, n_repeats=2, random_state=RANDOM)
        for tr_idx, va_idx in kf.split(X_train):
            X_tr, X_va = X_train[tr_idx], X_train[va_idx]
            y_tr, y_va = y_train[tr_idx], y_train[va_idx]

            # Fold-only normalization
            X_tr_n, X_va_n = zscore_fit_apply(X_tr, X_va, axis=0)
            y_tr_n, y_va_n = zscore_fit_apply(y_tr, y_va, axis=0)
            _assert_finite("X_tr_n", X_tr_n); _assert_finite("X_va_n", X_va_n)
            _assert_finite("y_tr_n", y_tr_n); _assert_finite("y_va_n", y_va_n)


            # sklearn’s RidgeCV with SVD solver
            ridge = SKRidgeCV(
                alphas=alphas,
            #     # store_cv_results=False,
            #     # cv=ALPHA_CV_SPLITS,
            )

            # ridge = RidgeCV(
            #     alphas=alphas,
            #     fit_intercept=False,   # if you z-score X and y
            #     solver="svd",          # Himalaya’s optimized solver
            #     cv=ALPHA_CV_SPLITS,
            # )

            ridge.fit(X_tr_n, y_tr_n)
            y_va_pred = ridge.predict(X_va_n)
            if y_va_pred.ndim == 1:
                y_va_pred = y_va_pred[:, None]
            y_oof[va_idx, :] = y_va_pred

            fold_scores.append(float(_nanmean_safe(_pearson_per_target(y_va, y_va_pred))))

            # sklearn’s RidgeCV stores selected alpha in ridge.alpha_
            alpha_each_fold.append(float(ridge.alpha_))

            # # Himalaya’s RidgeCV selects a single alpha for multioutput
            # alpha_each_fold.append(float(np.atleast_1d(getattr(ridge, "best_alphas_", [np.nan]))[0]))


        # Scores on OOF predictions (train set)
        r_per = _pearson_per_target(y_train, y_oof)
        r2_per = _r2_per_target(y_train, y_oof)

        out.update({
            "y_hat": y_oof.squeeze(),
            "score_r_per_target": r_per,
            "score_r": _nanmean_safe(r_per),
            "score_r2_per_target": r2_per,
            "score_r2_mean": _nanmean_safe(r2_per),
            "alpha_selected": alpha_each_fold,   # list[float], one per outer fold
            "score_r_per_fold": fold_scores,     # list[float], one per CV fold
        })

    elif eval_mode == "test":
        # Train-only normalization; predict on test
        X_tr_n, X_te_n = zscore_fit_apply(X_train, X_test, axis=0)
        y_tr_n, y_te_n = zscore_fit_apply(y_train, y_test, axis=0)
        _assert_finite("X_tr_n", X_tr_n); _assert_finite("X_te_n", X_te_n)
        _assert_finite("y_tr_n", y_tr_n); _assert_finite("y_te_n", y_te_n)


        # sklearn's RidgeCV with SVD solver
        ridge = SKRidgeCV(
            alphas=alphas,
            # fit_intercept=False,   # z-scored X and y, no intercept needed
            # gcv_mode='svd'
            # store_cv_results=False,
            # cv=ALPHA_CV_SPLITS,
        )

        # ridge = RidgeCV(
        #     alphas=alphas,
        #     fit_intercept=False,   # if you z-score X and y
        #     solver="svd",          # Himalaya’s optimized solver
        # )

        ridge.fit(X_tr_n, y_tr_n)
        y_te_pred = ridge.predict(X_te_n)

        # Himalaya's alpha
        # alpha_selected = float(np.atleast_1d(getattr(ridge, "best_alphas_", [np.nan]))[0])

        # sklearn's RidgeCV stores selected alpha in ridge.alpha_
        alpha_selected = float(ridge.alpha_)

        r_per = _pearson_per_target(y_te_n.squeeze(), y_te_pred.squeeze())
        r2_per = _r2_per_target(y_te_n.squeeze(), y_te_pred.squeeze())

        out.update({
            "y_hat": y_te_pred.squeeze(),
            "score_r_per_target": r_per,
            "score_r": _nanmean_safe(r_per),
            "score_r2_per_target": r2_per,
            "score_r2_mean": _nanmean_safe(r2_per),
            "alpha_selected": alpha_selected,
        })
    else:
        raise ValueError("`eval_mode` must be 'test' or 'cv'.")

    return out


def pose_target_encoding(
    pose_features,
    targets,
    eval_mode='test',
    alphas=ALPHAS,
    cv_splits=CV_SPLITS,
    n_jobs=JOBS,
    pca_n_components=None,
    save_path=None,
    show_progress=False,
):
    """
    Evaluate regression performance using either a test set or cross-validation (CV),
    with an optional PCA dimensionality reduction.

    Parameters:
      pose_features (list of str): feature names to loop over.
      method (str): 'ridge' uses RidgeCV, 'linear' uses LinearRegression.
      eval_mode (str): 'test' evaluates on the test set, 'cv' performs CV on the training set.
      alphas (array-like): List of regularization strengths (used for ridge regression).
      pca_n_components (int or None): Number of PCA components to keep. If None, skip PCA.

    Returns:
      all_model_results (list): A list of dicts, one per target, each containing:
          - 'x': The feature name used for encoding (from pose_features).
          - 'y': The target rating name from RATING_OF_INTEREST.
          - 'scor_r': The Pearson correlation coefficient.
    """

    for i in targets:
        if i not in RATING_OF_INTEREST:
            raise ValueError(f"{i} is not available in {', '.join(RATING_OF_INTEREST)}")
        
    jobs = []
    for feature_name in pose_features:
        pose_data = get_pose_features(feature_name)
        # Optional PCA
        if pca_n_components is not None:
            pcaed_train, pcaed_test = apply_pca(pose_data['train'], pose_data['test'], n_components=pca_n_components)
            pose_data['train'] = pcaed_train
            pose_data['test'] = pcaed_test

        for target_name in targets:
            target_data = get_target_ratings(target_name)
            pose_data, target_data = align_multiple_vars(pose_data, target_data)

            # Build flat job list (feature × layer)
            jobs.append((
                pose_data['train'], pose_data['test'], 
                target_data['train'], target_data['test'],
                feature_name, None, target_name,
                eval_mode, alphas, cv_splits
            ))

    # Execute in parallel with one progress bar
    iterator = tqdm(jobs, desc=f"Target encoding") if show_progress else jobs
    results = Parallel(n_jobs=n_jobs)(
        delayed(encode)(*job) for job in iterator
    )

    df = pd.DataFrame(results)
    if save_path:
        save_pickle(df, save_path)
        # df.to_csv(save_path, index=False)

    return df


def sota_target_encoding(
    model_dir,
    targets,
    eval_mode="test",
    alphas=ALPHAS,
    cv_splits=CV_SPLITS,
    n_jobs=JOBS,
    pca=None,
    layer=None,
    save_path=None,
    show_progress=True,
    post_order=True,
):
    # 1) Checking targets
    for i in targets:
        if i not in RATING_OF_INTEREST:
            raise ValueError(f"{i} is not available in {', '.join(RATING_OF_INTEREST)}")

    # 2) Loading all model layers using parallel processing
    if post_order:
        layer_files = order_files(model_dir)
    else: 
        layer_files = sorted([p.stem for p in Path(model_dir).rglob('*.npz')])
    if layer and layer in layer_files:
        layer_files = [layer]
    if not layer_files:
        raise ValueError(f"No layers found in {model_dir}") 

    sota_jobs = [(os.path.join(model_dir, f'{layer_name}.npz'), pca) for layer_name in layer_files]
    iterator = tqdm(sota_jobs, desc="[LOGGING] Loading model layers...") if show_progress else sota_jobs
    layers_data = Parallel(n_jobs=n_jobs)(delayed(get_sota_model_layers)(*job) for job in iterator)
    
    # 3) Loading targets and build encoding jobs
    jobs = []
    for target_name in targets:
        target_data = get_target_ratings(target_name)
        for layer_data in layers_data:
            layer_data, target_data = align_multiple_vars(layer_data, target_data)
            jobs.append((
                layer_data['train'], layer_data['test'], 
                target_data['train'], target_data['test'],
                layer_data['type'], layer_data['layer_name'], target_name,
                eval_mode, alphas, cv_splits
            ))

    # 4) Execute in parallel with one progress bar
    iterator = tqdm(jobs, desc=f"target encoding {model_dir}") if show_progress else jobs
    results = Parallel(n_jobs=n_jobs)(
        delayed(encode)(*job) for job in iterator
    )

    # 5) Collect & save
    df = pd.DataFrame(results)
    if save_path:
        save_pickle(df, save_path)
        # df.to_csv(save_path, index=False)
    return df


def sota_pose_encoding(
    model_dir,
    pose_features,
    eval_mode="test",
    alphas=ALPHAS,
    cv_splits=CV_SPLITS,
    n_jobs=JOBS,
    pca=None,
    layer=None,
    save_path=None,
    show_progress=True,
    post_order=True,
):

    # 2) Prepare layer data: (layer_idx, layer_name, full_train_repr, full_test_repr)
    if post_order:
        layer_files = order_files(model_dir)
    else: 
        layer_files = sorted([p.stem for p in Path(model_dir).rglob('*.npz')])
    if layer and layer in layer_files:
        layer_files = [layer]
    if not layer_files:
        raise ValueError(f"No layers found in {model_dir}") 

    sota_jobs = [(os.path.join(model_dir, f'{layer_name}.npz'), pca) for layer_name in layer_files]
    iterator = tqdm(sota_jobs, desc="[LOGGING] Loading model layers...") if show_progress else sota_jobs
    layers_data = Parallel(n_jobs=n_jobs)(delayed(get_sota_model_layers)(*job) for job in iterator)

    # 3) Build flat job list (feature × layer)
    jobs = []
    for feature_name in pose_features:
        pose_data = get_pose_features(feature_name)
        for layer_data in layers_data:
            pose_data, layer_data = align_multiple_vars(pose_data, layer_data)
            jobs.append((
                layer_data['train'], layer_data['test'], 
                pose_data['train'], pose_data['test'],
                layer_data['type'], layer_data['layer_name'], feature_name,
                eval_mode, alphas, cv_splits
            ))

    # 4) Execute in parallel with one progress bar
    iterator = tqdm(jobs, desc=f"Pose encoding {model_dir}") if show_progress else jobs
    results = Parallel(n_jobs=n_jobs)(
        delayed(encode)(*job) for job in iterator
    )

    # 5) Wrap up
    df = pd.DataFrame(results)
    if save_path:
        save_pickle(df, save_path)
        # df.to_csv(save_path, index=False)
    return df


# ===================== Compile Encoding Results =====================
def get_sota_encoding_scores(
    targets, 
    pose_features, 
    pca=None, 
    save_plots=True, 
    overwrite=False, 
    task_id=None, 
    max_tasks=20
):
    sota_model_path = Path(SOTA_MODEL_PATH)

    def list_models(model_type: str):
        root = sota_model_path / model_type
        if not root.exists():
            return []
        return sorted([d.name for d in root.iterdir() if d.is_dir()])

    # 1) Build one combined, ordered list of (model_type, model)
    ordered_types = ["video_models", "image_models"]
    combined = []
    for mt in ordered_types:
        for m in list_models(mt):
            combined.append((mt, m))
    total_models = len(combined)
    if total_models == 0:
        print("[Logging] No models found. Nothing to do.")
        return
    
    # 2) Determine PCA-dependent folder scheme (created lazily per type)
    def base_paths(model_type: str):
        if pca is None:
            test_result = Path(f"experiments/SOTA_beh/{model_type}/test")
            test_fig    = Path(f"results/SOTA/{model_type}/test")
            cv_result   = Path(f"experiments/SOTA_beh/{model_type}/cv")
            cv_fig      = Path(f"results/SOTA/{model_type}/cv")
        else:
            test_result = Path(f"experiments/SOTA_beh/PCA{pca}/{model_type}/test")
            test_fig    = Path(f"results/SOTA/PCA{pca}/{model_type}/test")
            cv_result   = Path(f"experiments/SOTA_beh/PCA{pca}/{model_type}/cv")
            cv_fig      = Path(f"results/SOTA/PCA{pca}/{model_type}/cv")
        return test_result, test_fig, cv_result, cv_fig
    
    # 3) Split once by task_id (ceil division)
    if task_id is not None:
        if not (1 <= task_id <= max_tasks):
            raise ValueError(f"⚠️ task_id must be between 1 and {max_tasks} inclusive.")
        chunk_size = ceil(total_models / max_tasks)
        start_idx = (task_id - 1) * chunk_size
        end_idx = min(task_id * chunk_size, total_models)
        combined = combined[start_idx:end_idx]
        print(f"🧩 Task {task_id}: processing models {start_idx}-{end_idx - 1} ({len(combined)} total).")

    # 4) Create per-type directories once (but only when at least one model of that type is in the subset)
    subset_types = {mt for (mt, _) in combined}
    for mt in subset_types:
        test_result, test_fig, cv_result, cv_fig = base_paths(mt)
        test_result.mkdir(parents=True, exist_ok=True)
        test_fig.mkdir(parents=True, exist_ok=True)
        cv_result.mkdir(parents=True, exist_ok=True)
        cv_fig.mkdir(parents=True, exist_ok=True)

    # 5) Process models
    pbar = tqdm(combined, desc="Encoding (video+image combined)")
    for model_type, model in pbar:
        pbar.set_description(f"Encoding: {model_type}/{model}")

        model_path = sota_model_path / model_type / model
        test_result_path, test_fig_folder, cv_result_path, cv_fig_folder = base_paths(model_type)

        # Expected output files
        test_target_encoding_path = test_result_path / f"{model}_target_encoding.pkl"
        test_pose_encoding_path   = test_result_path / f"{model}_pose_encoding.pkl"
        cv_target_encoding_path   = cv_result_path / f"{model}_target_encoding.pkl"

        # Skip early (and avoid mkdirs per-model) if outputs already exist
        if (not overwrite) and test_target_encoding_path.exists() and test_pose_encoding_path.exists() and cv_target_encoding_path.exists():
            print(f"\n[Logging] {model_type}/{model} already encoded, skipped.")
            continue

        # ---- Run encodings ----
        print('[LOGGING] Running test target encoding...')
        test_target_encoding_scores = sota_target_encoding(
            str(model_path), targets, eval_mode="test", pca=pca,
            save_path=str(test_target_encoding_path), show_progress=True
        )

        print('[LOGGING] Running test pose encoding...')
        test_pose_encoding_scores = sota_pose_encoding(
            str(model_path), pose_features, eval_mode="test", pca=pca,
            save_path=str(test_pose_encoding_path), show_progress=True
        )

        print('[LOGGING] Running CV target encoding...')
        cv_target_encoding_scores = sota_target_encoding(
            str(model_path), targets, eval_mode="cv", pca=pca,
            save_path=str(cv_target_encoding_path), show_progress=True
        )

        # ---- Plots ----
        # if save_plots:
        #     test_target_encoding_fig_dir = test_fig_folder / f"{model}_target_encoding"
        #     test_pose_encoding_fig_dir   = test_fig_folder / f"{model}_pose_encoding"
        #     cv_target_encoding_fig_dir   = cv_fig_folder / f"{model}_target_encoding"
        #     test_target_encoding_fig_dir.mkdir(parents=True, exist_ok=True)
        #     test_pose_encoding_fig_dir.mkdir(parents=True, exist_ok=True)
        #     cv_target_encoding_fig_dir.mkdir(parents=True, exist_ok=True)

        #     plot_score_by_layer(results=test_target_encoding_scores,
        #                         save_dir=str(test_target_encoding_fig_dir),
        #                         y_lim=(-0.1, 0.9))
        #     plot_score_by_layer(results=test_pose_encoding_scores,
        #                         save_dir=str(test_pose_encoding_fig_dir),
        #                         y_lim=(-0.1, 0.9))
        #     plot_score_by_layer(results=cv_target_encoding_scores,
        #                         save_dir=str(cv_target_encoding_fig_dir),
        #                         y_lim=(-0.1, 0.9))


def get_top_sota_scores(
    targets,
    pose_features=None,
    collect='target',
    model_type='both',
    pca=None,
    top_n=1,
    save_path=None,
    # show_progress=True,
    n_jobs=-1,                 # NEW: parallelism (default: use all cores)
    prefer="processes"         # NEW: prefer processes for pandas/numpy safety
) -> pd.DataFrame:
    """
    Parallelized version with joblib. Per-model work is executed in parallel.
    """
    print(f'[LOGGING] Loading sota scores on targets({targets}) and pose_features({pose_features})')
    if targets is None:
        raise ValueError('targets are None')

    if collect not in ('both', 'target'):
        raise ValueError("collect must be either 'both' or 'target'")

    if model_type == 'both':
        model_types = ['image_models', 'video_models']
    else:
        model_types = [model_type]

    if pose_features is None:
        pose_features = []

    all_chunks = []

    for m_type in model_types:
        # Resolve directories & labels
        if pca is None:
            score_base_dir = Path(f"experiments/SOTA_beh/{m_type}")
            x_name = SOTA_PLOT_NAME
        else:
            score_base_dir = Path(f"experiments/SOTA_beh/PCA{pca}/{m_type}")
            x_name = f'{SOTA_PLOT_NAME} (6 PCA)'

        # Enumerate models once
        models_dir = Path(SOTA_MODEL_PATH) / m_type
        if not models_dir.exists():
            print(f"[Warning] No directory for {m_type} at {models_dir}. Skipping.")
            continue
        sota_models = os.listdir(models_dir)

        # Optional progress wrapper for the list of models
        iterator = sota_models
        # if show_progress:
        #     from tqdm import tqdm
        #     iterator = tqdm(iterator, desc=f"Compiling SOTA data ({m_type})", leave=False)

        # Worker to process one model
        def _process_one_model(model: str) -> pd.DataFrame:
            cv_p    = score_base_dir / "cv"   / f"{model}_target_encoding.pkl"
            test_p  = score_base_dir / "test" / f"{model}_target_encoding.pkl"
            pose_p  = score_base_dir / "test" / f"{model}_pose_encoding.pkl"

            # Load three pickles (I/O-bound; benefits from parallel)
            cv_df   = load_pickle(str(cv_p))
            test_df = load_pickle(str(test_p))
            # pose_df only needed if 'both'
            pose_df = load_pickle(str(pose_p)) if collect == 'both' else None

            # Precompute top layers per target from CV
            # (avoid repeated sorting/filtering)
            top_layers_per_target = {}
            for t in targets:
                sub = cv_df[cv_df['y'] == t]
                if sub.empty:
                    # nothing for this target in this model
                    top_layers_per_target[t] = []
                    continue
                # sort by r descending, take top_n, tolist
                top_layers_per_target[t] = (
                    sub.sort_values('score_r', ascending=False)
                       .head(top_n)['layer_name'].tolist()
                )

            rows = []


            # Collect only target scores (and y_hat)
            for t in targets:
                for layer in top_layers_per_target.get(t, []):
                    mask = (test_df['y'] == t) & (test_df['layer_name'] == layer)
                    if not mask.any():
                        continue
                    # .iloc[0] is safer than .item() for Series with 1 element
                    target_r = test_df.loc[mask, 'score_r'].iloc[0]
                    target_hat = test_df.loc[mask, 'y_hat'].iloc[0]

                    if collect == 'target':
                        rows.append({
                            'model_name': model,
                            'layer_name': layer,
                            'x':          x_name,
                            'y':          t,
                            'score_r':    target_r,
                            'model_type': m_type,
                            'y_hat':      target_hat,
                        })
                    
                    elif collect == 'both':
                        if pose_df is None:
                            return pd.DataFrame()  # nothing to add

                        # match pose_r for each feature at the SAME layer
                        for feat in pose_features:
                            pmask = (pose_df['y'] == feat) & (pose_df['layer_name'] == layer)
                            if not pmask.any():
                                continue
                            pose_r = pose_df.loc[pmask, 'score_r'].iloc[0]
                            pose_hat = pose_df.loc[pmask, 'y_hat'].iloc[0]
                            rows.append({
                                'model_name':   model,
                                'layer_name':   layer,
                                'x':            'Vision DNNs',
                                'target':       t,
                                'target_r':     target_r,
                                'target_hat':   target_hat,
                                'pose_feature': feat,
                                'pose_r':       pose_r,
                                'pose_hat':     pose_hat
                            })

            return pd.DataFrame(rows)

        # Run the per-model jobs in parallel
        # Note: wrap `iterator` into a plain list now to avoid tqdm exhaustion by joblib
        model_list = list(iterator)
        chunks = Parallel(n_jobs=n_jobs, prefer=prefer)(
            delayed(_process_one_model)(model) for model in model_list
        )
        # Concatenate results for this model_type
        if chunks:
            all_chunks.append(pd.concat([c for c in chunks if not c.empty], ignore_index=True))

    # Final concatenation
    if all_chunks:
        df = pd.concat(all_chunks, ignore_index=True)
    else:
        df = pd.DataFrame()

    if save_path is not None and not df.empty:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(save_path, index=False)

    return df


def get_top_sota_layers(
        target, 
        top_n_models,
        top_k_layers=1, 
        pca_dim=None, 
        n_jobs=-1,
        show_progress=True
    ):
    print('[LOGGING] Loading top SOTA layers ...')
    if pca_dim is not None and pca_dim != 6:
        raise NotImplementedError

    df = get_top_sota_scores(
        targets=[target],
        pose_features=[],
        collect='target',
        pca=pca_dim,
        top_n=top_k_layers,
    )

    # Rank models by their best layer score
    best_per_model = df.groupby("model_name")["score_r"].max()

    # Handle top_n = -1 (return all models)
    if top_n_models == -1:
        top_model_names = best_per_model.sort_values(ascending=False).index.tolist()
    else:
        top_model_names = (
            best_per_model.sort_values(ascending=False)
            .head(top_n_models)
            .index
            .tolist()
        )

    # Keep all layers for those selected models
    top_models = (df[df["model_name"].isin(top_model_names)].sort_values(["score_r"], ascending=False).reset_index(drop=True))

    sota_jobs = []
    for _, row in top_models.iterrows():
        layer_path = os.path.join(SOTA_MODEL_PATH, row["model_type"], row["model_name"], f"{row['layer_name']}.npz")
        sota_jobs.append((layer_path, pca_dim))
    iterator = tqdm(sota_jobs, desc="[LOGGING] Loading model layers...") if show_progress else sota_jobs

    # Run in parallel
    sota_layer_data = Parallel(n_jobs=n_jobs)(
        delayed(get_sota_model_layers)(*job) for job in iterator
    )

    return sota_layer_data



"""
TO DO: UPDATE THE PCA ON 4D HUMAN
"""
def get_4dhumans_encoding_scores(
    targets,
    pose_features,
    pca=None,
    save_plots=True,
    overwrite=False,
):
    """
    Run 4D-Humans encodings (test targets, test pose, CV targets)
    Returns
    -------
    """

    # ---- Model location (single 4D-Humans bundle) ----
    model_path = 'experiments/4DHuman_per_layer_matrices/'

    # ---- PCA-dependent folder scheme (match SOTA helper style) ----
    if pca is None:
        test_result_path = Path('experiments/4DHuman_beh/test')
        cv_result_path   = Path('experiments/4DHuman_beh/cv')
        test_fig_folder  = Path('results/linear_encoding/4DHuman/test')
        cv_fig_folder    = Path('results/linear_encoding/4DHuman/cv')
    else:
        test_result_path = Path(f'experiments/4DHuman_beh/PCA{pca}/test')
        cv_result_path   = Path(f'experiments/4DHuman_beh/PCA{pca}/cv')
        test_fig_folder  = Path(f'results/linear_encoding/4DHuman/PCA{pca}/test')
        cv_fig_folder    = Path(f'results/linear_encoding/4DHuman/PCA{pca}/cv')

    # ---- Ensure directories exist (created once) ----
    test_result_path.mkdir(parents=True, exist_ok=True)
    cv_result_path.mkdir(parents=True, exist_ok=True)
    test_fig_folder.mkdir(parents=True, exist_ok=True)
    cv_fig_folder.mkdir(parents=True, exist_ok=True)

    # ---- Output files ----
    test_target_encoding_path = test_result_path / '4DHuman_target_encoding.pkl'
    test_pose_encoding_path   = test_result_path / '4DHuman_pose_encoding.pkl'
    cv_target_encoding_path   = cv_result_path   / '4DHuman_target_encoding.pkl'

    # ---- (Optional) figure subfolders (if your plotting utils use them) ----
    test_target_encoding_fig_dir = test_fig_folder / '4DHuman_target_encoding'
    test_pose_encoding_fig_dir   = test_fig_folder / '4DHuman_pose_encoding'
    cv_target_encoding_fig_dir   = cv_fig_folder   / '4DHuman_target_encoding'
    if save_plots:
        test_target_encoding_fig_dir.mkdir(parents=True, exist_ok=True)
        test_pose_encoding_fig_dir.mkdir(parents=True, exist_ok=True)
        cv_target_encoding_fig_dir.mkdir(parents=True, exist_ok=True)

    # ---- Early skip if everything already exists and overwrite is False ----
    if (not overwrite
        and test_target_encoding_path.exists()
        and test_pose_encoding_path.exists()
        and cv_target_encoding_path.exists()):
        print("[Logging] 4D-Humans already encoded; skipped (use overwrite=True to recompute).")
        return

    # ---- Run encodings ----
    print('[LOGGING] Running test target encoding...')
    test_target_encoding_scores = sota_target_encoding(
        model_path, targets, eval_mode='test',
        save_path=str(test_target_encoding_path),
        post_order=False,  # keep your original
        show_progress=True,
        pca=pca,           # pass through for parity with SOTA helper
    )

    print('[LOGGING] Running test pose encoding...')
    test_pose_encoding_scores = sota_pose_encoding(
        model_path, pose_features, eval_mode='test',
        save_path=str(test_pose_encoding_path),
        post_order=False,
        show_progress=True,
        pca=pca,
    )
    print('[LOGGING] Running cross-validation target encoding...')
    cv_target_encoding_scores = sota_target_encoding(
        model_path, targets, eval_mode='cv',
        save_path=str(cv_target_encoding_path),
        post_order=False,
        show_progress=True,
        pca=pca,
    )
    # if save_plots:
        # plot_score_by_layer(results=test_target_encoding_scores, save_dir=test_target_encoding_fig_dir, y_lim=(-0.1, 0.9))
        # plot_score_by_layer(results=test_pose_encoding_scores, save_dir=test_pose_encoding_fig_dir, y_lim=(-0.1, 0.9))
        # plot_score_by_layer(results=cv_target_encoding_scores, save_dir=cv_target_encoding_fig_dir, y_lim=(-0.1, 0.9))



def get_4dhumans_top_scores(
    targets,
    pose_features=None,
    collect='target',      # 'target' or 'both'
    top_n=1,
    pca=None,              # PCA-aware routing to .../PCA{pca}/...
    save_path=None,
    show_progress=True,
) -> pd.DataFrame:
    """
    For 4DHumans (single model):
      1) Select top_n layers per target by CV score_r
         - If top_n == -1, use ALL layers available for that target (sorted by CV score_r desc).
      2) Collect test-set target_r for those layers
      3) If collect='both', also collect pose_r for same layers/features
    """

    if targets is None:
        raise ValueError("targets is None (must be a non-empty iterable of target names).")
    if collect not in ("target", "both"):
        raise ValueError("collect must be 'target' or 'both'.")
    if top_n != -1 and (not isinstance(top_n, int) or top_n < 1):
        raise ValueError(f"top_n must be a positive int or -1 (got {top_n}).")
    if pose_features is None:
        pose_features = []

    # ---- PCA-aware file paths ----
    if pca is None:
        cv_pkl   = Path("experiments/4DHuman_beh/cv/4DHuman_target_encoding.pkl")
        test_pkl = Path("experiments/4DHuman_beh/test/4DHuman_target_encoding.pkl")
        pose_pkl = Path("experiments/4DHuman_beh/test/4DHuman_pose_encoding.pkl")
        x_name   = SOTA_PLOT_NAME
    else:
        cv_pkl   = Path(f"experiments/4DHuman_beh/PCA{pca}/cv/4DHuman_target_encoding.pkl")
        test_pkl = Path(f"experiments/4DHuman_beh/PCA{pca}/test/4DHuman_target_encoding.pkl")
        pose_pkl = Path(f"experiments/4DHuman_beh/PCA{pca}/test/4DHuman_pose_encoding.pkl")
        x_name   = f"{SOTA_PLOT_NAME} (PCA{pca})"

    # ---- Load Pickles ----
    try:
        cv_df = pd.read_pickle(cv_pkl)
    except Exception as e:
        raise ValueError(f"Failed to read CV pickle at {cv_pkl}: {e}")
    try:
        test_df = pd.read_pickle(test_pkl)
    except Exception as e:
        raise ValueError(f"Failed to read TEST pickle at {test_pkl}: {e}")
    pose_df = None
    if collect == "both":
        try:
            pose_df = pd.read_pickle(pose_pkl)
        except Exception as e:
            raise ValueError(f"collect='both' but failed to read POSE pickle at {pose_pkl}: {e}")

    # ---- Iterate (optionally with tqdm) ----
    iterator = tqdm(targets, desc="Compiling 4DHumans data") if show_progress else targets

    rows = []
    for t in iterator:
        cv_sub = cv_df[cv_df['y'] == t]
        if cv_sub.empty:
            raise ValueError(f"No CV rows found for target '{t}' in {cv_pkl}.")

        # Determine which layers to collect
        if top_n == -1:
            # All layers for this target (sorted by CV score_r desc)
            top_layers = (
                cv_sub.sort_values('score_r', ascending=False)['layer_name']
                .astype(str).tolist()
            )
        else:
            top_layers = (
                cv_sub.sort_values('score_r', ascending=False)
                      .head(top_n)['layer_name']
                      .astype(str).tolist()
            )

        for layer in top_layers:
            # target_r from test_df
            tmask = (test_df['y'] == t) & (test_df['layer_name'] == layer)
            if not tmask.any():
                raise ValueError(
                    f"No TEST row for target '{t}' at layer '{layer}' in {test_pkl}."
                )
            target_r = test_df.loc[tmask, 'score_r'].iloc[0]

            if collect == 'target':
                # Also include y_hat if present
                if 'y_hat' in test_df.columns:
                    y_hat = test_df.loc[tmask, 'y_hat'].iloc[0]
                else:
                    y_hat = None  # keep robust even if column is absent

                rows.append({
                    'model_name': '3D joint model embeddings',
                    'layer_name': layer,
                    'x':          x_name,
                    'y':          t,
                    'score_r':    target_r,
                    'model_type': '4DHumans',
                    'y_hat':      y_hat
                })

            else:  # collect == 'both'
                if pose_df is None:
                    # Should not happen because we load pose_df above when collect=='both'
                    raise ValueError("Internal error: pose_df is None while collect='both'.")
                for feat in pose_features:
                    pmask = (pose_df['y'] == feat) & (pose_df['layer_name'] == layer)
                    if not pmask.any():
                        raise ValueError(
                            f"No POSE row for pose_feature '{feat}' at layer '{layer}' in {pose_pkl}."
                        )
                    pose_r = pose_df.loc[pmask, 'score_r'].iloc[0]
                    rows.append({
                        'x':            x_name,
                        'target':       t,
                        'target_r':     target_r,
                        'pose_feature': feat,
                        'pose_r':       pose_r,
                        'layer_name':   layer,
                        'model_name':   '4DHumans',
                        'model_type':   '4DHumans',
                    })

    df = pd.DataFrame(rows)

    if save_path is not None and not df.empty:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_pickle(save_path)

    return df
