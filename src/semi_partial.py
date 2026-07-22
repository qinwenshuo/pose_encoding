import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from himalaya.ridge import RidgeCV
from tqdm import tqdm

from src.data_utils import zscore_fit_apply, get_pose_features, get_target_ratings, align_multiple_vars
from src.config import ALPHAS, CV_SPLITS, ALPHA_CV_SPLITS

def semi_partial_corr(predictor1, predictor2, target, alphas=ALPHAS, cv=ALPHA_CV_SPLITS):
    """
    Block-level semi-partial correlation of predictor1 (multivariate) with target,
    controlling for predictor2, evaluated on a held-out test set.

    Parameters
    ----------
    predictor1 : dict with keys {'train': (n_tr, p), 'test': (n_te, p)}
    predictor2 : dict with keys {'train': (n_tr, q), 'test': (n_te, q)}
    target     : dict with keys {'train': (n_tr,), 'test': (n_te,)}
    alphas     : iterable of ridge strengths for CV (optional; defaults to ALPHAS or a logspace grid)
    cv         : inner-CV folds for alpha selection in both stages
    random_state : used only for reproducibility where applicable (kept for API symmetry)

    Returns
    -------
    dict with keys:
        - r: semi-partial correlation on test
        - r2: r**2 (unique variance explained on test)
        - yhat_test: predictions from residualized block on test
        - model_type: "ridge(himalaya)"
        - alpha_: chosen ridge alpha for stage 2 (if available; else None)
    """
    # ------------- coerce shapes -------------
    X1_tr = np.asarray(predictor1['train'])  # (n_tr, p)
    X1_te = np.asarray(predictor1['test'])   # (n_te, p)
    Z_tr  = np.asarray(predictor2['train'])
    Z_te  = np.asarray(predictor2['test'])
    y_tr  = np.asarray(target['train']).ravel()
    y_te  = np.asarray(target['test']).ravel()

    if X1_tr.ndim != 2 or X1_te.ndim != 2:
        raise ValueError("predictor1 must be 2D (n_samples, n_features).")
    if Z_tr.ndim == 1: Z_tr = Z_tr.reshape(-1, 1)
    if Z_te.ndim == 1: Z_te = Z_te.reshape(-1, 1)

    # ------------- helpers -------------
    def _get_alpha_attr(model):
        # Himalaya RidgeCV may expose alpha_ (scalar), alphas_ (grid),
        # or alpha_per_target_ (vector). Try to return something informative.
        if hasattr(model, "alpha_"):
            return float(np.asarray(model.alpha_).ravel()[0])
        if hasattr(model, "alpha_per_target_"):
            arr = np.asarray(model.alpha_per_target_).ravel()
            return float(np.nanmean(arr))
        return None

    # ========== Stage 1: residualize X1 against Z (train-only fit) ==========
    # Train-only standardization for Z and X1
    Z_tr_n, Z_te_n   = zscore_fit_apply(Z_tr, Z_te, axis=0)
    X1_tr_n, X1_te_n = zscore_fit_apply(X1_tr, X1_te, axis=0)

    # Multi-output regression: predict each column of X1 from Z using Himalaya RidgeCV
    # (Himalaya supports multi-target Y directly.)
    # reg_X1_on_Z = LinearRegression()
    reg_X1_on_Z = RidgeCV(alphas=alphas, fit_intercept=False, cv=cv)
    reg_X1_on_Z.fit(Z_tr_n, X1_tr_n)
    X1_hat_tr = reg_X1_on_Z.predict(Z_tr_n)
    X1_hat_te = reg_X1_on_Z.predict(Z_te_n)
    R1_tr = X1_tr_n - X1_hat_tr
    R1_te = X1_te_n - X1_hat_te

    # ========== Stage 2: predict target from residualized X1 (train-only fit) ==========
    # Standardize R1 and y on train only
    R1_tr_n, R1_te_n = zscore_fit_apply(R1_tr, R1_te, axis=0)
    y_tr_n,  y_te_n  = zscore_fit_apply(y_tr[:, None], y_te[:, None], axis=0)  # keep 2D, then ravel later

    stage2 = RidgeCV(alphas=alphas, fit_intercept=False, cv=cv)
    stage2.fit(R1_tr_n, y_tr_n)  # y can be (n,1)
    yhat_te = stage2.predict(R1_te_n).ravel()

    # Pearson r on test (scale-invariant, but we used standardized y for a clean intercept)
    r = pearsonr(yhat_te, y_te_n.ravel())[0]

    return {
        "r": float(r),
        "r2": float(r ** 2),
        "yhat_test": yhat_te,
        "alpha_": _get_alpha_attr(stage2),
    }


def get_semi_partial_scores(
    pose_feat_x, 
    pose_feat_z, 
    target_y,
    partial_name,
):
    
    X_data = get_pose_features(pose_feat_x)
    z_data = get_pose_features(pose_feat_z)
    target_data = get_target_ratings(target_y)

    X_data, z_data, target_data = align_multiple_vars(X_data, z_data, target_data)
    results = semi_partial_corr(X_data, z_data, target_data)
    # ['joints partial out \n2D social pose features', 'joints partial out \n3D social pose features']
    semipartial_results = {
        'x': partial_name,
        'y': target_y,
        'score_r': results['r']
    }
    return semipartial_results