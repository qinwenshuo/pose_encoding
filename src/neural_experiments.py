import os
import pandas as pd
import numpy as np
import nibabel as nib
from tqdm import tqdm
from itertools import combinations
from joblib import Parallel, delayed
from scipy.stats import spearmanr, pearsonr
from scipy.spatial.distance import pdist
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from sklearn.model_selection import RepeatedKFold
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RidgeCV as SKRidgeCV
from src.data_utils import read_idx_name, zscore_fit_apply, apply_pca, get_pose_features, align_multiple_vars
from src.config import ALPHAS, RANDOM, RATING_OF_INTEREST, TRAIN_NAME, TEST_NAME, SOTA_PLOT_NAME, STIMULUS_DATA, CV_SPLITS, ALPHA_CV_SPLITS
from src.config import FEAT_INPUT_PATH, TARGET_RATING_PATH, AVAILABLE_TRAIN_NAMES, AVAILABLE_TEST_NAMES



SUBJS = ['sub-01', 'sub-02', 'sub-03', 'sub-04']
ROIS = ['EVC', 'MT', 'EBA', 'pSTS', 'aSTS', 'FFA', 'PPA']

# =========== Encoding ===========

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
    
    def _nanmean_safe(arr):
        arr = np.asarray(arr, dtype=float)
        return float(np.nanmean(arr)) if arr.size else np.nan

    y_train = _ensure_2d_y(y_train)
    y_test  = _ensure_2d_y(y_test) if y_test is not None else None

    if eval_mode == "cv":
        # Build OOF predictions on the training set
        n = X_train.shape[0]
        n_targets = y_train.shape[1]
        y_oof = np.full((n, n_targets), np.nan, dtype=float)
        alpha_each_fold = []

        kf = RepeatedKFold(n_splits=cv_splits, n_repeats=2, random_state=RANDOM)
        for tr_idx, va_idx in kf.split(X_train):
            X_tr, X_va = X_train[tr_idx], X_train[va_idx]
            y_tr, y_va = y_train[tr_idx], y_train[va_idx]

            # Fold-only normalization
            X_tr_n, X_va_n = zscore_fit_apply(X_tr, X_va, axis=0)
            y_tr_n, y_va_n = zscore_fit_apply(y_tr, y_va, axis=0)
            _assert_finite("X_tr_n", X_tr_n); _assert_finite("X_va_n", X_va_n)
            _assert_finite("y_tr_n", y_tr_n); _assert_finite("y_va_n", y_va_n)

            # sklearn's RidgeCV with SVD solver
            ridge = SKRidgeCV(alphas=alphas)
            ridge.fit(X_tr_n, y_tr_n)
            y_va_pred = ridge.predict(X_va_n)
            if y_va_pred.ndim == 1:
                y_va_pred = y_va_pred[:, None]
            y_oof[va_idx, :] = y_va_pred

            # sklearn's RidgeCV stores selected alpha in ridge.alpha_
            alpha_each_fold.append(float(ridge.alpha_))

        # Scores on OOF predictions (train set)
        r_per = _pearson_per_target(y_train, y_oof)

        out = _nanmean_safe(r_per)
        
    elif eval_mode == "test":
        # Train-only normalization; predict on test
        X_tr_n, X_te_n = zscore_fit_apply(X_train, X_test, axis=0)
        y_tr_n, y_te_n = zscore_fit_apply(y_train, y_test, axis=0)
        _assert_finite("X_tr_n", X_tr_n); _assert_finite("X_te_n", X_te_n)
        _assert_finite("y_tr_n", y_tr_n); _assert_finite("y_te_n", y_te_n)

        # sklearn's RidgeCV with SVD solver
        ridge = SKRidgeCV(alphas=alphas)
        ridge.fit(X_tr_n, y_tr_n)
        y_te_pred = ridge.predict(X_te_n)
        r_per = _pearson_per_target(y_te_n.squeeze(), y_te_pred.squeeze())

        out = _nanmean_safe(r_per)
    else:
        raise ValueError("`eval_mode` must be 'test' or 'cv'.")

    return out




def get_neural_data(subjs, rois):

    print('[LOGGING] Loading neural data...')
    neural_data = []
    total_voxels = 0
    # Load split names
    train_names = read_idx_name(TRAIN_NAME)
    test_names  = read_idx_name(TEST_NAME)


    for subj_id in subjs:
        fmri_train_path = f'data/raw/dyad_videos/neural_scans/betas/{subj_id}/{subj_id}_space-T1w_desc-train-fracridge_data.nii.gz'
        fmri_test_path = f'data/raw/dyad_videos/neural_scans/betas/{subj_id}/{subj_id}_space-T1w_desc-test-fracridge_data.nii.gz'

        y_train_subj = nib.load(fmri_train_path).get_fdata()
        y_test_subj = nib.load(fmri_test_path).get_fdata()

        subject_path = f'data/raw/dyad_videos/neural_scans/localizers/{subj_id}'
        roi_mask_paths = []

        for roi in rois:
            for roi_mask_file_name in os.listdir(subject_path):
                if 'roi-' + roi in roi_mask_file_name and roi_mask_file_name.endswith('.nii.gz'):
                    if 'hemi-lh' in roi_mask_file_name:
                        roi_mask_paths.append((roi + '_lh', os.path.join(subject_path, roi_mask_file_name)))
                    elif 'hemi-rh' in roi_mask_file_name:
                        roi_mask_paths.append((roi + '_rh', os.path.join(subject_path, roi_mask_file_name)))
                    else:
                        print(f'Unrecognized ROI mask file name: {roi}')

        for roi, roi_mask_path in roi_mask_paths:
            roi_mask = nib.load(roi_mask_path).get_fdata().astype(bool)
            train_masked_beta_values = y_train_subj[roi_mask]
            test_masked_beta_values = y_test_subj[roi_mask]

            n_train_stimuli = y_train_subj.shape[-1]
            n_test_stimuli = y_test_subj.shape[-1]

            y_train_roi = train_masked_beta_values.reshape(-1, n_train_stimuli).T
            y_test_roi = test_masked_beta_values.reshape(-1, n_test_stimuli).T

            assert y_train_roi.shape[1] == y_test_roi.shape[1]
            n_voxels_in_roi = y_train_roi.shape[1]

            for voxel_idx in range(n_voxels_in_roi):
                y_train_voxel = y_train_roi[:, voxel_idx]
                y_test_voxel = y_test_roi[:, voxel_idx]

                if np.isnan(y_train_voxel).any() or np.isnan(y_test_voxel).any():
                    continue

                neural_data.append({
                    "subject": subj_id,
                    "roi": roi,
                    "voxel_idx": voxel_idx,
                    "train": y_train_voxel,
                    "test": y_test_voxel,
                    "train_names": train_names,
                    "test_names": test_names
                })

                total_voxels += 1

    return neural_data


def pose_neural_encoding(
        pose_features,
        method='ridge',
        eval_mode='test',
        alphas=ALPHAS,
        cv_splits=4,
        subjs=SUBJS,
        rois=ROIS,
        n_jobs=-1,
        pca_n_components=None,
        show_progress=False
):
    """
    Perform neural encoding across ROI and subjects for each pose feature.
    """
    if eval_mode not in ['test', 'cv']:
        raise ValueError("eval_mode must be either 'test' or 'cv'")

    records = {'model_class': [], 'score': [], 'ROI': [], 'subject': []}

    for feature_name in tqdm(pose_features, desc='Encoding in parallel'):
        pose_data = get_pose_features(feature_name)
        if pca_n_components is not None:
            pcaed_train, pcaed_test = apply_pca(pose_data['train'], pose_data['test'], n_components=pca_n_components)
            pose_data['train'] = pcaed_train
            pose_data['test'] = pcaed_test        
        for subj in subjs:
            for roi in rois:
                neural_data = get_neural_data(subjs=[subj], rois=[roi])
                jobs = []
                for voxel_data in neural_data:
                    pose_data, voxel_data = align_multiple_vars(pose_data, voxel_data)
                    # Build flat job list (feature × layer)
                    jobs.append((
                        pose_data['train'], pose_data['test'], 
                        voxel_data['train'], voxel_data['test'],
                        feature_name, None, voxel_data['voxel_idx'],
                        eval_mode, alphas, cv_splits
                    ))

                # Execute in parallel with one progress bar
                iterator = tqdm(jobs, desc=f"Target encoding") if show_progress else jobs
                results = Parallel(n_jobs=n_jobs)(
                    delayed(encode)(*job) for job in iterator
                )

                roi_score = float(np.nanmean(results))
                records['model_class'].append(feature_name)
                records['score'].append(roi_score)
                records['ROI'].append(roi)
                records['subject'].append(subj)

    df = pd.DataFrame(records)
    agg_df = (
        df.groupby(['model_class', 'ROI'], as_index=False)['score']
        .mean()
        .rename(columns={'score': 'mean_score'})
    )

    return agg_df



def sota_neural_encoding(
        model_name,
        method='ridge',
        eval_mode='test',
        alphas=ALPHAS,
        subjs=None,
        rois=None,
        cv_splits=4,
        n_jobs=-1,
):
    """
    Perform target-rating encoding for each layer × each rating,
    parallelizing only the inner loop over targets.
    """

    if eval_mode not in ['test', 'cv']:
        raise ValueError('eval_mode must be either test or cv')

    all_scores = {'layer_idx': [], 'layer_name': [], 'ROI': [], 'score': []}

    model_path = os.path.join(SOTA_MODEL_PATH, model_name)
    layer_files = [f for f in os.listdir(model_path) if f.endswith('.npz')]
    layers_sorted = order_files(layer_files)
    neural_data = get_neural_data(subjs=subjs, rois=rois)

    for layer_idx, filename in tqdm(enumerate(layers_sorted), total=len(layers_sorted), desc=f'Encoding {model_name}'):
        layer_name = filename.rsplit('.npz', 1)[0]
        path = os.path.join(model_path, filename)
        train_repr, test_repr = get_sota_model_layers(path)
        for subj, subj_rois in neural_data.items():
            for roi, roi_file in subj_rois.items():
                # 4) build encoding tasks
                n_voxels = roi_file['train'].shape[1]
                tasks = [
                    (v_idx, train_repr, test_repr, roi_file,
                     method, eval_mode, alphas, cv_splits)
                    for v_idx in range(n_voxels)
                ]
                # 5) run in parallel
                scores = Parallel(n_jobs=n_jobs)(
                    delayed(_encode_voxel)(*task) for task in tasks
                )
                roi_score = np.mean(scores)
                all_scores['layer_name'].append(layer_name)
                all_scores['score'].append(roi_score)
                all_scores['ROI'].append(roi)
                all_scores['layer_idx'].append(layer_idx)

    df = pd.DataFrame(all_scores)
    return df


# ===================== Compile Encoding Results =====================
def get_sota_encoding_scores(
    rois,
    subjs,
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
            test_result = Path(f"experiments/SOTA_neural/{model_type}/test")
            test_fig    = Path(f"results/SOTA/{model_type}/test")
            cv_result   = Path(f"experiments/SOTA_neural/{model_type}/cv")
            cv_fig      = Path(f"results/SOTA/{model_type}/cv")
        else:
            test_result = Path(f"experiments/SOTA_neural/PCA{pca}/{model_type}/test")
            test_fig    = Path(f"results/SOTA/PCA{pca}/{model_type}/test")
            cv_result   = Path(f"experiments/SOTA_neural/PCA{pca}/{model_type}/cv")
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



# ## ==================== RSA ====================


# def _rsa_roi(feat_name,
#              train_feat,
#              test_feat,
#              roi,
#              roi_file,
#              eval_mode,
#              feat_metric='euclidean',
#              neural_metric='correlation',
#              rsa_mode='standard'):
#     """
#     Compute RSA between model features and neural responses within one ROI.

#     Parameters
#     ----------
#     feat_name     : str
#     train_feat    : ndarray, shape (n_train_samples, n_features)
#     test_feat     : ndarray, shape (n_test_samples,  n_features)
#     roi           : str
#     roi_file      : dict with keys 'train' and 'test', arrays shape (n_samples, n_voxels)
#     eval_mode     : one of 'train', 'test', 'all'
#     feat_metric   : distance metric for feature RDM (passed to pdist)
#     neural_metric : distance metric for neural RDM (passed to pdist)
#     rsa_mode      : one of
#                     - 'standard'           : classical RSA
#                     - 'feature_reweighted' : learns feature weights to maximize RSA
#                     - 'searchlight'        : voxel-wise RSA map

#     Returns
#     -------
#     dict
#       For 'standard' and 'feature_reweighted', returns {
#           'model_class': feat_name,
#           'ROI':         roi,
#           'rsa_score':   <Spearman r>,
#           **('weights':  <array of feature weights> if feature_reweighted)
#       }
#       For 'searchlight', returns {
#           'model_class': feat_name,
#           'ROI':         roi,
#           'rsa_map':     ndarray, shape (n_voxels,)
#       }
#     """

#     # --- 1) Select split ---
#     train_neural = roi_file['train']
#     test_neural  = roi_file['test']
#     if eval_mode == 'train':
#         X = train_feat; Y = train_neural
#     elif eval_mode == 'test':
#         X = test_feat;  Y = test_neural
#     else:  # 'all'
#         X = np.vstack([train_feat, test_feat])
#         Y = np.vstack([train_neural, test_neural])

#     # --- 2) Z-score ---
#     X, _ = z_score_normalize(X)
#     Y, _ = z_score_normalize(Y)

#     # --- 3) Standard RSA setup ---
#     # We’ll always need the feature RDM in some form
#     rdm_X = pdist(X, metric=feat_metric)

#     if rsa_mode == 'standard':
#         # classical RSA
#         rdm_Y = pdist(Y, metric=neural_metric)
#         rsa_score = spearmanr(rdm_X, rdm_Y).correlation
#         return {
#             'model_class': feat_name,
#             'ROI':         roi,
#             'score':   rsa_score
#         }

#     elif rsa_mode == 'feature_reweighted':
#         # learn a set of feature weights so that weighted RDM_X best aligns with RDM_Y
#         # 1) build design matrix of pairwise absolute feature diffs
#         pairs = list(combinations(range(X.shape[0]), 2))
#         pair_diffs = np.vstack([np.abs(X[i] - X[j]) for i, j in pairs])

#         # 2) target is the neural RDM
#         rdm_Y = pdist(Y, metric=neural_metric)

#         # 3) fit a ridge regression (you can adjust alphas as needed)
#         reg = RidgeCV(alphas=np.logspace(-6, 6, 13), cv=5)
#         reg.fit(pair_diffs, rdm_Y)
#         weights = reg.coef_

#         # 4) compute weighted-feature RSA
#         X_weighted = X * weights  # broadcast across samples
#         rdm_X_wr = pdist(X_weighted, metric=feat_metric)
#         rsa_score = spearmanr(rdm_X_wr, rdm_Y).correlation

#         return {
#             'model_class': feat_name,
#             'ROI':         roi,
#             'score':   rsa_score,
#             'weights':     weights
#         }

#     elif rsa_mode == 'searchlight':
#         # voxel-wise RSA: returns one RSA score per voxel
#         n_voxels = Y.shape[1]
#         rsa_map = np.zeros(n_voxels)

#         # reuse rdm_X from above
#         for v in range(n_voxels):
#             # RDM for this single voxel across samples
#             rdm_Y_v = pdist(Y[:, [v]], metric=neural_metric)
#             rsa_map[v] = spearmanr(rdm_X, rdm_Y_v).correlation

#         return {
#             'model_class': feat_name,
#             'ROI':         roi,
#             'score':     np.mean(rsa_map)
#         }

#     else:
#         raise ValueError(f"Unknown rsa_mode: {rsa_mode}")


# def pose_neural_rsa(
#         pose_features,
#         eval_mode='all',
#         feat_metric='euclidean',
#         neural_metric='correlation',
#         rsa_mode='standard',
#         n_jobs=-1
# ):
#     """
#     Run ROI-based RSA over a set of pose feature representations.

#     Parameters
#     ----------
#     pose_features : iterable of feature names
#     eval_mode     : 'train', 'test', or 'all'
#     n_jobs        : number of parallel jobs
#     """
#     if eval_mode not in ['train', 'test', 'all']:
#         raise ValueError("eval_mode must be one of 'train', 'test', or 'all'")

#     all_results = []
#     for feat_name in tqdm(pose_features, desc='ROI-based RSA'):
#         # load features and subject splits
#         train_feat, test_feat, train_names, test_names = get_pose_features(feat_name)
#         # load neural data dict: { subj_id: { ROI_name: roi_file_dict, ... }, ... }
#         neural_data = get_neural_data(train_names, test_names)

#         # prepare one task per (subj, ROI)
#         tasks = [
#             (feat_name, train_feat, test_feat, roi, roi_file, eval_mode, feat_metric, neural_metric, rsa_mode)
#             for subj_rois in neural_data.values()
#             for roi, roi_file in subj_rois.items()
#         ]

#         # parallel RSA
#         rsa_dicts = Parallel(n_jobs=n_jobs)(
#             delayed(_rsa_roi)(*task) for task in tasks
#         )
#         all_results.extend(rsa_dicts)

#     # aggregate across subjects by averaging RSA per (feature, ROI)
#     df = pd.DataFrame(all_results)
#     agg_df = df.groupby(['model_class', 'ROI'], as_index=False)['score'].mean()
#     return agg_df


# def my_model_neural_rsa(
#         eval_mode='all',
#         feat_metric='euclidean',
#         neural_metric='correlation',
#         rsa_mode='standard',
#         n_jobs=-1
# ):
#     """
#     Compute RSA scores between model representations and target ratings.

#     Parameters:
#       eval_mode (str): 'test', 'train', or 'all'
#       feature_metric (str): metric for feature RDM (default: 'correlation')
#       target_metric (str): metric for target RDM (default: 'euclidean')

#     Returns:
#       all_rsa_results (list of dict): with keys 'feature', 'target', 'rsa_score'
#     """
#     if eval_mode not in ['train', 'test', 'all']:
#         raise ValueError("eval_mode must be one of 'train', 'test', or 'all'")

#     all_results = []
#     # Load and normalize features and targets
#     for model_type, params in MODEL_DICT.items():
#         param_list = [f'{param}{value}' for param, value in params[0].items()]
#         param_string = '_'.join(['bootstrap'] + param_list)
#         layer = params[1]
#         activation_path = os.path.join(MODEL_PATH, model_type, param_string)
#         for model_name in tqdm(sorted(os.listdir(activation_path)), desc=f'Encoding {model_type}'):
#             model_path = os.path.join(activation_path, model_name, f'{layer}.pkl')
#             train_repr, test_repr, train_names, test_names = get_my_model_layers(model_path)
#             # load neural data dict: { subj_id: { ROI_name: roi_file_dict, ... }, ... }
#             neural_data = get_neural_data(train_names, test_names)

#             # prepare one task per (subj, ROI)
#             tasks = [
#                 (change_x_ticks(model_type), train_repr, test_repr, roi, roi_file, eval_mode, feat_metric, neural_metric)
#                 for subj_rois in neural_data.values()
#                 for roi, roi_file in subj_rois.items()
#             ]

#             # parallel RSA
#             rsa_dicts = Parallel(n_jobs=n_jobs)(
#                 delayed(_rsa_roi)(*task) for task in tasks
#             )
#             all_results.extend(rsa_dicts)

#     # aggregate across subjects by averaging RSA per (feature, ROI)
#     df = pd.DataFrame(all_results)
#     agg_df = df.groupby(['model_class', 'ROI'], as_index=False)['score'].mean()
#     return agg_df

