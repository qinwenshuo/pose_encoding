"""
Nested-CV DNN encoding on the 500-video set, for supp_fig_2.py only.

Adds exactly one thing on top of figure_2.py's original methodology: an outer
5-fold rotation (data/raw/dyad_videos/folds_500.csv, fixed/shared across every
model/job so results are comparable) so the honest test score isn't locked to one
predefined train/test split. Everything else -- the inner-loop layer selection and
the LOO/GCV alpha selection inside every ridge fit -- is exactly figure_2's original
encode() calls, unmodified and reused as-is (eval_mode='cv' with CV_SPLITS=5, i.e.
RepeatedKFold(5, n_repeats=2), for selection; eval_mode='test' for honest scoring).

For each outer fold: the other 4 folds (400 videos) are train, this fold (100
videos) is test. Within that 400, every layer is scored via encode(eval_mode='cv',
cv_splits=CV_SPLITS) -- figure_2's exact inner-CV call -- and the best-scoring layer
per target is selected. That layer is then fit on the full 400 and evaluated via
encode(eval_mode='test') on the 100 held-out test videos, never touched during
selection. Repeating over all 5 outer folds covers all 500 videos as test exactly
once; supp_fig_2.py pools those out-of-fold predictions into a single Pearson r per
model/target.

Saves, per model:
    experiments/SOTA_beh_500_nested/{model_type}/inner/{model}_inner_cv.pkl
        every layer's inner-CV r, per outer_fold x target (for auditing/completeness)
    experiments/SOTA_beh_500_nested/{model_type}/pooled/{model}_pooled.pkl
        the z-scored (true, predicted) pairs for every video's honest held-out
        prediction, per target -- everything supp_fig_2.py needs to pool into a
        final r without redoing any compute.
"""
import argparse
from math import ceil
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

from src.config import ALPHAS, CV_SPLITS, JOBS, SOTA_MODEL_PATH
from src.data_utils import get_sota_model_layers_500, get_target_ratings_500, order_files, save_pickle, zscore_fit_apply
from src.encoding import encode

FOLDS_500_PATH = 'data/raw/dyad_videos/folds_500.csv'
NESTED_BEH_PATH = 'experiments/SOTA_beh_500_nested'
N_OUTER = 5

targets_500 = {
    'spatial_expanse': 'spatial expanse',
    'agent_distance':  'interagent distance',
    'communication':   'communication',
    'joint_action':    'joint action',
}


def _load_matched_layers(model_type, model_name, n_jobs, show_progress):
    """Same original/additional_250 layer-name matching as sota_target_encoding_500;
    duplicated here (rather than imported) since this script's methodology is
    specific to supp_fig_2 and not meant to be reused elsewhere."""
    model_dir = Path(SOTA_MODEL_PATH) / model_type / model_name
    add_dir   = Path(SOTA_MODEL_PATH) / f"{model_type}_additional_250" / model_name
    layer_files = order_files(str(model_dir))
    if not layer_files:
        raise ValueError(f"No layers found in {model_dir}")

    available_add = {p.stem for p in add_dir.glob("*.npz")}
    matched = [l for l in layer_files if l in available_add]
    n_skipped = len(layer_files) - len(matched)
    if n_skipped:
        print(f"[Warning] {model_type}/{model_name}: {n_skipped}/{len(layer_files)} layers from the "
              f"original 250 have no matching filename in the additional 250 set; skipping those.")
    if not matched:
        print(f"[Warning] {model_type}/{model_name}: no layer filenames match between "
              f"{model_dir} and {add_dir} -- skipping this model entirely.")
        return None

    sota_jobs = [(model_type, model_name, layer_name) for layer_name in matched]
    iterator = tqdm(sota_jobs, desc="[LOGGING] Loading 500-video layers...") if show_progress else sota_jobs
    layers_data = Parallel(n_jobs=n_jobs)(delayed(get_sota_model_layers_500)(*job) for job in iterator)
    return layers_data


def nested_cv_encode_model(model_type, model_name, n_jobs=JOBS, show_progress=True, save_dir=None):
    layers_data = _load_matched_layers(model_type, model_name, n_jobs, show_progress)
    if layers_data is None:
        return None, None

    video_names = layers_data[0]['names']
    folds_df = pd.read_csv(FOLDS_500_PATH).set_index('video_name')
    fold_of = folds_df.loc[video_names, 'fold'].to_numpy()

    inner_rows = []
    # pooled[target] holds, for every video, its z-scored true/pred pair from
    # whichever outer fold used it as test -- filled in exactly once per video.
    pooled_true = {t: np.full(len(video_names), np.nan) for t in targets_500.values()}
    pooled_pred = {t: np.full(len(video_names), np.nan) for t in targets_500.values()}
    selected_layers = []

    y_by_target = {t: get_target_ratings_500(col, video_names) for col, t in targets_500.items()}

    outer_iter = tqdm(range(N_OUTER), desc=f"{model_type}/{model_name} outer folds") if show_progress else range(N_OUTER)
    for outer_fold in outer_iter:
        train_mask = fold_of != outer_fold
        test_mask = fold_of == outer_fold

        for target_display, y in y_by_target.items():
            y_train, y_test = y[train_mask], y[test_mask]

            # Inner-loop layer selection: figure_2's exact call (encode(), unmodified,
            # eval_mode='cv', cv_splits=CV_SPLITS -> RepeatedKFold(5, n_repeats=2)),
            # run within this outer fold's 400 train videos only.
            inner_jobs = [
                (ld['X'][train_mask], None, y_train, None, ld['type'], ld['layer_name'], target_display,
                 'cv', ALPHAS, CV_SPLITS)
                for ld in layers_data
            ]
            inner_results = Parallel(n_jobs=n_jobs)(delayed(encode)(*job) for job in inner_jobs)
            for r in inner_results:
                inner_rows.append({'outer_fold': outer_fold, 'y': target_display,
                                    'layer_name': r['layer_name'], 'inner_cv_r': r['score_r']})

            best = max(inner_results, key=lambda r: r['score_r'])
            best_layer_name, best_inner_r = best['layer_name'], best['score_r']
            selected_layers.append({'outer_fold': outer_fold, 'y': target_display,
                                     'best_layer': best_layer_name, 'inner_cv_r': best_inner_r})

            ld_best = next(ld for ld in layers_data if ld['layer_name'] == best_layer_name)
            X_train, X_test = ld_best['X'][train_mask], ld_best['X'][test_mask]
            test_result = encode(
                X_train, X_test, y_train, y_test,
                ld_best['type'], best_layer_name, target_display,
                eval_mode='test', alphas=ALPHAS,
            )
            # recompute the matching z-scored true test y (same call encode() made
            # internally) so pooling stays on a consistent basis with the prediction
            _, y_test_n = zscore_fit_apply(y_train[:, None], y_test[:, None], axis=0)

            test_idx_positions = np.where(test_mask)[0]
            pooled_pred[target_display][test_idx_positions] = test_result['y_hat']
            pooled_true[target_display][test_idx_positions] = y_test_n.ravel()

    inner_df = pd.DataFrame(inner_rows)
    pooled_rows = []
    for target_display in targets_500.values():
        assert not np.isnan(pooled_pred[target_display]).any(), "every video should be predicted exactly once"
        for name, true_v, pred_v in zip(video_names, pooled_true[target_display], pooled_pred[target_display]):
            pooled_rows.append({'y': target_display, 'video_name': name, 'y_true_z': true_v, 'y_pred_z': pred_v})
    pooled_df = pd.DataFrame(pooled_rows)
    selected_df = pd.DataFrame(selected_layers)

    if save_dir:
        inner_path = Path(save_dir) / "inner" / f"{model_name}_inner_cv.pkl"
        pooled_path = Path(save_dir) / "pooled" / f"{model_name}_pooled.pkl"
        inner_path.parent.mkdir(parents=True, exist_ok=True)
        pooled_path.parent.mkdir(parents=True, exist_ok=True)
        save_pickle(inner_df, str(inner_path))
        save_pickle({'pooled': pooled_df, 'selected_layers': selected_df}, str(pooled_path))

    return inner_df, pooled_df


def get_nested_500_encoding_scores(overwrite=False, task_id=None, max_tasks=20):
    sota_model_path = Path(SOTA_MODEL_PATH)

    def list_common_models(model_type):
        orig_root = sota_model_path / model_type
        add_root = sota_model_path / f"{model_type}_additional_250"
        if not orig_root.exists() or not add_root.exists():
            return []
        orig_models = {d.name for d in orig_root.iterdir() if d.is_dir()}
        add_models = {d.name for d in add_root.iterdir() if d.is_dir()}
        return sorted(orig_models & add_models)

    combined = []
    for mt in ["video_models", "image_models"]:
        for m in list_common_models(mt):
            combined.append((mt, m))
    total_models = len(combined)
    if total_models == 0:
        print("[Logging] No models found. Nothing to do.")
        return

    if task_id is not None:
        if not (1 <= task_id <= max_tasks):
            raise ValueError(f"task_id must be between 1 and {max_tasks} inclusive.")
        chunk_size = ceil(total_models / max_tasks)
        start_idx = (task_id - 1) * chunk_size
        end_idx = min(task_id * chunk_size, total_models)
        combined = combined[start_idx:end_idx]
        print(f"Task {task_id}: processing models {start_idx}-{end_idx - 1} ({len(combined)} total).")

    subset_types = {mt for mt, _ in combined}
    for mt in subset_types:
        (Path(NESTED_BEH_PATH) / mt / "inner").mkdir(parents=True, exist_ok=True)
        (Path(NESTED_BEH_PATH) / mt / "pooled").mkdir(parents=True, exist_ok=True)

    pbar = tqdm(combined, desc="500-video nested-CV encoding")
    for model_type, model in pbar:
        pbar.set_description(f"Nested CV: {model_type}/{model}")
        pooled_path = Path(NESTED_BEH_PATH) / model_type / "pooled" / f"{model}_pooled.pkl"
        if (not overwrite) and pooled_path.exists():
            print(f"[Logging] {model_type}/{model} already encoded (nested CV), skipped.")
            continue
        inner_df, pooled_df = nested_cv_encode_model(
            model_type, model, save_dir=str(Path(NESTED_BEH_PATH) / model_type),
        )
        if inner_df is None:
            print(f"[Logging] {model_type}/{model} skipped (no matching layers between video halves).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run nested-CV 500-video DNN encoding by task ID.")
    parser.add_argument("--task_id", type=int, required=True)
    parser.add_argument("--max_tasks", type=int, required=True)
    args = parser.parse_args()
    get_nested_500_encoding_scores(overwrite=True, task_id=args.task_id, max_tasks=args.max_tasks)
