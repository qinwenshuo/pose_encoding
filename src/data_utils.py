import os
import csv
import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.decomposition import PCA
from src.config import ALPHAS, RANDOM, RATING_OF_INTEREST, TRAIN_NAME, TEST_NAME, SOTA_PLOT_NAME, STIMULUS_DATA, CV_SPLITS, ALPHA_CV_SPLITS
from src.config import FEAT_INPUT_PATH, TARGET_RATING_PATH, AVAILABLE_TRAIN_NAMES, AVAILABLE_TEST_NAMES


def load_pickle(path):
    with open(path, 'rb') as f:
        pickled = pickle.load(f)
    return pickled


def save_pickle(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)


def read_json(path):
    with open(path, 'r') as file:
        data = json.load(file)
    return data

def save_json(data, path):
    with open(path, 'w') as file:
        json.dump(data, file, indent=4)


def read_idx_name(f, extension=True):
    with open(f, newline='', encoding='utf-8') as file:
        reader = csv.reader(file)
        next(reader)  # Skip the first row (header)
        if extension:
            return [r[0] for r in reader]
        if not extension:
            return [r[0].replace('.mp4', '') for r in reader]
        else:
            return []
       

def align_multiple_vars(*vars):
    """
    Align multiple variables by *external* train/test name lists stored in .txt files.
    
    Each var must be a dict with keys:
        - "train": np.ndarray (samples x features or similar)
        - "test":  np.ndarray
      Additional keys are preserved and returned.

    The .txt files must contain one name per line. Empty lines are ignored.
    Order in the files determines the output order.

    Raises
    ------
    ValueError
        - if any required name (from the .txt files) is missing in any variable
        - if there are duplicate names within a variable's name list
    """
    def _read_names(path: str | Path) -> list[str]:
        names = []
        with open(path, "r") as f:
            for line in f:
                name = line.strip()
                if name:  # skip empty lines
                    names.append(name)
        # Preserve order; remove exact duplicates while keeping first
        seen = set()
        deduped = []
        for n in names:
            if n not in seen:
                seen.add(n)
                deduped.append(n)
        return deduped

    def _check_no_duplicates(names: list[str], which: str, var_ix: int):
        seen = set()
        dups = set()
        for n in names:
            if n in seen:
                dups.add(n)
            seen.add(n)
        if dups:
            raise ValueError(
                f"Variable #{var_ix} has duplicate entries in {which}_names: {sorted(dups)}"
            )

    # Load the target alignment names (order-defining)
    target_train = _read_names(AVAILABLE_TRAIN_NAMES)
    target_test  = _read_names(AVAILABLE_TEST_NAMES)

    aligned_vars = []
    for i, v in enumerate(vars, start=1):
        # Basic key checks
        required = {"train", "test", "train_names", "test_names"}
        missing_keys = required - set(v.keys())
        if missing_keys:
            raise ValueError(f"Variable #{i} is missing keys: {sorted(missing_keys)}")

        tr = np.asarray(v["train"])
        te = np.asarray(v["test"])
        tr_names = list(v["train_names"])
        te_names = list(v["test_names"])

        # Sanity: name lengths must match data rows
        if tr.shape[0] != len(tr_names):
            raise ValueError(
                f"Variable #{i}: len(train_names)={len(tr_names)} does not match train.shape[0]={tr.shape[0]}"
            )
        if te.shape[0] != len(te_names):
            raise ValueError(
                f"Variable #{i}: len(test_names)={len(te_names)} does not match test.shape[0]={te.shape[0]}"
            )

        # No duplicates inside each provided list
        _check_no_duplicates(tr_names, "train", i)
        _check_no_duplicates(te_names, "test", i)

        # Build fast lookup
        tr_index = {name: idx for idx, name in enumerate(tr_names)}
        te_index = {name: idx for idx, name in enumerate(te_names)}

        # Verify all required names exist and build index lists in the order of target files
        missing_train = [n for n in target_train if n not in tr_index]
        missing_test  = [n for n in target_test  if n not in te_index]

        if missing_train or missing_test:
            msgs = []
            if missing_train:
                msgs.append(f"missing TRAIN names: {missing_train}")
            if missing_test:
                msgs.append(f"missing TEST names: {missing_test}")
            raise ValueError(f"Variable #{i} is missing required names from files: " + "; ".join(msgs))

        tr_idx = [tr_index[n] for n in target_train]
        te_idx = [te_index[n] for n in target_test]

        # Slice and rebuild dict (preserve extra keys)
        aligned_v = {k: val for k, val in v.items()
                     if k not in ["train", "test", "train_names", "test_names"]}

        aligned_v.update({
            "train": tr[tr_idx],
            "test": te[te_idx],
            "train_names": target_train,
            "test_names": target_test,
        })
        aligned_vars.append(aligned_v)

    return aligned_vars



# ======================
# Get Features Functions
# ======================


def get_target_ratings(target_name):
    if target_name not in RATING_OF_INTEREST:
        raise ValueError(
            f"Invalid target: {target_name}. Choose from: {', '.join(RATING_OF_INTEREST)}"
        )

    # Load ratings (video_name + requested target)
    cols = ['video_name', target_name]
    data_df = pd.read_csv(TARGET_RATING_PATH, usecols=cols).set_index('video_name')

    # Load split names
    train_names = read_idx_name(TRAIN_NAME)
    test_names  = read_idx_name(TEST_NAME)

    # Check for missing videos
    missing_train = set(train_names) - set(data_df.index)
    missing_test  = set(test_names) - set(data_df.index)
    if missing_train or missing_test:
        raise ValueError(
            f"Missing ratings for videos:\n"
            f"  Train: {sorted(missing_train)}\n"
            f"  Test: {sorted(missing_test)}"
        )

    # Align strictly to split order
    train_df = data_df.reindex(train_names)
    test_df  = data_df.reindex(test_names)

    # Build matrices (1D arrays)
    train_matrix = train_df[target_name].to_numpy(dtype=float).ravel()
    test_matrix  = test_df[target_name].to_numpy(dtype=float).ravel()

    return {
        "type": target_name,
        "train_names": list(train_names),
        "test_names": list(test_names),
        "train": train_matrix,
        "test": test_matrix,
    }


def get_pose_features(pose_name):
    all_train = read_idx_name(TRAIN_NAME) # list of 200 original training sample names
    all_test = read_idx_name(TEST_NAME)   # list of 50 original test sample names
    # Lists to collect data
    train_data = []
    test_data = []
    # Collect available video names for the target to match
    available_train = []
    available_test = []
    # Loop over files in the directory
    for video_name in all_train:
        json_path = os.path.join(FEAT_INPUT_PATH, video_name.replace('.mp4', '.json'))
        row = read_json(json_path)[pose_name]
        if row is not None and None not in row:
            train_data.append(row)
            available_train.append(video_name)
    for video_name in all_test:
        json_path = os.path.join(FEAT_INPUT_PATH, video_name.replace('.mp4', '.json'))
        row = read_json(json_path)[pose_name]
        if row is not None and None not in row:
            test_data.append(row)
            available_test.append(video_name)
    
    # Convert lists to NumPy matrices (arrays)
    train_matrix = np.array(train_data)
    test_matrix = np.array(test_data)
    pose_feature_data = {
        'type': pose_name,
        'train_names': available_train,
        'test_names': available_test,
        'train': train_matrix,
        'test': test_matrix,
    }
    return pose_feature_data


def get_sota_model_layers(layer_path, pca):
    """
    Loads representations from an .npz file and splits into train/test using CSV metadata by index alignment.
    Returns:
        train_reps, test_reps, train_names, test_names
    """
    p = Path(layer_path)


    # ---- Infer model_type, model_name, layer_name robustly ----
    parts = p.parts
    if len(parts) < 3:
        model_type = None
        if len(parts) < 2:
            raise ValueError(f"Path too short to infer type/model/layer: {layer_path}")
    else:
        model_type = parts[-3]
    model_name = parts[-2]
    layer_name = p.stem  # filename without extension

    # ---- Load split metadata once, using zero-copy where possible ----
    split_df = pd.read_csv(STIMULUS_DATA)

    if 'stimulus_set' not in split_df.columns:
        raise ValueError("CSV must contain a 'stimulus_set' column with 'train'/'test' values.")

    # Boolean masks (NumPy arrays for fast indexing)
    stim_set = split_df['stimulus_set'].to_numpy(copy=False)
    train_mask = (stim_set == 'train')
    test_mask  = (stim_set == 'test')

    # ---- Load features (supports .npz with 'feature_map' or single array) ----
    suffix = p.suffix.lower()
    if suffix == '.npz':
        data = np.load(p)
        try:
            feature_map = data['feature_map']  # 2D matrix where each row corresponds to CSV row
        except (KeyError, IndexError):
            feature_map = data
    else:
        raise ValueError(f"Unsupported file extension: {p}. Use .npz")

    # Ensure 2D array
    feature_map = np.asarray(feature_map)
    if feature_map.ndim != 2:
        raise ValueError(f"feature_map must be 2D, got shape {feature_map.shape}")

    # ---- Validate alignment ----
    if feature_map.shape[0] != len(split_df):
        raise ValueError(
            f"Feature rows ({feature_map.shape[0]}) must equal CSV rows ({len(split_df)})."
        )

    # ---- Split reps ----
    train_reps = feature_map[train_mask]
    test_reps  = feature_map[test_mask]

    # ---- Optional PCA ----
    if pca is not None:
        train_reps, test_reps = apply_pca(train_reps, test_reps, n_components=pca)

    # ---- Names (zero-copy to NumPy, then list) ----
    names = split_df['video_name'].to_numpy(copy=False)
    train_names = names[train_mask].tolist()
    test_names  = names[test_mask].tolist()

    return {
        'type': SOTA_PLOT_NAME,
        'model_type': model_type,
        'model_name': model_name,
        'layer_name': layer_name,
        'train_names': train_names,
        'test_names': test_names,
        'train': train_reps,
        'test': test_reps,
    }


# =============================================
# Helper functions to order the SOTA layers by depth
# =============================================

class Node:
    def __init__(self, line, depth, name):
        self.line = line
        self.depth = depth
        self.name = name
        self.children = []


def parse_filename(filename):
    base = os.path.splitext(filename)[0]
    if base.startswith('0-Net'):
        return 0, 1, filename
    parts = base.split('-')
    if len(parts) < 4:
        return None, None, filename
    return int(parts[0]), int(parts[2]), filename


def build_tree(files):
    # 1) parse and sort by (line, depth)
    nodes = []
    for f in files:
        parsed = parse_filename(f)
        if None not in parsed:
            nodes.append(parsed)
    nodes.sort(key=lambda x: (x[0], x[1]))

    if not nodes:
        return None

    # 2) pick the very first as 'root'
    root_line, root_depth, root_name = nodes[0]
    root = Node(root_line, root_depth, root_name)

    stack = [root]
    # 3) iterate the rest
    for line, depth, name in nodes[1:]:
        curr = Node(line, depth, name)

        # pop until we find something strictly shallower
        while stack and stack[-1].depth >= depth:
            stack.pop()

        if stack:
            # attach to the nearest shallower ancestor
            stack[-1].children.append(curr)
        else:
            # nothing shallower? just hang it off the root
            root.children.append(curr)

        stack.append(curr)

    return root


def post_order(node, result):
    if node is None:
        return
    for c in node.children:
        post_order(c, result)
    result.append(node.name)


def order_files(model_dir):
    files = [p.stem for p in Path(model_dir).glob("*.npz")]
    root = build_tree(files)
    result = []
    post_order(root, result) 
    return result


def zscore_fit_apply(A_fit, A_apply, axis=0):
    """Fit μ/σ on A_fit, apply to both. Handles all-NaN and zero-variance columns.
    Returns finite arrays (NaNs/Infs mapped to 0 after standardization)."""
    A_fit = np.asarray(A_fit, dtype=np.float64)
    A_apply = np.asarray(A_apply, dtype=np.float64)

    # compute mean/std ignoring NaNs
    mu = np.nanmean(A_fit, axis=axis, keepdims=True)
    sd = np.nanstd(A_fit, axis=axis, keepdims=True)

    # bad if mean or std is non-finite or std==0
    bad = ~np.isfinite(mu) | ~np.isfinite(sd) | (sd == 0)
    # for bad cols: use mu=0, sd=1 so standardized values become 0
    mu = np.where(bad, 0.0, mu)
    sd = np.where(bad, 1.0, sd)

    A_fit_z = (A_fit - mu) / sd
    A_apply_z = (A_apply - mu) / sd

    # ensure finite (map NaN/Inf to 0)
    A_fit_z = np.nan_to_num(A_fit_z, nan=0.0, posinf=0.0, neginf=0.0)
    A_apply_z = np.nan_to_num(A_apply_z, nan=0.0, posinf=0.0, neginf=0.0)
    return A_fit_z, A_apply_z



def apply_pca(train_data, test_data, n_components):
    if train_data.shape[1] < n_components:
        raise ValueError(f"Cannot apply PCA with n_components={n_components} greater than number of features {train_data.shape[1]}")
    pca = PCA(n_components=n_components)
    pca.fit(train_data)
    train_pca = pca.transform(train_data)
    test_pca = pca.transform(test_data)
    return train_pca, test_pca


