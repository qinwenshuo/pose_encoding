import os
import shutil
import numpy as np
from tqdm import tqdm
from src.data_utils import load_pickle, save_json


EXPECTED_FRAMES = 90

DATASETS = [
    {
        'frame_level_path': 'data/processed/dyad_videos_500/mesh_frame_level_features/',
        'video_level_path': 'data/processed/dyad_videos_500/mesh_video_level_features/',
    },
    {
        'frame_level_path': 'data/processed/dyad_videos/mesh_frame_level_features/',
        'video_level_path': 'data/processed/dyad_videos/mesh_video_level_features/',
    },
]


def average_features(features, name, method='mean'):
    """
    Aggregate a list of feature vectors (or scalars) by mean, median, or median+IQR.

    Parameters
    ----------
    features : dict
        Mapping from feature‐name to list of arrays or floats.
    name : str
        The key in `features` whose list you want to aggregate.
    method : {'mean', 'median', 'median_iqr'}
        - 'mean': arithmetic mean
        - 'median': median only
        - 'median_iqr': returns an L×3 list of [Q1, median, Q3] per element

    Returns
    -------
    list or None
        - If method is 'mean' or 'median': a single list of length L.
        - If method is 'median_iqr': a list of L lists, each of length 3.
        - None if there are no valid (non‐None) entries.
    """
    matrix = features[name]

    # Figure out expected length (treat scalars as length‐1)
    lengths = {
        1 if isinstance(item, (float, np.floating)) else len(item)
        for item in matrix
        if item is not None
    }
    if not lengths:
        return None
    if len(lengths) > 1:
        raise ValueError(f"Feature {name} lengths do not match {lengths!r}")
    L = lengths.pop()

    # Collect valid entries into a (n_valid × L) array
    valid = []
    for item in matrix:
        if item is None:
            continue
        arr = np.asarray(item).reshape(-1)
        if arr.shape[0] != L:
            raise ValueError(f"Feature {name} has inconsistent length {arr.shape[0]} != {L}")
        valid.append(arr)
    data = np.vstack(valid)  # shape = (n_valid, L)

    if method == 'mean':
        # Return a list of length L
        return data.mean(axis=0).tolist()

    elif method == 'median':
        # Return a list of length L
        return np.median(data, axis=0).tolist()

    elif method == 'median_iqr':
        # Compute Q1, median, Q3 for each of the L positions
        q1 = np.percentile(data, 25, axis=0).tolist()[0]
        med = np.percentile(data, 50, axis=0).tolist()[0]
        q3 = np.percentile(data, 75, axis=0).tolist()[0]
        # Stack as L×3 and return as list of lists
        return [q1, med, q3]

    else:
        raise ValueError(f"Unknown method {method!r}; choose 'mean', 'median', or 'median_iqr'")




def _swap_halves_per_frame(frame_list):
    """
    Swap the two person-halves within every frame vector.

    Each frame vector is [person_A_feats | person_B_feats]. When a video needs
    its person identity corrected (detected from the averaged head_center_3d),
    we apply the same global swap to every frame so that across all videos the
    left-most and right-most person are encoded in a consistent position.
    """
    result = []
    for item in frame_list:
        if item is None:
            result.append(None)
            continue
        arr = np.asarray(item).reshape(-1)
        mid = len(arr) // 2
        result.append(np.concatenate([arr[mid:], arr[:mid]]).tolist())
    return result


def interpolate_frames(frame_list):
    """
    Fill None frames by linear interpolation between valid frames.
    Leading/trailing Nones are filled by copying the nearest valid frame value
    (no extrapolation). Returns a (n_frames, dim) array, or None if no valid
    frames exist at all.
    """
    dim = next(
        (len(np.asarray(item).reshape(-1)) for item in frame_list if item is not None),
        None,
    )
    if dim is None:
        return None

    n = len(frame_list)
    valid_idx = np.array([i for i, item in enumerate(frame_list) if item is not None])
    valid_vals = np.vstack([np.asarray(frame_list[i]).reshape(-1) for i in valid_idx])

    # np.interp copies edge values outside [valid_idx[0], valid_idx[-1]]
    all_idx = np.arange(n)
    result = np.column_stack([
        np.interp(all_idx, valid_idx, valid_vals[:, d])
        for d in range(dim)
    ])
    return result  # (n_frames, dim)


def concat_frames(frame_list):
    """
    Concatenate all frame vectors into one flat vector (preserves temporal order).
    None frames are filled via linear interpolation (boundary Nones copied from
    the nearest valid frame) so the output length is always n_frames × feature_dim.
    """
    filled = interpolate_frames(frame_list)
    if filled is None:
        return None
    return filled.reshape(-1).tolist()


def diff_frames(frame_list):
    """
    Mean of frame-to-frame differences (temporal derivative).
    None frames are filled via interpolation before differencing so temporal
    spacing is preserved. Returns a zero vector when fewer than 2 frames exist.
    """
    filled = interpolate_frames(frame_list)
    if filled is None:
        return None
    if len(filled) < 2:
        return np.zeros(filled.shape[1]).tolist()
    diffs = np.diff(filled, axis=0)  # (n_frames-1, dim)
    return diffs.mean(axis=0).tolist()


def std_frames(frame_list):
    """
    Per-dimension standard deviation across interpolated frames.
    Returns a zero vector when fewer than 2 frames exist.
    """
    filled = interpolate_frames(frame_list)
    if filled is None:
        return None
    if len(filled) < 2:
        return np.zeros(filled.shape[1]).tolist()
    return filled.std(axis=0).tolist()


def concat_avg_and_diff(avg_vec, diff_vec):
    """
    Concatenate the time-averaged feature vector with the mean temporal-diff
    vector into a single descriptor. Returns None if either input is None.
    """
    if avg_vec is None or diff_vec is None:
        return None
    return list(avg_vec) + list(diff_vec)


def concat_avg_diff_std(avg_vec, diff_vec, std_vec):
    """
    Concatenate mean, mean temporal-diff, and std into a single descriptor.
    Returns None if any input is None.
    """
    if avg_vec is None or diff_vec is None or std_vec is None:
        return None
    return list(avg_vec) + list(diff_vec) + list(std_vec)


def reorder_list(lst, force_reorder=False):
    if lst is None or len(lst) == 0:
        return None

    lst_len = len(lst)
    mid = lst_len // 2
    if force_reorder or lst[0] > lst[mid]:
        # Split into first half and second half, then swap them
        first_half = lst[:mid]
        second_half = lst[mid:]
        return second_half + first_half
    else:
        return lst.copy()


def process_dataset(frame_level_path, video_level_path):
    if os.path.exists(video_level_path):
        shutil.rmtree(video_level_path)
    os.makedirs(video_level_path, exist_ok=True)

    for file_name in tqdm(sorted(os.listdir(frame_level_path)), desc=video_level_path):
        if not file_name.endswith(".pkl"):
            continue

        pkl_path = os.path.join(frame_level_path, file_name)
        output_path = os.path.join(video_level_path, file_name.replace('.pkl', '.json'))
        features = load_pickle(pkl_path)

        # Frame count check
        n_frames = len([x for x in features['2d joints'] if x is not None])
        if n_frames != EXPECTED_FRAMES:
            print(f'WARNING: {file_name} has {n_frames} valid frames (expected {EXPECTED_FRAMES})')

        # --- Averaged features ---
        social_feat_2d  = average_features(features, '2d head direction + head center', method='mean')
        social_feat_3d  = average_features(features, '3d head direction + head center', method='mean')
        joints_2d       = average_features(features, '2d joints', method='mean')
        joints_3d       = average_features(features, '3d joints', method='mean')
        vertices_3d     = average_features(features, '3d vertices', method='mean')
        head_center_3d  = average_features(features, '3d head center', method='mean')
        head_dir_3d     = average_features(features, '3d head direction', method='mean')

        new_social_feat_2d = social_feat_2d
        new_social_feat_3d = social_feat_3d
        new_head_center_3d = reorder_list(head_center_3d)
        new_head_dir_3d    = head_dir_3d
        new_joints_2d      = joints_2d
        new_joints_3d      = joints_3d
        new_vertices_3d    = vertices_3d

        need_reorder = head_center_3d != new_head_center_3d
        if need_reorder:
            print(f'{file_name} features were split into half and swapped')
            new_social_feat_2d = reorder_list(social_feat_2d, force_reorder=True)
            new_social_feat_3d = reorder_list(social_feat_3d, force_reorder=True)
            new_joints_2d      = reorder_list(joints_2d,      force_reorder=True)
            new_joints_3d      = reorder_list(joints_3d,      force_reorder=True)
            new_vertices_3d    = reorder_list(vertices_3d,    force_reorder=True)
            new_head_dir_3d    = reorder_list(head_dir_3d,    force_reorder=True)

        # --- Raw frame lists, globally swapped if needed ---
        raw_social_2d = features['2d head direction + head center']
        raw_social_3d = features['3d head direction + head center']
        raw_joints_2d = features['2d joints']
        raw_joints_3d = features['3d joints']

        if need_reorder:
            raw_social_2d = _swap_halves_per_frame(raw_social_2d)
            raw_social_3d = _swap_halves_per_frame(raw_social_3d)
            raw_joints_2d = _swap_halves_per_frame(raw_joints_2d)
            raw_joints_3d = _swap_halves_per_frame(raw_joints_3d)

        diff_social_2d = diff_frames(raw_social_2d)
        diff_social_3d = diff_frames(raw_social_3d)
        diff_joints_2d = diff_frames(raw_joints_2d)
        diff_joints_3d = diff_frames(raw_joints_3d)

        std_social_2d  = std_frames(raw_social_2d)
        std_social_3d  = std_frames(raw_social_3d)
        std_joints_2d  = std_frames(raw_joints_2d)
        std_joints_3d  = std_frames(raw_joints_3d)

        output_feature = {
            # averaged
            '2D social pose features':                  new_social_feat_2d,
            '3D social pose features':                  new_social_feat_3d,
            '3D head directions':                       new_head_dir_3d,
            '3D head positions':                        new_head_center_3d,
            '2D body joints':                           new_joints_2d,
            '3D body joints':                           new_joints_3d,
            '3D body vertices':                         new_vertices_3d,
            # temporal concatenation (90 frames × feature_dim flattened)
            '2D social pose features (concat)':         concat_frames(raw_social_2d),
            '3D social pose features (concat)':         concat_frames(raw_social_3d),
            '2D body joints (concat)':                  concat_frames(raw_joints_2d),
            '3D body joints (concat)':                  concat_frames(raw_joints_3d),
            # temporal difference (mean frame-to-frame change)
            '2D social pose features (diff)':           diff_social_2d,
            '3D social pose features (diff)':           diff_social_3d,
            '2D body joints (diff)':                    diff_joints_2d,
            '3D body joints (diff)':                    diff_joints_3d,
            # averaged features + mean temporal diff
            '2D social pose features (mean+diff)':      concat_avg_and_diff(new_social_feat_2d, diff_social_2d),
            '3D social pose features (mean+diff)':      concat_avg_and_diff(new_social_feat_3d, diff_social_3d),
            '2D body joints (mean+diff)':               concat_avg_and_diff(new_joints_2d, diff_joints_2d),
            '3D body joints (mean+diff)':               concat_avg_and_diff(new_joints_3d, diff_joints_3d),
            # averaged features + mean temporal diff + std concatenated
            '2D social pose features (mean+diff+std)':  concat_avg_diff_std(new_social_feat_2d, diff_social_2d, std_social_2d),
            '3D social pose features (mean+diff+std)':  concat_avg_diff_std(new_social_feat_3d, diff_social_3d, std_social_3d),
            '2D body joints (mean+diff+std)':           concat_avg_diff_std(new_joints_2d, diff_joints_2d, std_joints_2d),
            '3D body joints (mean+diff+std)':           concat_avg_diff_std(new_joints_3d, diff_joints_3d, std_joints_3d),
        }

        save_json(output_feature, output_path)


for ds in DATASETS:
    process_dataset(ds['frame_level_path'], ds['video_level_path'])