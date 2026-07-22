import os
import cv2
import pickle
import smplx
import torch
import joblib
import shutil
import argparse
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed
from src.config import RAW_VIDEO_PATH, MESH_PATH, MESH_DEPTH_PATH


## Input Paths:
img_base_path = 'data/raw/dyad_videos/4d-humans'
video_ratings_path = 'data/raw/dyad_videos/video_ratings.csv'
## Output Paths:
output_video_base_path = "data/processed/dyad_videos_500/mesh_annotated_videos/"
output_feature_base_path = "data/processed/dyad_videos_500/mesh_frame_level_features/"
# Rewrite the output directory and all its contents for each run
if os.path.exists(output_video_base_path): shutil.rmtree(output_video_base_path)
if os.path.exists(output_feature_base_path): shutil.rmtree(output_feature_base_path)
os.makedirs(output_video_base_path, exist_ok=True)
os.makedirs(output_feature_base_path, exist_ok=True)

color_mapping = {
    0: (0, 0, 255),
    1: (0, 255, 0),
    2: (255, 0, 0),
    3: (0, 0, 255),
    4: (0, 255, 0),
    5: (255, 0, 0),
}

# Lazy per-process SMPL loader to avoid pickling issues
_smpl_model = None
def get_smpl_model():
    global _smpl_model
    if _smpl_model is None:
        _smpl_model = smplx.create(model_path="SMPL_NEUTRAL.pkl", model_type="smpl")
    return _smpl_model


def draw_meshes(views, pose, color, W, H):
    front_view, top_view = views
    _joints = pose['joints']
    camera = pose['camera translation']
    front_tz = pose['4Dhuman tz']

    x = _joints[:, 0]
    y = _joints[:, 1]
    z = _joints[:, 2]
    tx = camera[0]
    ty = camera[1]
    top_tz = camera[2]

    # Project joints for the front view.
    front_proj_x = (x + tx) * 10000 / (z + front_tz) + W // 2
    front_proj_y = (y + ty) * 10000 / (z + front_tz) + H // 2
    front_points = np.column_stack([front_proj_x, front_proj_y]).astype(int)

    # Project joints for the top view.
    top_proj_x = (x + tx) * 100 + W // 5
    top_proj_z = (z + top_tz) * 100

    # Convert projected coordinates and invert z-axis for the top view.
    top_points = []
    for (xi, zi) in np.column_stack([top_proj_x, top_proj_z]).astype(int):
        zi_inverted = H - zi
        top_points.append((xi, zi_inverted))
    top_points = np.array(top_points)

    # Define skeleton connectivity for SMPL joints.
    skeleton = [
        (0, 1), (0, 2), (0, 3),
        (1, 4), (2, 5), (3, 6),
        (4, 7), (5, 8), (6, 9),
        (7, 10), (8, 11), (9, 12),
        (12, 13), (12, 14), (12, 15),
        (13, 16), (14, 17), (16, 18), (17, 19),
        (18, 20), (19, 21), (20, 22), (21, 23)
    ]

    # Draw joints (as circles) on the front view.
    for (xi, yi) in front_points:
        if 0 <= xi < W and 0 <= yi < H:
            cv2.circle(front_view, (xi, yi), 2, color, -1)

    # Draw joints on the top view.
    for (xi, zi) in top_points:
        if 0 <= xi < W and 0 <= zi < H:
            cv2.circle(top_view, (xi, zi), 2, color, -1)

    # Draw lines connecting joints based on the skeleton connectivity.
    for i, j in skeleton:
        if i < len(front_points) and j < len(front_points):
            pt1_front = tuple(front_points[i])
            pt2_front = tuple(front_points[j])
            cv2.line(front_view, pt1_front, pt2_front, color, 2)

            pt1_top = tuple(top_points[i])
            pt2_top = tuple(top_points[j])
            cv2.line(top_view, pt1_top, pt2_top, color, 2)

    return [front_view, top_view]


def draw_features(views, pose, W, H):
    color = (139, 0, 0)
    front_view, top_view = views
    front_tz = pose['4Dhuman tz']
    facing_direction = pose['3d head direction']
    head_center = pose['3d head center']

    # For head_center (arrow base)
    fx_base = head_center[0] * 10000 / (head_center[2] + front_tz) + W / 2
    fy_base = head_center[1] * 10000 / (head_center[2] + front_tz) + H / 2

    # For arrow end point (head_center + facing_direction)
    arrow_end = head_center + facing_direction
    fx_arrow = arrow_end[0] * 10000 / (arrow_end[2] + front_tz) + W / 2
    fy_arrow = arrow_end[1] * 10000 / (arrow_end[2] + front_tz) + H / 2

    cv2.arrowedLine(front_view, (int(fx_base), int(fy_base)), (int(fx_arrow), int(fy_arrow)), color, 3)

    # Top view projection
    tx_base = head_center[0] * 100 + W // 5
    tz_base =  head_center[2] * 100
    ty_base = H - tz_base

    tx_arrow = arrow_end[0] * 100 + W // 5
    tz_arrow =  arrow_end[2]  * 100
    ty_arrow = H - tz_arrow

    cv2.arrowedLine(top_view, (int(tx_base), int(ty_base)), (int(tx_arrow), int(ty_arrow)), color, 2)

    return [front_view, top_view]


def draw_annotation(img, features, W, H):
    front_img = img.copy()
    top_down_img = np.full((H, W, 3), 255, dtype=np.uint8)
    frame_views = [front_img, top_down_img]
    for id, pose in features.items():
        color = color_mapping.get(id, (0, 0, 0))
        frame_views = draw_meshes(frame_views, pose, color, W, H)
        frame_views = draw_features(frame_views, pose, W, H)
    return frame_views


def get_head_direction(translated_joints):
    # neck: 15, mid shoulder: 12, nose: 24, r eye: 25, l eye: 26
    left_eye = translated_joints[26]
    right_eye = translated_joints[25]
    eye_midpoint = (left_eye + right_eye) / 2.0
    neck_pos = translated_joints[15]
    nose_pos = translated_joints[24]

    v_neck_nose = nose_pos - neck_pos
    v_eye_nose = nose_pos - eye_midpoint

    gaze_vector = (v_neck_nose + v_eye_nose) / 2.0
    norm_gaze = np.linalg.norm(gaze_vector)
    if norm_gaze > 0:
        gaze_vector_normalized = (gaze_vector / norm_gaze) * 0.2
    else:
        gaze_vector_normalized = gaze_vector
    return gaze_vector_normalized, eye_midpoint


def process_one_video(video_file, output_videos):
    """
    Process a single video and return (video_base_name, num_identified_for_video).
    This function is designed to be called in parallel across videos.
    """
    smpl_model = get_smpl_model()

    video_base_name = video_file.replace('.mp4', '')

    # Read 4D Human mesh file
    mesh = joblib.load(os.path.join(MESH_PATH, f'demo_{video_base_name}.pkl'))
    # BEV input mesh files
    bev_folder = os.path.join(MESH_DEPTH_PATH, video_base_name)

    output_feature_path = os.path.join(output_feature_base_path, f'{video_base_name}.pkl')
    output_video_path = os.path.join(output_video_base_path, video_file)

    # Derive ordered frame keys from the mesh — format-agnostic across both
    # the original and additional 250 videos which may have different key prefixes.
    frame_keys = sorted(k for k in mesh if f'/{video_base_name}/img/' in k)
    if not frame_keys:
        print(f'Warning: no frame keys found in mesh for {video_base_name}. Skipping.')
        return (video_base_name, 0)

    # Locate the mp4, checking both video directories.
    video_path = None
    for _dir in [RAW_VIDEO_PATH, 'data/raw/dyad_videos/additional_250/']:
        _candidate = os.path.join(_dir, video_file)
        if os.path.exists(_candidate):
            video_path = _candidate
            break
    if video_path is None:
        print(f'Warning: mp4 not found for {video_file}. Skipping.')
        return (video_base_name, 0)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f'Could not open video {video_path}. Skipping.')
        return (video_base_name, 0)

    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if output_videos:
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (W * 2, H))
    else:
        out = None

    video_features = {
        '3d head direction + head center': [],
        '2d head direction + head center': [],
        '3d head center': [],
        '3d head direction': [],
        '2d joints': [],
        '3d joints': [],
        '3d vertices': []
    }
    video_num_identified = []

    frame_num = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            if frame_num != 90:
                print(f'Warning {video_file} only has {frame_num}/90 frames read successfully.')
            break

        if frame_num >= len(frame_keys):
            print(f'Warning {video_file}: video frames ({frame_num + 1}) exceed mesh frame keys ({len(frame_keys)}). Stopping.')
            break

        frame_key = frame_keys[frame_num]
        num_mesh = len(mesh[frame_key]["tracked_ids"])

        # Read BEV mesh file by frame
        bev_file = os.path.join(bev_folder, f'{frame_num:08d}__2_0.08.npz')
        bev_cam = None
        num_bev = 0
        if os.path.exists(bev_file):
            bev = np.load(bev_file, allow_pickle=True)['results'][()]
            bev_cam = bev['cam_trans'].copy()
            num_bev = bev['verts'].shape[0]

        frame_data = {}

        for detection_idx in range(min(num_bev, num_mesh)):
            tz = bev_cam[detection_idx, 2]
            tx, ty, mesh_tz = mesh[frame_key]["camera"][detection_idx]
            cam_trans = np.array([tx, ty, tz])

            smpl_params = mesh[frame_key]["smpl"][detection_idx]
            betas = torch.tensor(smpl_params["betas"]).unsqueeze(0)
            global_orient = torch.tensor(smpl_params["global_orient"]).unsqueeze(0)
            body_pose = torch.tensor(smpl_params["body_pose"]).unsqueeze(0)

            # Generate vertices
            output = smpl_model(
                betas=betas,
                global_orient=global_orient,
                body_pose=body_pose,
                pose2rot=False
            )

            joints = output.joints.detach().numpy().squeeze()
            joints_3d = joints + cam_trans
            vertices = output.vertices.detach().numpy().squeeze()
            vertices_3d = vertices + cam_trans

            # Project the 3D joints into 2D plane, then calculate the 2D features
            denom = joints[:, 2] + mesh_tz
            joints_2d = joints_3d[:, :2] / denom[:, np.newaxis] * 10000 + np.array([[W / 2, H / 2]])

            head_direction_3d, head_center_3d = get_head_direction(joints_3d)
            head_direction_2d, head_center_2d = get_head_direction(joints_2d)

            frame_data[detection_idx] = {
                'joints': joints,
                'camera translation': cam_trans,
                '4Dhuman tz': mesh_tz,
                '3d joints': joints_3d,
                '2d joints': joints_2d,
                '3d vertices': vertices_3d,
                '3d head direction': head_direction_3d,
                '2d head direction': head_direction_2d,
                '3d head center': head_center_3d,
                '2d head center': head_center_2d,
            }
            video_num_identified.append(detection_idx)

        if output_videos:
            annotated_frame = draw_annotation(frame, frame_data, W, H)
            out.write(np.hstack(annotated_frame))

        if len(frame_data) == 2:
            video_features['3d head direction + head center'].append(
                np.hstack((
                    frame_data[0]['3d head direction'],
                    frame_data[0]['3d head center'],
                    frame_data[1]['3d head direction'],
                    frame_data[1]['3d head center']))
            )
            video_features['2d head direction + head center'].append(
                np.hstack((
                    frame_data[0]['2d head direction'],
                    frame_data[0]['2d head center'],
                    frame_data[1]['2d head direction'],
                    frame_data[1]['2d head center']))
            )
            video_features['3d head center'].append(
                np.hstack((
                    frame_data[0]['3d head center'],
                    frame_data[1]['3d head center']))
            )
            video_features['3d head direction'].append(
                np.hstack((
                    frame_data[0]['3d head direction'],
                    frame_data[1]['3d head direction']))
            )
            video_features['2d joints'].append(
                np.hstack((
                    frame_data[0]['2d joints'].ravel(),
                    frame_data[1]['2d joints'].ravel()))
            )
            video_features['3d joints'].append(
                np.hstack((
                    frame_data[0]['3d joints'].ravel(),
                    frame_data[1]['3d joints'].ravel()))
            )
            video_features['3d vertices'].append(
                np.hstack((
                    frame_data[0]['3d vertices'].ravel(),
                    frame_data[1]['3d vertices'].ravel()))
            )
        else:
            video_features['3d head direction + head center'].append(None)
            video_features['2d head direction + head center'].append(None)
            video_features['3d head center'].append(None)
            video_features['3d head direction'].append(None)
            video_features['2d joints'].append(None)
            video_features['3d joints'].append(None)
            video_features['3d vertices'].append(None)

        frame_num += 1

    # Save features for this video
    with open(output_feature_path, 'wb') as f:
        pickle.dump(video_features, f)

    if out is not None:
        out.release()
    cap.release()

    num_identified_for_video = (max(video_num_identified) + 1) if len(video_num_identified) else 0
    return (video_base_name, num_identified_for_video)


parser = argparse.ArgumentParser(description='Process video features with optional video output')
parser.add_argument('--output_videos', action='store_true', help='Whether to output annotated videos')
parser.add_argument('--jobs', type=int, default=-1, help='Number of parallel workers for per-video processing')
args = parser.parse_args()

# ----------------------- Parallel Main -----------------------
import csv
with open(video_ratings_path, newline='') as _f:
    _reader = csv.reader(_f)
    next(_reader)  # skip header
    video_files = sorted(row[0] for row in _reader)

# Run per video in parallel
results = Parallel(n_jobs=args.jobs, backend="loky")(
    delayed(process_one_video)(video_file, args.output_videos) for video_file in tqdm(video_files, desc="Scheduling Videos")
)

# Aggregate results
num_identified = {video_name: num for video_name, num in results}

# Report videos that do not have exactly 2 identified people
for video_name, num in num_identified.items():
    if num != 2:
        print(video_name, num)