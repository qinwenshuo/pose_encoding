import numpy as np

TARGET_RATING_PATH = 'data/raw/dyad_videos/behavioral_ratings.csv'
STIMULUS_DATA = 'data/raw/dyad_videos/stimulus_data.csv'
TRAIN_NAME = 'data/raw/dyad_videos/train.csv'
TEST_NAME = 'data/raw/dyad_videos/test.csv'
BETA_PATH = 'data/raw/dyad_videos/neural_scans/betas/'
ROI_MASK_PATH = 'data/raw/dyad_videos/neural_scans/localizers/'
RAW_VIDEO_PATH = "data/raw/dyad_videos/dyad_videos_3000ms_250/"
MESH_PATH = 'data/raw/dyad_videos/4d-humans/outputs/results/'
MESH_DEPTH_PATH = 'data/raw/dyad_videos/BEV/'

FEAT_INPUT_PATH = 'data/processed/dyad_videos/mesh_video_level_features'
TRAIN_IDX = 'data/processed/dyad_videos/train_idx.pkl'
TEST_IDX = 'data/processed/dyad_videos/test_idx.pkl'

AVAILABLE_TRAIN_NAMES = 'data/available_train.txt'
AVAILABLE_TEST_NAMES = 'data/available_test.txt'

MODEL_PATH = 'experiments/VACATION'
SOTA_MODEL_PATH = 'experiments/SOTA_layers'
SOTA_PLOT_NAME = 'Vision DNN embeddings'

RATING_OF_INTEREST = ['spatial expanse', 
                      'interagent distance', 'agents facing',
                      'communication', 'joint action']

ALPHAS = np.logspace(-10, 10, 20)
JOBS = -1
CV_SPLITS = 5
ALPHA_CV_SPLITS = 4
RANDOM = 0