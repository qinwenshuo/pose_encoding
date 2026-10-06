
# Pose Features Encoding

This repository implements the full pipeline described in
**“Simple 3D Pose Features Support Human and Machine Social Scene Understanding” (Qin & Isik, 2025)**.
It extracts interpretable **3D visuospatial pose representations** from dyadic interaction videos and encodes them to predict **human social interaction ratings**.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://drive.google.com/file/d/1jEi38gYnJTkpiH_PkgAHBWq7GqPLw46D/view?usp=sharing)

## 1. Installation

### System Requirements

Our code relies on CPU cores and was tested and runnable on Linux, Windows 11, and MacOS. However, if you would like to extract pose meshes on your own using [4D Humans](https://github.com/shubham-goel/4D-Humans) and [BEV](https://github.com/Arthur151/ROMP/blob/master/simple_romp/README.md), special system and GPU configurations for their pipelines might be required. Recommended hardware: ≥32 CPUs, ≥32 GB RAM, GPU optional.

To facilitate reproduction, we recommend checking out our [colab demo](https://drive.google.com/file/d/1jEi38gYnJTkpiH_PkgAHBWq7GqPLw46D/view?usp=sharing). The data generated in this study, including intermediate pose features and encoding scores, are available from Zenodo (see Section 3).

### Create Environment

Installation is simple and should take ~< 10 minutes on a normal desktop computer

```bash
conda env create -f environment.yml
conda activate pose
```

## 2. Directory Structure

Paths are defined in `src/config.py`. The `data/`, `experiments/` and `results/` folders are not tracked by git; they are created by the steps below or filled from the Zenodo deposit.

```
pose_encoding/
│
├── data/
│   ├── raw/
│   │   ├── dyad_videos/
│   │   │   ├── dyad_videos_3000ms_250/        # original 250 MiT videos (not redistributed)
│   │   │   ├── additional_250/                # additional 250 MiT videos (not redistributed)
│   │   │   ├── behavioral_ratings.csv
│   │   │   ├── stimulus_data.csv
│   │   │   ├── stimulus_data_additional_250.csv
│   │   │   ├── video_ratings.csv              # ratings for the 500-video set
│   │   │   ├── train.csv
│   │   │   ├── test.csv
│   │   │   ├── 4d-humans/outputs/results/     # 4D Humans tracking output (demo_<video>.pkl)
│   │   │   └── BEV/                           # BEV depth output
│   │   └── synthetic_videos/
│   │       ├── position.csv
│   │       ├── facing.csv
│   │       └── videos/
│   ├── processed/
│   │   ├── dyad_videos/                       # 250-video set
│   │   │   ├── mesh_frame_level_features/
│   │   │   └── mesh_video_level_features/
│   │   ├── dyad_videos_500/                   # 500-video set
│   │   │   ├── mesh_frame_level_features/
│   │   │   └── mesh_video_level_features/
│   │   └── grouped_models.csv
│   ├── available_train.txt                    # videos with valid pose estimates (train)
│   └── available_test.txt                     # videos with valid pose estimates (test)
│
├── experiments/
│   ├── SOTA_beh/                              # encoding scores of the 351 vision DNNs
│   ├── 4DHuman_beh/                           # encoding scores of the 4D Humans embeddings
│   ├── ridge_results/                         # grouped ridge (DNN + 3D pose) scores
│   └── all_models_list.csv
│
├── scripts/
│   ├── 00_classify_model_families.py
│   ├── 01_frame_level_features.py
│   ├── 02_video_level_features.py
│   ├── 03_pose_model_encoding.py
│   ├── 03_dnn_encoding.py
│   ├── 04_grouped_encoding.py
│   ├── 05_make_folds_500.py
│   ├── 06_dnn_nested_cv500_encoding.py
│   ├── figure_2.py
│   ├── figure_3.py
│   ├── figure_4.py
│   ├── figure_5.py
│   ├── figure_6.py
│   ├── supp_fig_1.py
│   ├── supp_fig_2.py
│   ├── supp_fig_3.py
│   ├── supp_fig_4.py
│   ├── supp_fig_5.py
│   ├── supp_fig_6.py
│   ├── supp_fig_7.py
│   └── supp_table_1.py
│
├── src/
├── slurms/
├── logs/
│   └── parallel/
│
├── SMPL_NEUTRAL.pkl                           # SMPL model, download separately (Section 4)
├── environment.yml
└── README.md
```

---

## 3. Data

All data generated in this study are deposited at Zenodo (DOI: 10.5281/zenodo.23195014; CC BY 4.0). The deposit includes:

* behavioral ratings and train/test splits for the 250-video and 500-video sets;
* frame-level and video-level 3D pose features (body joints, 3D/2D positions and facing directions) for the 500-video set;
* the DNN family assignments (`grouped_models.csv`);
* per-model encoding scores for the 351 vision DNNs and the 4D Humans embeddings (the `experiments/` folder);
* the synthetic videos and the human communication ratings from the online experiment;
* the source data for all figures.

The deposit does **not** include:

* The Moments in Time video clips, which cannot be redistributed under the dataset license. Request them from [moments.csail.mit.edu](http://moments.csail.mit.edu); the video names needed to retrieve them are in the deposit.
* The raw 4D Humans and BEV output files. The frame-level and video-level features derived from them are included.
* The DNN embeddings, because of their size (see Section 6.1).
* The SMPL body model, which must be downloaded from its authors after registration (see Section 4).

### 3.1 Where to put the deposited files

| Zenodo file or folder | Location in this repository |
| --- | --- |
| `250_train.csv`, `250_test.csv` | `data/raw/dyad_videos/train.csv`, `data/raw/dyad_videos/test.csv` (rename) |
| `500_video_ratings.csv` | `data/raw/dyad_videos/video_ratings.csv` (rename) |
| `synthetic_videos/` | `data/raw/synthetic_videos/` |
| `dyad_videos_500/` | `data/processed/dyad_videos_500/` |
| `grouped_models.csv` | `data/processed/grouped_models.csv` |
| `experiments/` | `experiments/` (keep the folder structure) |
| `250_video_ratings.csv`, `500_video_stats.csv`, `additional_250_video_names.csv` | `behavioral_ratings.csv`, `stimulus_data.csv`, `stimulus_data_additional_250.csv`|

The main (250-video) analyses read `data/processed/dyad_videos/mesh_video_level_features/`. The deposited `dyad_videos_500/` folder contains the files of all 500 videos, including the original 250. Copy the files of the videos listed in `train.csv` and `test.csv` into `data/processed/dyad_videos/`.

### 3.2 Reproducing the figures from the deposited data

No GPU, DeepJuice or Moments in Time video is needed for this path.

1. Download the Zenodo deposit and place the files as in Section 3.1.
2. Create the environment (Section 1).
3. Run the figure scripts in Section 6.5.

The figure scripts read the saved encoding scores in `experiments/SOTA_beh/<model_type>/{cv,test}/<model>_target_encoding.pkl` (and `<model>_pose_encoding.pkl` in `test/`), `experiments/4DHuman_beh/`, and `experiments/ridge_results/`.

---

## 4. Data Preparation

This section is only needed if you want to re-extract poses from the videos. This pipeline is built for **dyadic interaction videos**, but it can be adapted for other two-person video datasets. 

1. **Obtain Dataset**

   * Request the videos from the [Moments in Time dataset](http://moments.csail.mit.edu).
   * Keep only the videos that are in our train and test set. Train and test video names are under `data/raw/dyad_videos/train.csv` and `data/raw/dyad_videos/test.csv`
   * Place these files under:

     ```
     data/raw/dyad_videos/dyad_videos_3000ms_250/
     ```

   * For the 500-video analyses, also place the additional 250 videos (names in `additional_250_video_names.csv`) under:

     ```
     data/raw/dyad_videos/additional_250/
     ```

2. **Download the SMPL Neutral Model**

   4D Humans (HMR 2.0) predicts the parameters of the **SMPL** body model (Loper et al., 2015), not SMPL-X. Register at [smpl.is.tue.mpg.de](https://smpl.is.tue.mpg.de), download the neutral model `basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl`, and place it in the project directory. You can rename it to `SMPL_NEUTRAL.pkl` for compatibility. See also the [4D Humans installation guide](https://github.com/shubham-goel/4D-Humans?tab=readme-ov-file#installation-and-setup). The SMPL model files are subject to their own license and must not be redistributed.

3. **Extract Poses with 4D Humans**

   * Follow [4D Humans setup guide](https://github.com/shubham-goel/4D-Humans)
     → “Run tracking demo on videos”.
   * Save results to (one `demo_<video>.pkl` per video):

     ```
     data/raw/dyad_videos/4d-humans/outputs/results/
     ```

4. **Correct Depth with BEV**

   * Use [BEV](https://github.com/Arthur151/ROMP/blob/master/simple_romp/README.md) to compute 3D body and depth.
   * Save results to:

     ```
     data/raw/dyad_videos/BEV/
     ```

The raw 4D Humans and BEV outputs from Steps 3 and 4 are not included in the Zenodo deposit. The frame-level and video-level features derived from them (Section 5) are.

---

## 5. Feature Extraction

Run sequentially:

```bash
conda activate pose
```
```
python -u -m scripts.01_frame_level_features --output_videos
```
Rebuilds each person's SMPL joints from the 4D Humans parameters with the `smplx` package, adds the camera translation (with BEV depth), and writes each frame's 3D body joints and social pose features (positions + facing directions) to `data/processed/dyad_videos_500/mesh_frame_level_features/<video>.pkl`. Annotated videos are written to `data/processed/dyad_videos_500/mesh_annotated_videos/` when `--output_videos` is set.

The 45 joints are the 24 SMPL skeleton joints plus 21 additional keypoints that `smplx` defines as fixed vertices on the SMPL mesh (nose, eyes and ears; big toe, small toe and heel of each foot; ten fingertips).

```
python -u -m scripts.02_video_level_features
```
Outputs each video's averaged 3D body joints and 3D social pose features from 90 frames to `data/processed/dyad_videos_500/mesh_video_level_features/<video>.json` (and the corresponding folders for the 250-video set). Sorted so that the first person is always the left most person in the video.


## 6. Encoding

Encoding is parallelized across CPU cores.


### 6.1 Model Embedding Extraction

All model embeddings were extracted using the DeepJuice package (except for 4D Humans) following the same procedures as Garcia et al. For embedding extraction code, please refer to https://github.com/Isik-lab/SIfMRI_modeling.

DeepJuice (Conwell et al., 2024) is not publicly released (it is currently in private beta). To obtain access, contact its author, Colin Conwell (conwell@g.harvard.edu). Instructions are also included in the README of the SIfMRI_modeling repository.

The extracted embeddings are not deposited because of their size and are available from the authors on request. The per-model encoding scores derived from them are deposited at Zenodo (Section 3), so the figure scripts in Section 6.5 can be run without DeepJuice.

### 6.2 Pose Model Encoding

Predict social ratings using pose model (4D Humans) embeddings:

```bash
python -u -m scripts.03_pose_model_encoding
```

### 6.3 Parallel DNN Encoding (HPC Example)

Run across multiple models in parallel using SLURM:

```bash
#!/bin/bash
#SBATCH --partition=your_partition
#SBATCH -A your_account
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=48
#SBATCH --array=1-351
#SBATCH --job-name=parallel
#SBATCH --output=logs/parallel/parallel_%a.log

ml load anaconda3/2024.02-1
conda activate pose

python -u -m scripts.03_dnn_encoding --task_id $SLURM_ARRAY_TASK_ID --max_tasks 351
python -u -m scripts.04_grouped_encoding --task_id $SLURM_ARRAY_TASK_ID --max_tasks 351
```

Scores are written to `experiments/SOTA_beh/<model_type>/{cv,test}/`.

### 6.3b DNN Encoding on the 500-Video Set (Supplemental)

First build the fixed fold assignment (shared by every model/job so
results are comparable), then run the encoding as a SLURM array:

```bash
python -u -m scripts.05_make_folds_500
sbatch slurms/parallel_dnn_500.sh
```

Saves per-model results to `experiments/SOTA_beh_500_nested/`.

### 6.4 Model Family Classification

Before running the DNN-family figures, classify every benchmarked model into an architecture/training family (e.g. CLIP, DINO, ConvNext, ResNet/ResNext/SE) based on its model UID:

```bash
python -u -m scripts.00_classify_model_families
```

Reads `experiments/all_models_list.csv` and writes `data/processed/grouped_models.csv`. The resulting file is included in the Zenodo deposit, so this step can be skipped.

### 6.5 Figures and Supplemental Analyses

Each figure in the paper is produced by its own dedicated script (these replace the old, single `05_behavioral_encoding.py` pipeline). All scripts read previously saved encoding results (from `experiments/SOTA_beh`, `experiments/4DHuman_beh` and `experiments/ridge_results`) and save plots/tables to `results/`. `figure_6.py` and `supp_fig_7.py` read the synthetic-experiment ratings from `data/raw/synthetic_videos/`:

```bash
python -u -m scripts.figure_2
python -u -m scripts.figure_3
python -u -m scripts.figure_4
python -u -m scripts.figure_5
python -u -m scripts.figure_6
python -u -m scripts.supp_fig_1
python -u -m scripts.supp_fig_2
python -u -m scripts.supp_fig_3
python -u -m scripts.supp_fig_4
python -u -m scripts.supp_fig_5
python -u -m scripts.supp_fig_6
python -u -m scripts.supp_fig_7
python -u -m scripts.supp_table_1
```

## 7. Outputs

| Output Type                 | Description                                                                                                |
| --------------------------- | ---------------------------------------------------------------------------------------------------------- |
| **3D Joints**               | 45 keypoints (x, y, z) per person averaged per video                                                       |
| **3D Social Pose Features** | Head position and facing direction per agent                                                               |
| **Encoding Scores**         | Pearson correlation between predicted and actual human ratings                                             |
| **Plotted Data**            |  `results/` contains all the figures and tables; `results/plot_data/<figure>/` contains CSVs of exactly what each figure plots                                |



## 8. Expected Runtime

| Task                             | Approx. Duration (per 250 videos) | Hardware |
| -------------------------------- | --------------------------------- | -------- |
| Pose extraction (4D Humans)      | ~2–3 h                            | GPU      |
| BEV depth correction             | ~1 h                              | GPU      |
| Feature extraction               | ~5 min                           | CPU      |
| Encoding (per model)             | ~2–3 min                          | CPU      |
| Full group encoding (351 models) | ~1 h on 48 cores                  | HPC      |


## 9. Citation

If you use this code, please cite the paper, the code archive, and the data deposit:

```
@article{qin2025simple3dpose,
  title={Simple 3D Pose Features Support Human and Machine Social Scene Understanding},
  author={Qin, Wenshuo and Isik, Leyla},
  journal={arXiv preprint arXiv:2511.03988},
  year={2025}
}

@dataset{qin_pose_encoding_data,
  title={Data for ``Simple 3D Pose Features Support Human and Machine Social Scene Understanding''},
  author={Qin, Wenshuo and Isik, Leyla},
  publisher={Zenodo},
  doi={10.5281/zenodo.23195014},
  year={2026}
}
```


## 10. License and Acknowledgements

This repository is released under the MIT License. The data deposit is released under CC BY 4.0.

This repository builds on:

* [4D Humans](https://github.com/shubham-goel/4D-Humans)
* [BEV (Bird’s-Eye View Estimation)](https://github.com/Arthur151/ROMP)
* [350+ Vision DNN Benchmarking](https://github.com/Isik-lab/SIfMRI_modeling)
* DeepJuice (Conwell et al., 2024; not publicly released, see Section 6.1)
* [Himalaya](https://github.com/gallantlab/himalaya)
* The [SMPL](https://smpl.is.tue.mpg.de) body model and the [smplx](https://github.com/vchoutas/smplx) package


All components are used under their respective licenses.
