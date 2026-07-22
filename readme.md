

# Pose Features Encoding

This repository implements the full pipeline described in
**“Simple 3D Pose Features Support Human and Machine Social Scene Understanding” (Qin & Isik, 2025)**.
It extracts interpretable **3D visuospatial pose representations** from dyadic interaction videos and encodes them to predict **human social interaction ratings**.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://drive.google.com/file/d/1jEi38gYnJTkpiH_PkgAHBWq7GqPLw46D/view?usp=sharing)

## 1. Installation

### System Requirements

Our codes relys on CPU cores and were tested and runable on Linux, Windows 11, and MacOS. However, if you would like to extract pose meshes on your own using [4D Humans](https://github.com/shubham-goel/4D-Humans) and [BEV](https://github.com/Arthur151/ROMP/blob/master/simple_romp/README.md), special system and GPU configurations for their pipelines might be required. Recommended hardware: ≥32 CPUs, ≥32 GB RAM, GPU optional.

To facilitate reproduction, we recommend checking out our [colab demo](https://drive.google.com/file/d/1jEi38gYnJTkpiH_PkgAHBWq7GqPLw46D/view?usp=sharing). You can also request any intermediate step outputs from the authors.

### Create Environment

Installation is simple and should take ~< 10 minutes on a normal desktop computer

```bash
conda env create -f environment.yml
conda activate pose
```

## 2. Directory Structure

```
Pose-Features-Encoding/
│
├── data/
│   └── raw/
│       └── dyad_videos/
│           ├── dyad_videos_3000ms_250/
│           ├── annotations.csv
│           ├── behavioral_ratings.csv
│           ├── stimulus_data.csv
│           ├── train.csv
│           ├── test.csv
│           ├── 4d-humans/
│           └── BEV/
│
├── scripts/
│   ├── 00_classify_model_families.py
│   ├── 01_frame_level_features.py
│   ├── 02_video_level_features.py
│   ├── 03_pose_model_encoding.py
│   ├── 03_dnn_encoding.py
│   ├── 04_grouped_encoding.py
│   ├── figure_2.py
│   ├── figure_3.py
│   ├── figure_4.py
│   ├── figure_5.py
│   ├── plot_responses.ipynb   # Figure 6 + Supp. Fig. 5
│   ├── supp_fig_1.py
│   ├── supp_fig_2.py
│   ├── supp_fig_3.py
│   ├── supp_fig_4.py
│   └── supp_table_1.py
│
├── logs/
│   └── parallel/
│
├── SMPL_NEUTRAL.pkl
├── environment.yml
└── README.md
```

---

## 4. Data Preparation

This pipeline is built for **dyadic interaction videos**, but it can be adapted for other two-person video datasets. 

1. **Obtain Dataset**

   * Request the videos from the [Moments in Times dataset](http://moments.csail.mit.edu).
   * Keep only the videos that are in our train and test set. Train and test video names are under data/raw/dyad_videos/train.csv and data/raw/dyad_videos/test.csv
   * Place these files under:

     ```
     data/raw/dyad_videos/dyad_videos_3000ms_250/
     ```

2. **Download SMPL Neutral Model**

   Follow [4D human installation guide](https://github.com/shubham-goel/4D-Humans?tab=readme-ov-file#installation-and-setup) to download `basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl` to the project directory. You can rename it to `SMPL_NEUTRAL.pkl` for compatibility.

Note: To facilitate reproduction, we provide the outputs from the belowed steps (4D Human and BEV annotated pose information). Please reach out to us if you would like to have these outputs.

3. **Extract Poses with 4D Humans**

   * Follow [4D Humans setup guide](https://github.com/shubham-goel/4D-Humans)
     → “Run tracking demo on videos”.
   * Save results to:

     ```
     data/raw/dyad_videos/4d-humans/
     ```

4. **Correct Depth with BEV**

   * Use [BEV](https://github.com/Arthur151/ROMP/blob/master/simple_romp/README.md) to compute 3D body and depth.
   * Save results to:

     ```
     data/raw/dyad_videos/BEV/
     ```

---

## 5. Feature Extraction

Run sequentially:

```bash
conda activate pose
```
```
python -u -m scripts.01_frame_level_features --output_videos
```
Output each frame's 3D body joints and social pose features (positions + facing directions)

```
python -u -m scripts.02_video_level_features
```
Outputs each video's averaged 3D body joints and 3D social pose features from 90 frames in jsons. Sorted so that the first person is always the left most person in the video.


## 6. Encoding

Encoding is parallelized across CPU cores.


### 6.1 Model Embedding Extraction

All model embeddings were extracted using deepjuice package (except for 4D Human) following the same procedures as Garcia et al. For embedding extraction code, please refer to https://github.com/Isik-lab/SIfMRI_modeling. We also provide the extrated embeddings, please rearch out to authors.

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

### 6.4 Model Family Classification

Before running the DNN-family figures, classify every benchmarked model into an architecture/training family (e.g. CLIP, DINO, ConvNext, ResNet/ResNext/SE) based on its model UID:

```bash
python -u -m scripts.00_classify_model_families
```

Reads `experiments/all_models_list.csv` and writes `data/processed/grouped_models.csv`, which is consumed by `figure_2.py`, `figure_4.py`, `figure_5.py`, and `supp_table_1.py`.

### 6.5 Figures and Supplemental Analyses

Each figure in the paper is produced by its own dedicated script (these replace the old, single `05_behavioral_encoding.py` pipeline). All scripts read previously saved encoding results (from `experiments/SOTA_beh` and `experiments/ridge_results`) and save plots/tables to `results/`:

```bash
python -u -m scripts.figure_2       # DNN vs. pose features, grouped by model family
python -u -m scripts.figure_3       # Pose feature encoding scores by rating target
python -u -m scripts.figure_4       # Behavioral rating score vs. pose encoding score trends
python -u -m scripts.figure_5       # DNN + 3D social pose features vs. DNN alone

python -u -m scripts.supp_fig_1     # Dataset composition summary (in-plane / baby counts)
python -u -m scripts.supp_fig_2     # 3D body joints under different temporal aggregations
python -u -m scripts.supp_fig_3     # Semipartial correlation analysis of 3D body joints
python -u -m scripts.supp_fig_4     # CV encoding across feature variants and datasets
python -u -m scripts.supp_table_1   # Per-model, per-target test scores table (CSV)
```

Figure 6 and Supplemental Figure 5 (perceptual validation of the 3D pose position/facing controls against communication ratings) are produced by `scripts/plot_responses.ipynb`:

```bash
jupyter nbconvert --to notebook --execute scripts/plot_responses.ipynb
```

This saves `responses_by_condition.png` (Figure 6) and `c1_near_comparison.png` (Supplemental Figure 5).

To test any of these on an HPC cluster, edit `SCRIPT` in `slurms/encoding.sh` to the desired script name (e.g. `figure_2`) and submit with `sbatch slurms/encoding.sh`.


## 7. Outputs

| Output Type                 | Description                                                                                                |
| --------------------------- | ---------------------------------------------------------------------------------------------------------- |
| **3D Joints**               | 45 keypoints (x, y, z) per person averaged per video                                                       |
| **3D Social Pose Features** | Head position and facing direction per agent                                                               |
| **Encoding Scores**         | Pearson correlation between predicted and actual human ratings                                             |
| **Figures**                 | `results/fig2–5.png`, `supp_fig_1–4.png` — spatial expanse, interagent distance, facing, communicative and physical interaction predictions; `responses_by_condition.png` (Fig. 6) and `c1_near_comparison.png` (Supp. Fig. 5) — perceptual validation of 3D pose position/facing controls |
| **Supplemental Table**      | `results/supp_table_1.csv` — per-model, per-target test-set Pearson r and model family                    |



## 8. Expected Runtime

| Task                             | Approx. Duration (per 250 videos) | Hardware |
| -------------------------------- | --------------------------------- | -------- |
| Pose extraction (4D Humans)      | ~2–3 h                            | GPU      |
| BEV depth correction             | ~1 h                              | GPU      |
| Feature extraction               | ~5 min                           | CPU      |
| Encoding (per model)             | ~2–3 min                          | CPU      |
| Full group encoding (351 models) | ~1 h on 48 cores                  | HPC      |


## 9. Citation

If you use this code, please cite:

```
@article{qin2024simple3dpose,
  title={Simple 3D Pose Features Support Human and Machine Social Scene Understanding},
  author={Qin, Wenshuo and Isik, Leyla},
  journal={arXiv preprint arXiv:2511.03988},
  year={2024}
}
```


## 10. License and Acknowledgements

This repository builds on:

* [4D Humans](https://github.com/shubham-goel/4D-Humans)
* [BEV (Bird’s-Eye View Estimation)](https://github.com/Arthur151/ROMP)
* [350+ Vision DNN Benchmarking](https://github.com/Isik-lab/SIfMRI_modeling)
* [DeepJuice Benchmarking Toolkit](https://github.com/neuroai-bench/DeepJuice)
* [Himalaya](https://github.com/gallantlab/himalaya)


All components are used under their respective licenses.
