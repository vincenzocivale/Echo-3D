# US-Motion4D (Echo-3D)

Trackerless 6-DoF probe trajectory estimation for freehand 2D B-mode ultrasound (TUS-REC2024). Implemented so far: **M0** (data + geometry + tests) and
the **A0/A1** pose baselines on the frozen, already-prepared **USFM** encoder (not USF-MAE). A2/A3 (confidence, pose-support) are implemented and smoke-tested, not yet evaluated.
Pose graph, volumetric refinement and deformation (Phases 2-4) are **not implemented**. Status and decisions: `docs/EXPERIMENT_REPORT.md`; next-step hypotheses and literature: `docs/IMPROVEMENT_HYPOTHESES.md`.

Legacy D4RT-style experiments (earlier, different project) live in the repo root (`geom.py`, `model.py`, `train*.py`, `docs/LOG.md`, ...); the package below does not import them.

## Setup
```
PY=~/miniconda3/envs/echo3d/bin/python          # torch 2.5.1+cu121 (driver: CUDA 12.2), h5py, numpy, scipy, pyyaml, pytest
$PY -m pip install -e .                          # optional; tests/scripts also work through src/ on PYTHONPATH
```
Data: `/raid/DATASETS/TUS-REC2024` (50 train subjects), `/raid/DATASETS/TUS-REC2024_val` (3 public-validation subjects, never used for choices).
Encoder: `/raid/DATASETS/Echo3D_weights/usfm/USFM_latest.pth` (ViT-B/16, 100% of the weights loaded, audit in `runs/pretrained_audit.json`).
Official baseline code (for tests only): `third_party/tus-rec-challenge_baseline` (pytorch3d stubbed).

## Conventions
`T_A_from_B` maps B-coordinates to A. `T_camera_from_plane = T_camera_from_tool @ C` (`C` = `plane_to_tool`, image-mm -> tool);
`T_ref_from_plane_t = inv(T_camera_from_plane_0) @ T_camera_from_plane_t`; `Delta_ij = inv(T_ref_from_plane_i) @ T_ref_from_plane_j` (maps plane-j mm to plane-i mm).
Frames are already scan-converted (curvilinear sector): no polar conversion. Valid mask `assets/valid_mask_480x640.npy` is identical for all scans (68.4% of the image).
Pixel -> mm uses the official 1-based convention; crop/resize/pad update the calibration exactly (`geometry/calib.py:PixelCalib`).

## Commands (all verified)
```
$PY -m pytest -q                                                    # 17 tests: SE(3), calibration, official-code comparison, split, leakage
$PY -c "from usmotion4d.data import split; split.make_manifest()"   # (re)create the frozen 40/10 split (already versioned: configs/splits/split_v1.json)
$PY scripts/train_pose.py configs/pose_base.yaml --arm A1 --name pilot_A1_s17      # train an arm (A0|A1|A2|A3); W&B: set wandb.enabled
$PY scripts/train_pose.py configs/pose_base.yaml --arm A0 --steps 600 --overfit 64 --name overfit_A0   # mandatory overfit check
$PY scripts/eval_pose.py outputs/pose/pilot_A1_s17/last.pt --subset dev --every 1   # reload + evaluate a checkpoint on its own
```
Outputs: `outputs/pose/<run>/{run.json,last.pt,eval_dev.json}`; `run.json` stores config, seed, split sha256, encoder checkpoint sha256. Checkpoints keep only the trainable parts (the frozen encoder is reloaded from its file).

## Layout
`src/usmotion4d/{geometry (se3, calib), data (split, tusrec), models (encoder, correspondence, pose_support, pose_estimator, losses), evaluation (metrics, pose_eval)}`;
`optimization/`, `volume/` are empty placeholders for Phases 2-3. Tests in `tests/`, configs in `configs/`.
