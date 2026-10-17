# Echo-3D — trackerless 3D freehand ultrasound, D4RT-inspired

Goal: reconstruct 3D geometry from freehand ultrasound video without an external tracker, using a global multi-view
encoder (idea from D4RT, arXiv 2512.08924) and test whether it beats frame-to-frame pose regression and the TUS-REC2024 methods.
Status: pipeline built and tested; first pilots show **no learning beyond "no motion"** (see `LOG.md`).

## Layout (`/home/vcivale/Echo-3D`)
| File | Role | Self-check |
|---|---|---|
| `geom.py` | calibration, pixel grid, relative poses `rel`, 4 DDFs, 6D pose, query GT, uv->mm, Kabsch | `python geom.py` |
| `data.py` | subject-disjoint split, `Clips` dataset (clip of L frames or whole scan, downsampled to 120x160) | `python data.py` |
| `model.py` | `Net` (spatio-temporal ViT, `window` = local/global switch, `encode()` = reusable features, pose head), `PointQueryDecoder`, `D4RT` | `python model.py` |
| `eval.py` | GPE/GLE/LPE/LLE, lag errors, clip-boundary diagnostic, `predict` (pose arm), `predict_q` (D4RT arm), `sensitivity` | `python eval.py` (oracle) |
| `train.py` | arm A: pose regression (point loss local + lam*global, logs per-term loss and grad norms) | — |
| `train_q.py` | arm B: D4RT-style, L1 on 3D points only | — |
| `test_query.py` | tests for the query pipeline (GT, gauge, identity, oracle, independence, Kabsch, no GT leakage) | `python test_query.py` |
| `floors.py` | reference errors: identity ("no motion") and mean-step prior | `python floors.py 5 1` |
| `runs/` | pilot outputs: `pilot/` (arm A, frozen code copy), `pilot_q/` (arm B) | — |

Run everything with `~/miniconda3/envs/echo3d/bin/python` (torch 2.5.1+cu121, h5py, numpy; `requirements.txt`).
The driver supports CUDA 12.2 only: install torch with `--index-url https://download.pytorch.org/whl/cu121`.

## Data
- TUS-REC2024 (Zenodo concept DOIs 10.5281/zenodo.11178508 / 11180794 / 11355499 / 12752245; use the *version* record id in URLs).
- Local: `/raid/DATASETS/TUS-REC2024/` — 50 subjects `000..049` (24 scans each, 1200 scans, 480x640 uint8, ~0.2 mm/px,
  207-852 frames/scan), `landmark_XXX.h5`, `calib_matrix.csv`. Official validation (3 subjects): `/raid/DATASETS/TUS-REC2024_val/`.
- The official test set (32 subjects) is NOT public: published numbers are not directly comparable to ours.
  Our protocol: subject-disjoint split of the 50 train subjects (`data.split(n_val=5, seed=0)` -> 45/5), same split for all arms.
- Use terms: research only, no commercial use, cite the TUS-REC2024 challenge paper (arXiv 2506.21765).

## Conventions (verified against the official baseline to <=1e-3 mm)
- `tforms[i]` = tool->world; `calib_matrix.csv` = pixel->mm scaling `S` and image_mm->tool `C`.
- `T_{i->j} = C^-1 inv(T_j) T_i C` maps image-mm(i) to image-mm(j); computed in float64 (tracker translations ~100s mm).
- Pixels are 1-based centres (x in 1..640, y in 1..480, z=0); query `(u,v) in [0,1]` -> `x=u*640+0.5`.
- Metrics (mm): GPE/GLE vs frame 0, LPE/LLE vs previous frame (pixel / 20 SIFT landmarks). Extra: `lag{k}` = point error of `T_{i+k->i}`.
- Landmark file rows: `[frame (0-based), x, y]`.

## Models
- **A. Pose regression** (`train.py`): tubelet embedding (2 frames x 16x16 px) -> 8-layer ViT (dim 384, 14.4 M params) -> mean-pool per time step ->
  fp32 head -> 6D rotation + translation per `T_{i->i-1}`; poses are chained. `window=None` global attention, `window=w` local (|t-t'|<=w), `w=0` no context.
  Loss: point error of 5 image points (corners+centre) in mm, `local + 0.05*global`.
- **B. D4RT-style** (`train_q.py`): same encoder (same init, seed 0) + `PointQueryDecoder` (dim 256, 4 cross-attention blocks, 3.3 M params,
  no query-query attention). Query `(u,v,t_src,t_ref)` + 9x9 patch -> 3D position in the frame of `t_ref`, predicted as (known source position) + 10 mm * offset.
  Loss: L1 on 3D coordinates only. Queries: uniform over the image, `|dt|` log-uniform in [1,63], 10% identity.
  Inference: 8x8 grid per frame -> rigid Kabsch (no scale) -> `T_{f->ref}`; modes `anchor` (ref = first frame of the clip) and `adjacent` (ref = f-1); clips of 64 frames chained.

## Reproduce
```
cd ~/Echo-3D; E=~/miniconda3/envs/echo3d/bin/python
$E geom.py && $E data.py && $E model.py && $E eval.py && $E test_query.py      # all checks
$E floors.py 5 1                                                               # reference errors on the 120 val scans
CUDA_VISIBLE_DEVICES=4 $E train.py   --name global --window -1 --steps 3000    # arm A (use --window 4 for local)
CUDA_VISIBLE_DEVICES=5 $E train_q.py --name d4rt_global --window -1 --steps 3000
```
Controlled comparison: same seed => identical encoder init and identical clip sampling (first-batch checksum is printed and equal across runs;
`train_q.py` draws queries from its own generator and builds the decoder under `fork_rng` to keep this true).
