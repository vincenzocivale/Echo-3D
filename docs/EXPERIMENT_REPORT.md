# EXPERIMENT_REPORT — M0 (data + geometry) and M1 pilot (A0/A1), 2026-10-13

Single seed (17), internal development split only. Not comparable with published TUS-REC numbers (hidden test set, other split, much longer training).

## 1. Implemented
- `src/usmotion4d/geometry`: SE(3) Lie-algebra (`exp/log/inverse/compose`, Taylor series for ~1e-3 rad rotations), `PixelCalib` (pixel->mm with exact crop/resize/pad updates), `plane_poses` / `delta` / `sequential`.
- `data`: frozen subject-disjoint split 40/10 (`configs/splits/split_v1.json`, sha256 `e8c3714a…303ae2`), HDF5 loader (train + public-validation layouts, metadata from the scan name), `PairDataset` (offsets 1/2/4/8).
- `models`: `UltrasoundEncoder`/`USFMEncoder` (frozen USFM ViT-B/16 **already prepared in this repo**, 100% of the weights loaded, 0 random-init; not USF-MAE), `CorrespondenceMatcher` (Sinkhorn-style bidirectional normalisation, confidence = (1-H_norm)·mutual), `PoseSupportHead`, `PoseEstimator` (A0..A3 by flags), separate rot/trans normalised Huber loss.
- `evaluation`: official GPE/GLE/LPE/LLE, per-offset pose error, drift (start/mid/end), breakdowns by trajectory/orientation/direction, pooling-weight diagnostics, identity floor.
- Not implemented: pose graph (Phase 2), volume/renderer/refinement (Phase 3), deformation interface (Phase 4), figure/DDF export, W&B images. A2/A3 are implemented and shape-tested only; **not trained**.

## 2. Data audit findings
- Frames are 480×640 uint8, **already scan-converted** (curvilinear sector on black): no polar conversion is applied. The valid mask is identical in all 40 sampled scans (IoU 1.0 at threshold 0.05–0.5), 68.4% of the image; ≤10 edge pixels/frame are non-zero outside it (value ≤2).
- Calibration: 0.229 × 0.221 mm/pixel. Step statistics (40 scans): per-frame translation 0.40 mm (std 0.16), rotation rms 2.0 mrad; at offset 8: 3.1 mm / 10.2 mrad (mean; rms 12.9).
- Official validation (3 subjects, `TUS-REC2024_val`) has a different layout (frames/ transfs/ landmark/); loader supports it, it was not used.

## 3. Geometry tests (all pass: `pytest` 16 passed)
1 pixel→3D→pixel; 2 SE(3) composition/inverse/exp/log (vs scipy); 3 trajectory relative to frame 0, Δ composition, gauge invariance; 4 pixel→mm (official convention); 5 crop/resize/pad vs block means;
6 comparison with the **official baseline code** (`LabelTransform`, `read_calib_matrices`, pytorch3d stubbed): Δ and global/local pixel DDF agree to <5e-3 mm (the official pipeline is float32; ours float64), plus the metric is 0 for GT and exactly 1 mm for a known 1 mm/frame error. Static leakage audit: model and prediction code never reference `tforms`/poses.
Diagnostics done: pose **overfit** (64 fixed pairs, 600 steps: A0 loss 0.87→2e-4, A1 →1e-5) passed. Not yet done: identity test, inverse consistency, pose-graph/renderer/refinement tests (components don't exist), degeneracy audit of rendering.

## 4. Pilot A0 / A1 (frozen USFM, 192×256 input = 12×16 tokens, 2.1/2.4 M trainable params, 5000 steps, batch 32, 17 min each, 1.6 GB)
Dev subset: every 7th dev scan = 35 scans (all trajectories and both directions; every 10th silently selects only DtP, and every 6th the same 4 scan types — avoid), sequential composition of k=1 predictions, mm.

| | GPE | GLE | LPE | LLE | trans err k=1 / k=8 (mm) | rot err k=1 / k=8 (mrad) | corner err start / mid / end (mm) |
|---|---|---|---|---|---|---|---|
| identity ("no motion") | 85.5 | 80.9 | 0.377 | 0.381 | ~0.40 / ~3.1 | ~1.8 / ~10.2 | – |
| A0 encoder + regressor | 63.7 | 60.5 | 0.338 | 0.332 | 0.351 / 2.65 | 1.56 / 9.4 | 25 / 64 / 104 |
| A1 + CAT matcher | **56.8** | **54.8** | **0.299** | **0.293** | **0.312 / 1.96** | 1.55 / 9.2 | 22 / 58 / 92 |

By type (A1): L best (LPE 0.273, trans k8 1.5 mm), S worst (0.311 / 2.3 mm); Par GPE 43.7 vs Per 69.1; DtP 53.6 vs PtD 60.1. Inference 0.08–0.17 s/scan, peak 1.6 GB.

## 5. Findings
1. **First pilots collapsed to "no motion"** (GPE 81 = identity floor, output independent of the input; training loss identical to 4 digits for encoder layers 0/4, 4/8, 8/12 and for A0/A1). Cause: my `PairDataset` swapped (i,j) at random, so both signs of the same visual change were targets. The sign of the elevational translation is not observable from two frames (speckle decorrelation gives |d|), so the Huber optimum is 0. Inference always queries (t, t+k) forward, so training is now forward-only (`reverse_prob: 0`); the reverse direction is left to an inverse-consistency term. Same code otherwise: GPE 82→64 (A0) and 71 (A1) in 1500 steps. The flawed runs are kept in `outputs/pilot_flawed_bidirectional/`.
2. Both arms beat the identity floor on translation; **rotation is not learned** (1.55 mrad vs ~1.8 for identity at k=1; 9.2 vs ~10.2 at k=8, mean GT rotation). Drift is large (corner error already 22 mm in the first third).
3. **A1 > A0** on every translation metric (LPE 0.299 vs 0.338, trans k8 1.96 vs 2.65 mm, GPE 56.8 vs 63.7): consistent with the correspondence hypothesis, but one seed, 35 scans, GPE noisy between checkpoints (e.g. A1 GPE 55.6 → 51.6 → 56.8 on different scan subsets) — **not a result yet**.
4. **Degenerate pooling**: effective number of tokens in the attention pooling is 6.1 (A0, of 384) and 1.5 (A1, of 192), top-5 mass 0.88 / 1.00; for A1, `match_confidence` is uncorrelated with the pose error (r = −0.02). The pooled estimate rests on ~1 token; this must be fixed (temperature / entropy regulariser / weights as a convex mixture) before A2/A3 can be judged, otherwise "pose-support" has no room to matter.
5. Published GPE (FlowNet 5.97, FiMoNet LPE 0.097) are far away; legacy EfficientNet-B1/ImageNet pairwise (GPE 26 on a 5-subject split, see `docs/DIAGNOSIS_AND_FEASIBILITY.md`) is still better than anything here.

## 6. Known issues / open problems
- Rotation not learned; sequential k=1 composition drifts. Candidates (each to be tested as one ablation, not stacked): learned temperature / entropy floor in the pooling; k>1 in the loss for consistency (i→j→k); inverse-consistency; unfreezing the last USFM blocks (planned ablation, **deliberately not done** because the spec asks for a frozen encoder first); higher input resolution (384×512).
- Dev evaluation uses a 35-scan subset and a single seed; confirmatory protocol (3 seeds 17/42/97, subject-grouped CIs) not run.
- Intermediate dev numbers in the training logs use the subset available at the time (every 10th scan in the logged pilot: GPE 51.6 for A1, 53.4 for A0); the table above is the balanced every-7th evaluation of the same checkpoints.
- Disk: `/home` 99% full (8.7 GB free); outputs are small (heads only, ~10 MB).

## 7. Decision
- **M0: GO** (all geometry tests pass, comparison with official code done, split frozen).
- **M1: conditional GO.** A0 runs and is evaluated; the pipeline learns translation but not rotation and is far from competitive. **M2 (A1 vs A2 vs A3 judgement) is NO-GO** until the pooling collapse is fixed and a rotation-learning check passes; running A2/A3 now would compare degenerate pooling schemes.
- Next minimal step: fix pooling degeneracy, re-run A0/A1 with the same protocol (same seed), then A2/A3.

---
# Update 2 — pooling alternatives, audits, image-free baselines (2026-10-13/14)
Same protocol as above (seed 17, 5000 steps, batch 32, frozen USFM, 35 dev scans = every 7th, sequential k=1 composition). A2/A3, pose graph and volume were **not** started; USFM stays frozen.

## 8. Audits (scripts: `audit_gt.py`, `audit_loss.py`, `eval_sign.py`; raw data `outputs/audit_*.json`, `outputs/pose/*/eval_*.json`)
- **GT rotations** (train, plane frame x lateral / y depth / z elevational): per-frame rotation is isotropic and zero-mean, std 1.28 / 1.01 / 1.35 mrad (wx/wy/wz), mean |ω| 1.64 mrad at k=1, 10.1 mrad at k=8; it scales ~linearly in k (smooth motion, not white noise). Translation per frame: vx 0.30, vy 0.13, vz 0.29 mm (std). Mean over all scans ≈ 0 on every axis; the elevational direction is set by the acquisition direction (mean vz: DtP +0.17, PtD −0.17 mm/frame at k=1; scan-mean signs split 51/49).
- **Loss scales/gradients**: normalised errors at zero prediction rot 0.71 / trans 1.03 (78% / 61% of samples in the Huber quadratic zone); gradient norm on trainable parameters rot 0.40 / trans 0.70 (untrained), 0.55 / 0.55 (A1 pilot), cosine ≈ 0 → the two components are balanced; the normalisation is not what prevents rotation learning.
- **Rotation on a small subset** (8 train scans, train on frames [0,0.7), test on [0.7,1] of the same scans; 1500 steps; mean abs error per axis wx/wy/wz, mrad, k=1 | identity: 0.52 / 0.54 / 0.78):

  | | k=1 rot | k=8 rot (identity 3.13 / 2.83 / 4.62) | k=8 trans x (identity 2.59 mm) |
  |---|---|---|---|
  | mean pool, both losses | 0.83 / 0.64 / 0.87 | 4.79 / 3.80 / 5.21 | 0.82 |
  | mean pool, rotation loss only | 0.94 / 0.65 / 0.84 | 6.32 / 3.81 / 4.51 | (not trained) |
  | spatial 4×4, both losses | 1.03 / 0.74 / 0.89 | 6.84 / 4.60 / 5.74 | 0.64 |

  Training rotation loss goes to ~0.08–0.1 (memorised) but held-out rotation is **worse than predicting identity**, even within the same scans, with a rotation-only loss: rotation is not extractable with the current frozen features/resolution/head; lateral translation vx is (0.82 vs 2.59 mm).
- **Image-free baselines** (same 35 scans; identity GPE 85.5 / LPE 0.377 / trans k8 3.05 mm / rot k8 4.84,4.13,5.11 mrad): mean transform per k (global) = identity (GPE 85.6). **Per scan-type mean (orientation × trajectory × direction, oracle metadata not available at inference): GPE 23.7, GLE 21.4, LPE 0.214, trans k8 1.57 mm, rot k8 4.74/3.34/5.05.** It beats every learned model here on translation: the gain of the learned models over identity is compatible with learning acquisition-protocol priors (e.g. the DtP/PtD sign of vz), not necessarily motion estimation. Rotation is not predictable from the prior (≈ identity).
- **Sign / reversal** (A1-attention): vx r = 0.47–0.80, vz r = 0.12 (within direction) / 0.32 (pooled), sign accuracy vz 0.66 (k=1) / 0.76 (k=8); predictions on reversed pairs are partly antisymmetric (corr(ξ_fwd, −ξ_rev) 0.44–0.94; 0 = pure prior) and the reversed *sequence* gives GPE 61.3 (forward 56.8, identity 85.5): the model uses image evidence for in-plane motion and part of the elevational sign. Identity test (two identical frames): |ω| ≈ 5e-5 rad, |v| = 0.05 / 0.01 / 0.04 mm (≈10% of a step in x, i.e. a small residual prior). Algebraic SE(3) tests (exp(−ξ)=exp(ξ)⁻¹, Δ_ji=Δ_ij⁻¹, Δ_ijΔ_jkΔ_ki=I, BCH, one-parameter subgroup, loss=0 at GT): pass. Forward/backward consistency of pair predictions is **not** imposed.

## 9. Pooling pilots (A1, seed 17, 5000 steps, identical protocol)
| | GPE | GLE | LPE | trans k=8 (mm) | rot k=8 x/y/z (mrad) | trans k=8 x/y/z (mm) | corner err start/mid/end (mm) | eff. tokens (of 192) / top-5 mass | antisymmetry vz k=1 / reversed-seq GPE |
|---|---|---|---|---|---|---|---|---|---|
| identity | 85.5 | 80.9 | 0.377 | 3.05 | 4.84 / 4.13 / 5.11 | 1.95 / 0.56 / 1.72 | – | – | – |
| A1 attention (P0, previous) | **56.8** | **54.8** | **0.299** | **1.96** | 4.80 / 3.59 / 5.14 | 0.97 / 0.53 / 1.32 | **22 / 58 / 92** | 1.5 / 1.00 | +0.51 / 61.3 |
| A1-P1 masked mean | 76.9 | 72.9 | 0.361 | 2.85 | 4.84 / 3.99 / 5.09 | 1.72 / 0.55 / 1.71 | 28 / 78 / 127 | 130 (uniform) / 0.04 | −0.96 / 91.6 |
| A1-P2 spatial 4×4 | 67.2 | 62.6 | 0.336 | 2.51 | 4.81 / 3.86 / 5.06 | 1.46 / 0.53 / 1.54 | 26 / 68 / 109 | 105 / 0.17 | −0.77 / 98.9 |

Per-axis k=1 rotation errors are 0.83 / 0.62–0.64 / 0.85 mrad for all three variants (identity 0.83 / 0.65 / 0.86): **no variant learns rotation.** Both alternatives remove the weight collapse but are **worse** than the collapsed attention pooling on every translation and trajectory metric; the reversal diagnostics show why: with mean/spatial pooling the output for a reversed pair has (almost) the *same* sign as for the forward pair (antisymmetry −0.8…−1.0) and the reversed sequence is as bad as or worse than identity (GPE 91.6 / 98.9 vs 85.5): these heads learn a forward-motion prior from positional/aggregate features instead of reading the displacement evidence. Attention concentrated on ~1–6 tokens is the only variant that keeps the antisymmetric (image-driven) behaviour. Single seed, 35 scans: differences of a few GPE points are not significant, differences of 10–20 points with consistent LPE/trans/reversal behaviour are suggestive, not confirmatory. Q-Former was not run (no run beyond the two requested).

## 10. Decision (update)
- **Pooling fix: NO-GO as a replacement.** Mean and spatial pooling do not improve on attention pooling; keep A1-attention as the working baseline and keep P1/P2 as ablations. The weight-collapse diagnostic stays open but is not what limits performance.
- **M1: still conditional, now with sharper limits.** (i) Rotation is not learnable with the frozen USFM tokens at 192×256 + this head (subset test worse than identity; all pilots at identity level); (ii) the oracle protocol prior (GPE 23.7) outperforms all learned models, so the A0/A1 comparisons must always be read against it, and any claim of "learned motion" needs the in-plane (vx) and per-axis evidence, not GPE alone; (iii) elevational sign is only partly observable. **M2 (A2/A3) remains NO-GO.**
- **Options for your decision (nothing launched):** (a) Q-Former with 8 queries (+ collapse check) — low cost but the evidence above says aggregation is not the bottleneck; (b) address the information bottleneck instead: higher input resolution (384×512) and/or earlier-layer tokens, both within the frozen-encoder constraint; (c) the planned last-blocks fine-tuning ablation; (d) add the per-scan-type prior as an explicit reference row in every future table. My recommendation: (b) first (cheapest test of the observability hypothesis for rotation), then (c) only if (b) fails.

---
# Update 3 — frozen-encoder information tests (option b) and a CNN control (2026-10-14/16)
Same protocol (seed 17, 5000 steps, batch 32, 35 dev scans = every 7th, sequential k=1 composition). Env: `~/micromamba/envs/methyl-predictor` (torch 2.5.1+cu121; `echo3d` env no longer exists; data now under `/mnt/hdd/vcivale/DATASETS`).

## 11. Option (b): earlier layers / higher resolution (A1, frozen USFM; final DEV from the training log, no eval_dev.json)
| | GPE | LPE | trans k=8 (mm) | rot k=1 (mrad) |
|---|---|---|---|---|
| A1 layers 8/12, 192×256 (§4) | 56.8 | 0.299 | 1.96 | 1.55 |
| A1-L48 layers 4/8, 192×256 | 75.9 | 0.360 | 2.87 | 1.57 |
| A1-R384 layers 8/12, 384×512 (2 h, 9.2 GB) | 79.8 | 0.369 | 2.93 | 1.57 |
Both worse than the baseline, rotation unchanged: **(b) NO-GO.**

## 12. PairCNN control (`models/pair_cnn.py`, `configs/pose_CNN.yaml`): 3.2 M params, trained from scratch, early fusion (2 frames = 2 channels), flattened final map; 33 min, 0.9 GB. Overfit check (64 pairs): loss 1.1 → 9e-6.
| | GPE | GLE | LPE | trans k=8 (mm) | rot k=8 x/y/z (mrad) | trans k=8 x/y/z (mm) | corner err start/mid/end (mm) | reversed-seq GPE |
|---|---|---|---|---|---|---|---|---|
| identity | 85.5 | 80.9 | 0.377 | 3.05 | 4.84 / 4.13 / 5.11 | 1.95 / 0.56 / 1.72 | – | – |
| per-scan-type prior (oracle metadata) | 23.7 | 21.4 | 0.214 | 1.57 | 4.74 / 3.34 / 5.05 | – | – | – |
| A1 attention (frozen USFM) | 56.8 | 54.8 | 0.299 | 1.96 | 4.80 / 3.59 / 5.14 | 0.97 / 0.53 / 1.32 | 22 / 58 / 92 | 61.3 |
| **PairCNN (scratch)** | **23.7** | 22.1 | **0.193** | **1.35** | 4.70 / **3.19** / **3.75** | **0.67 / 0.36 / 0.91** | **11 / 26 / 37** | **24.4** |
By type: L 20.0, C 24.8, S 26.7 GPE; Par 21.7 / Per 25.6; DtP 24.4 / PtD 22.9. Dev GPE over training: 75 → 83 → 35 → 33 → 24 (steps 1k..5k), still improving.
Sign test (`eval_sign.json`, k=1 / k=8): vx r 0.93 / 0.92, vz r 0.76 / 0.81 (sign acc. 0.92), wz r 0.55 / 0.73, wx r 0.27 / 0.35; antisymmetry 0.95 (vx), 0.79–0.87 (vz), 0.73–0.86 (wz), 0.39–0.52 (wx). Identity test |v| 0.05 / 0.02 / 0.07 mm.

**Findings.** (1) The bottleneck was the frozen USFM features, not the task or the protocol: a small scratch CNN on the same data/loss/steps halves the trajectory error of every USFM arm and matches the oracle per-scan-type prior without using any metadata. (2) It is image-driven, not a prior: the reversed sequence is as good as the forward one (24.4 vs 23.7; A1: 61.3; P1/P2: >90) and predictions are antisymmetric. (3) Rotation is now partly learned: wz (about the elevational axis, in-plane rotation) clearly beats identity and prior; wy (about depth) is at prior level; wx (about lateral, out-of-plane tilt) is barely learned. (4) Single seed, 35 scans, curve not converged.

## 13. Decision (update)
- **Frozen USFM as pose encoder: NO-GO** for this pipeline (options a/b exhausted; c, fine-tuning, is no longer the cheapest next step).
- **PairCNN becomes the working baseline** and the reference row for every future table, together with identity and the per-scan-type prior.
- Next candidates (one at a time): ImageNet initialisation of the same/an EfficientNet pair network (legacy: 61 → 26 GPE); longer training (curve still falling); 3 seeds for a confirmatory number; then the correspondence/pose-support ideas (A1–A3) on top of CNN features instead of USFM tokens.

## 14. ImageNet initialisation (EfficientNet-B1 pair network, `backbone: effb1`; same head/protocol as §12; first conv RGB -> 2 frames = sum_c W_c / 2; 35 min, 2.0 GB each, run in parallel)
| | GPE | GLE | LPE | trans k=8 (mm) | rot k=8 x/y/z (mrad) | trans k=8 x/y/z (mm) | corner err start/mid/end (mm) | reversed-seq GPE | vz r / sign acc (k=8) |
|---|---|---|---|---|---|---|---|---|---|
| per-scan-type prior (oracle) | 23.7 | 21.4 | 0.214 | 1.57 | 4.74 / 3.34 / 5.05 | – | – | – | – |
| PairCNN small, scratch (§12, 3.2 M) | 23.7 | 22.1 | 0.193 | 1.35 | 4.70 / 3.19 / 3.75 | 0.67 / 0.36 / 0.91 | 11 / 26 / 37 | 24.4 | 0.81 / 0.92 |
| EffB1 scratch (7.0 M) | 30.0 | 28.3 | 0.214 | 1.44 | 4.87 / 3.23 / 4.14 | 0.67 / 0.42 / 1.00 | 14 / 32 / 46 | 35.5 | 0.77 / 0.88 |
| **EffB1 ImageNet** (7.0 M) | **21.9** | **19.8** | **0.187** | **1.21** | **4.64 / 3.06 / 3.49** | **0.61 / 0.34 / 0.81** | **10 / 23 / 35** | 27.7 | **0.87 / 0.96** |
Dev GPE at steps 1k..5k: ImageNet 36.3 / 29.2 / 26.4 / 22.2 / 21.9; scratch 48.9 / 35.4 / 31.3 / 30.0 / 30.0 (both nearly flat at the end).

**Findings.** (1) ImageNet init helps on the same architecture: GPE 30.0 → 21.9, better on every axis, faster convergence. (2) Architecture matters as much: the larger EffB1 from scratch is worse than the small PairCNN from scratch (30.0 vs 23.7). (3) EffB1-ImageNet is the best arm so far and the first learned model under the oracle prior on GPE/GLE, but only ~2 GPE ahead of PairCNN (single seed, 35 scans: not significant); its reversed sequence is somewhat worse (27.7 vs 21.9 forward; PairCNN 24.4 vs 23.7), so slightly more acquisition-prior use. (4) Rotation: wz and wy improve modestly; wx (out-of-plane tilt about the lateral axis) is still essentially not learned in any arm.
Decision: low-level (ImageNet CNN) initialisation is useful but not decisive; next step needs 3 seeds of PairCNN vs EffB1-ImageNet before choosing, and wx remains the open problem.

## 15. Resolution test for wx (EffB1-ImageNet at 384×512, only change vs §14; 42 min, 7.6 GB)
| | GPE | GLE | LPE | trans k=8 (mm) | rot k=8 x/y/z (mrad) | trans k=8 x/y/z (mm) | corner err start/mid/end (mm) |
|---|---|---|---|---|---|---|---|
| identity | 85.5 | 80.9 | 0.377 | 3.05 | 4.84 / 4.13 / 5.11 | 1.95 / 0.56 / 1.72 | – |
| EffB1-ImageNet 192×256 (§14) | 21.9 | 19.8 | 0.187 | 1.21 | 4.64 / 3.06 / 3.49 | 0.61 / 0.34 / 0.81 | 9.9 / 23.0 / 35.2 |
| EffB1-ImageNet 384×512 | 21.4 | 20.0 | 0.177 | 1.21 | 4.73 / 3.10 / 3.60 | 0.58 / 0.34 / 0.83 | 9.6 / 22.9 / 34.1 |
Dev GPE at steps 1k..5k: 34.1 / 24.5 / 26.9 / 21.5 / 21.4. 4× the pixels give a small LPE gain and **no rotation gain** (wx 4.73 vs 4.64, identity 4.84): wx is not limited by input resolution. Remaining hypothesis: tilt about the lateral axis is not separable from elevational translation with two frames (both decorrelate speckle); it needs multi-frame / sequence constraints. Keep 192×256 (4× cheaper).

## 16. Multi-offset pose-graph fusion at inference (option A; no training) — `optimization/pose_graph.py`, `scripts/eval_posegraph.py`
EffB1-ImageNet 192×256 checkpoint (§14). Variables = adjacent steps; residuals = vee(Z_{t,t+k}^-1 · step_t…step_{t+k-1}) for every predicted offset, whitened by per-axis error std per k estimated on 24 **train** scans (every 40th; never dev). L-BFGS, float64, CPU (the shared GPU was ~100× slower for this tiny problem). Sanity: exact measurements → GT (4e-6); noisy k=1 + exact k=8 halves drift.
Train error std (rot x/y/z mrad, trans x/y/z mm): k=1 1.29/0.75/0.93, 0.11/0.09/0.16; k=8 8.0/4.1/4.1, 0.79/0.39/1.01 (≈ linear in k: k=8 direct estimates are ~8× noisier, not √8 — i.e. errors are mostly a per-step bias, not white noise).
Per-axis k=8 errors here are of the *reconstructed trajectory* (relative pose of frames t, t+8 after composition/fusion), so the seq_k1 row differs slightly from §14 (direct k=8 network output).
| | GPE | GLE | LPE | rot k=8 x/y/z (mrad) | trans k=8 x/y/z (mm) | corner err start/mid/end (mm) |
|---|---|---|---|---|---|---|
| sequential k=1 (§14) | 21.9 | 19.8 | 0.187 | 4.54 / 3.10 / 3.71 | 0.49 / 0.35 / 0.82 | 9.9 / 23.0 / 35.2 |
| **pose graph {1,8}** | **16.6** | **14.9** | **0.180** | 4.54 / 3.01 / 3.31 | 0.47 / 0.31 / 0.72 | **7.9 / 18.5 / 26.1** |
| pose graph {1,2,4,8} | 16.8 | 15.0 | 0.181 | 4.52 / 3.00 / 3.29 | 0.48 / 0.31 / 0.71 | 7.8 / 18.7 / 26.6 |
**Findings.** Free −24% GPE / −26% end drift, now well below the oracle prior (23.7). Adding k=2,4 gives nothing over {1,8} (their errors are correlated with k=1/k=8). Gains come from wz, wy, vz; **wx unchanged** (4.54 → 4.52): no measurement in the graph carries wx information, so fusion cannot create it. Confirms wx needs a model that sees more temporal context (option B), not only consistency.

## 17. Option B, cold start (`models/seq_cnn.py`, `scripts/train_seq.py`): EffB1-ImageNet pair features of adjacent pairs + 2-layer temporal transformer over windows of L=16; loss on adjacent and in-window composed steps (k=1,2,4,8); inference with overlapping windows (stride 8), steps averaged. 5000 steps × 4 windows (60 pairs), 29 min, 3.6 GB each. Control: identical without the transformer.
| | GPE | GLE | LPE | rot k=8 x/y/z (mrad) | trans k=8 z (mm) | corner err start/mid/end (mm) | reversed-seq GPE |
|---|---|---|---|---|---|---|---|
| pair EffB1-ImageNet (§14/16, sequential) | 21.9 | 19.8 | 0.187 | 4.54 / 3.10 / 3.71 | 0.82 | 9.9 / 23.0 / 35.2 | 27.7 |
| SEQ, temporal | 34.2 | 33.3 | 0.223 | 4.77 / 3.36 / 4.91 | 1.24 | 15.3 / 38.3 / 52.0 | 41.5 |
| SEQ, no temporal (control) | 33.0 | 32.6 | 0.229 | 4.76 / 3.36 / 4.75 | 1.25 | 15.0 / 35.9 / 51.1 | 40.3 |
Dev GPE 1k..5k: temporal 84.6 / 69.8 / 55.4 / 36.7 / 34.2; control 78.0 / 73.4 / 49.6 / 34.5 / 33.0 (both flat at identity for ~2k steps, then late take-off). **No benefit from temporal context; both worse than the pair model.** The handicap is the setup, not the transformer: only adjacent (k=1, ~0.4 mm) pairs as input vs k up to 8 for the pair model, and 4 scans per batch. Next: warm start the pair CNN from the trained pair checkpoint so the transformer is judged on top of working features.

## 18. Option B, warm start (pair CNN trunk + feature layer from §14 checkpoint; lr 1e-4; otherwise as §17). Configs `seq_effb1{,_notemporal}_warm.yaml`.
| | GPE | GLE | LPE | rot k=8 x/y/z (mrad) | trans k=8 x/y/z (mm) | corner err start/mid/end (mm) | reversed-seq GPE (ratio) |
|---|---|---|---|---|---|---|---|
| identity | 85.5 | 80.9 | 0.377 | 4.84 / 4.13 / 5.11 | 1.95 / 0.56 / 1.72 | – | – |
| pair EffB1-ImageNet, sequential (§14) | 21.9 | 19.8 | 0.187 | 4.54 / 3.10 / 3.71 | 0.49 / 0.35 / 0.82 | 9.9 / 23.0 / 35.2 | 27.7 (1.27) |
| pair + pose graph {1,8} (§16) | 16.6 | 14.9 | 0.180 | 4.54 / 3.01 / 3.31 | 0.47 / 0.31 / 0.72 | 7.9 / 18.5 / 26.1 | – |
| SEQ warm, no temporal (control) | 15.9 | 14.7 | 0.172 | 4.56 / 3.01 / 3.42 | 0.31 / 0.31 / 0.68 | 7.3 / 18.0 / 25.3 | 22.0 (1.38) |
| **SEQ warm, temporal** | **12.9** | **11.1** | **0.152** | 4.58 / **2.94 / 3.12** | 0.31 / **0.29 / 0.59** | **6.3 / 14.7 / 20.5** | 16.7 (1.30) |
Dev GPE 1k..5k: temporal 18.0 / 17.2 / 14.8 / 13.2 / 12.9 (monotone); control 20.6 / 17.1 / 16.0 / 18.6 / 15.9. Temporal by type: C 13.0, L 11.2, S 14.5; Par 14.4 / Per 11.4; DtP 11.9 / PtD 13.9 (control PtD 19.6).
**Findings.** (1) With working features, temporal context helps: −3 GPE vs the identical no-temporal control, better on every axis except wx, steadier curve; −41% GPE and −42% end drift vs the starting pair model; GLE 11.1 below the official-baseline reference (hidden test, not comparable). (2) Fine-tuning on windows with the composed-step loss alone (control) already gives 21.9 → 15.9. (3) Reversed sequence 16.7 vs 12.9 forward: same prior reliance ratio as the pair model (1.30 vs 1.27), lower than the control (1.38): the gain is not a direction prior. (4) wx still at ~identity in every arm. Caveats: single seed, 35 dev scans, dev was used to choose among ~15 configurations → must be confirmed on the untouched public validation set (TUS-REC2024_val).

## 19. Hybrid pose graph and first public-validation result (`scripts/eval_final.py`; raw: `outputs/final_{dev,val}.json`, weights `outputs/final_sigma.json` from train scans)
Hybrid = sequence-model adjacent steps + pair-model direct k=8 estimates in the pose graph. **Chosen on dev before touching val:** final candidate = SEQ warm temporal alone (hybrid was worse on dev: 14.7 vs 12.9). Val (TUS-REC2024_val, subjects 050–052, all 72 scans) run once with that choice fixed; all rows are reported for context.
| | dev GPE | dev LPE | **val GPE** | **val GLE** | **val LPE** | val LLE | val rot k=8 x/y/z (mrad) | val trans z k=8 (mm) | val corner err end (mm) | val GPE by subject 050/051/052 |
|---|---|---|---|---|---|---|---|---|---|---|
| identity | 85.5 | 0.377 | 81.8 | 82.6 | 0.360 | 0.357 | 2.81 / 3.34 / 3.32 | 1.65 | 135.5 | 75.8 / 77.9 / 91.7 |
| pair, sequential | 21.9 | 0.187 | 15.8 | 15.2 | 0.159 | 0.149 | 2.72 / 2.13 / 2.58 | 0.64 | 24.6 | 12.9 / 14.5 / 19.8 |
| pair + pose graph {1,8} | 16.6 | 0.180 | 12.2 | 11.7 | 0.157 | 0.144 | 2.77 / 2.08 / 2.53 | 0.54 | 19.7 | 10.3 / 11.3 / 15.0 |
| **SEQ warm temporal (final)** | **12.9** | **0.152** | **9.7** | **9.1** | **0.134** | **0.122** | 2.81 / **2.04 / 2.45** | **0.47** | **15.3** | **9.1 / 8.8 / 11.2** |
| SEQ + pair k=8 pose graph | 14.7 | 0.160 | 11.4 | 11.0 | 0.140 | 0.127 | 2.79 / 2.04 / 2.49 | 0.48 | 18.4 | 10.5 / 10.7 / 13.0 |
**Findings.** Same ranking on val as on dev (no sign of dev overfitting in the ordering); val is easier than dev (smaller rotations: identity rot k=8 2.8–3.3 vs 4.1–5.1 mrad). Final model on untouched val: GPE 9.7 / GLE 9.1 / LPE 0.134 / LLE 0.122 — for orientation only, the official baseline reports 12.49 / 0.135 (GPE/LPE) on the hidden test set (different data, not comparable). wx is at identity level on val too (2.81 vs 2.81): the open problem. Hybrid fusion hurts because the pair k=8 estimates are noisier than the sequence model's own composed steps; weights assume independent errors, which they are not.
Next: retrain the official baseline on our split for a like-for-like reference; then wx.

## 20. Hypothesis 1 test: temporal context longer than L=16 on frozen features (2026-10-17; dev only, `TUS-REC2024_val` not touched)
Code: `scripts/cache_feats.py` (256-d per-pair features of the SEQ-warm checkpoint `outputs/pose/SEQ-effb1-warm_s17/last.pt`, eval-mode BN, fp16, all 960 train + 240 dev scans, `outputs/feat_cache/`), `scripts/train_longctx.py` (temporal head on cached features, same loss as `train_seq.py`, 4000 steps × 16 windows, lr 3e-4, ~2.5 min/run), `scripts/compare_longctx.py` (paired per-scan differences, subject bootstrap). Drivers: `outputs/run_{longctx,arch,seeds}.sh`; raw: `outputs/longctx/<run>/eval_dev.json`, logs `outputs/longctx_*.log`.
**Sanity:** the checkpoint's own head on cached features gives dev(every 7th) GPE 12.895 / GLE 11.170 / LPE 0.152 / LLE 0.138 vs 12.865 / 11.148 / 0.152 / 0.138 from the original eval (fp16 cache). **All numbers below are on all 240 dev scans** (not comparable with the 35-scan rows of §18/19); only differences vs the L16 control (same features, head and seed) are meaningful. Features are frozen and identical for every L; train features come from a checkpoint trained on those same scans (possible optimism, not tested).
| head | L | GPE s17 / s42 / s97 | mean ΔGPE vs L16 | LPE (mean) | mean ΔLPE |
|---|---|---|---|---|---|
| tf, learned pos (current) | 16 (control) | 11.50 / 11.42 / 11.47 | – | 0.1548 | – |
| tf, learned pos | 64 | 10.88 / 10.98 / 11.07 | −0.49 | 0.1563 | +0.002 |
| tf, learned pos | 256 | 10.79 / 10.88 / 10.88 | −0.61 | 0.1573 | +0.003 |
| tf, sinusoidal pos | 64 | 10.92 / 10.92 / 10.92 | −0.54 | 0.1414 | −0.013 |
| tf, sinusoidal pos | 256 | 10.65 / 10.80 / 10.93 | −0.67 | 0.1519 | −0.003 |
| bi-GRU (2 layers, no pos) | 64 | 11.00 / 11.04 / 11.02 | −0.44 | 0.1371 | −0.018 |
| bi-GRU | 256 | 10.89 / 10.97 / 10.70 | −0.60 | 0.1348 | −0.020 |
Seed 17 only: loss offsets up to k=32 (L64) / k=64 (L256) give GPE 10.92 / 10.84, i.e. no gain over k≤8 (10.88 / 10.79). Seed-17 subject bootstrap (10 dev subjects, 95% CI) of ΔGPE vs L16: tf L64 −0.62 [−0.98, −0.30], tf L256 −0.70 [−1.33, −0.19], tf_sin L256 −0.85 [−1.37, −0.42], gru L256 −0.60 [−1.11, −0.19]; 7-10 of 10 subjects improve. The bootstrap was not repeated for seeds 42/97. Reversed-sequence GPE was **not** measured (reversed features are not cached).
**Findings.** (1) Longer context helps GPE by a consistent but small ~0.5-0.7 mm (5-6%) at all three seeds; L64 → L256 adds ≤0.15 mm; extra loss horizons add nothing. This is far below the gains reported in the literature (§4 of IMPROVEMENT_HYPOTHESES) — on frozen adjacent-pair features the long-context head is saturated. It does not exclude a larger gain with end-to-end training. (2) The three head architectures are indistinguishable on GPE (differences ≤0.2, below seed spread). (3) LPE does depend on the head: fixed sinusoidal positions or a GRU lower LPE by 0.003-0.020 (GRU L256: 0.1348, −13%) while learned absolute positions slightly raise it; LPE ranking is stable across seeds, but the effect at L=16 was seen on one seed only. Open: tissue/feature distribution shift between train and dev features (fold-wise features), reversed-sequence check, end-to-end training (H2/H3).
