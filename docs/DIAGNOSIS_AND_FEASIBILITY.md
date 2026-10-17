# Diagnosis of motion learning + feasibility of pretrained encoders (2026-10-13)
Single seed, 3000 steps, 5 internal-validation subjects (120 scans); NOT comparable to the hidden TUS-REC test set. Raw data: `runs/diag_*.json`,
`runs/feasibility.json`, `runs/pretrained_audit.json`, `runs/speckle_res.json`. Code: `diagnose.py`, `feasibility.py`, `pretrained.py`, `pairwise.py`.

## 1. Results (val, all 120 scans, mm)
| Model (all from scratch unless noted) | GPE | GLE | LPE | LLE | lag16 | lag64 |
|---|---|---|---|---|---|---|
| no motion (identity) | 90.22 | 88.25 | 0.391 | 0.388 | 6.14 | 24.41 |
| ViT pose regression: global / w4 / w8 / w1 | 90.18 / 90.22 / 90.23 / 90.20 | 88.2 | 0.391 | 0.387 | 6.14 | 24.41 |
| D4RT-style, global encoder (anchor / adjacent) | 90.21 / 94.86 | 88.27 / 91.84 | 0.391 / 0.407 | 0.388 / 0.403 | 6.14 / 6.38 | 24.41 / 25.36 |
| D4RT-style, local4 encoder (anchor / adjacent) | 90.19 / 92.71 | 88.21 / 89.48 | 0.391 / 0.398 | 0.388 / 0.394 | 6.14 / 6.25 | 24.41 / 24.84 |
| EfficientNet-B1 pairwise, random, raw input | 85.77 | 84.43 | 0.38 | 0.38 | 5.95 | 23.46 |
| EfficientNet-B1 pairwise, random, gray-ImageNet norm | 61.4 | - | 0.313 | - | 4.55 | 17.43 |
| EfficientNet-B1 pairwise, **ImageNet weights**, same norm | **26.0** | - | **0.207** | - | **2.55** | **9.04** |
Published (hidden test, other split): FlowNet GPE 5.97, FiMoNet LPE 0.097, official baseline GPE 12.49 / LPE 0.135.

## 2. Diagnosis (train and val give the same picture; `runs/diag_*.json`)
- **ViT pose models (4 variants): total collapse to identity.** Std of predicted steps / std of GT = 0.002-0.013 (translation), correlation with GT |r|<0.03
  (train and val), skill vs identity 0.000 in every elevational-distance / overlap bin. Predictions are identical with true frames, frames of another subject, or
  a static frame (they ignore the images). Gradients are tiny (embed 1e-3, layers 1.4e-3 vs weight norm 45) and have *negative* cosine across batches (-0.1): noise.
  Window size (global, 1, 4, 8) makes no difference because none of them learns anything.
- **D4RT-style (global and local4): collapse as well.**
  - Identity queries (t_src = t_ref): error 0.16-0.17 mm, i.e. the decoder adds noise (offset std 0.12 mm) to a position it is given exactly.
  - Non-identity queries: skill vs "same position" -0.003 (global), no skill at any lag bin (dt 1 ... 63; at dt=1 it is worse than identity, skill -0.09), same error
    with another clip's images, offset correlation with GT |r|<0.06, std ratio 0.015-0.5 (largest at dt=1, where it is noise). No difference inside/outside the reference FOV.
  - Gradients: the encoder receives ~1e-4 (weight norm 45) while the last decoder layer receives 2.7, with cross-batch cosine 0.95-0.99: the only consistent signal
    pushes a constant bias; nothing flows to the encoder. So it does not learn long-range geometry, nor small frame-to-frame motion.
  - Anchor-mode reconstruction is stable (Kabsch singular-value ratio >= 0.52, rigid residual 0.14 mm); "adjacent" is worse than identity because each step adds noise.
- **EfficientNet pairwise (random): learns a little** (val corr up to 0.12, std ratio of rotations 0.3-0.55, translations 0.01-0.24; worse with other/static images).
- **EfficientNet pairwise (ImageNet): learns a lot.** Same data, preprocessing, steps and loss as the random run.
Caveats: single seed; the random-vs-ImageNet curves were still improving at 3000 steps (ImageNet: 75 -> 41 -> 34 -> 32 -> 28 -> 26 GPE at steps 500..3000; random-norm
dropped late, 87.7 -> 66.1 -> 61.4 at 2000/2500/3000); intermediate curve points use every 7th val scan, final uses all 120.
Input normalisation alone (the only difference between the two random runs, same seed/split/steps) moved the pairwise net from GPE 85.8 to 61.4, so preprocessing matters.
**Hypotheses, not shown:** (a) transformers from scratch need far more steps / get no early gradient signal; (b) pretrained or convolutional features (local inductive bias) are what
makes motion learnable here. The ImageNet-vs-random CNN gap supports (b) for CNNs; it says nothing yet about ViTs.

## 3. Weights actually loaded (audit in `runs/pretrained_audit.json`; our re-implementations match the references: USFM 4e-7, VideoMAE 1e-5 relative diff at native size)
| Component | Loaded | Adapted | Checkpoint not used | Random init |
|---|---|---|---|---|
| USFM `FrameViT` (backbone of `Net2D`) | 85.66 M (186 tensors, 100% of the ViT-B) | rel-pos table 27x27 -> 13x19 (bicubic) for the 7x10 grid; gray -> RGB replicate + ImageNet norm | `mask_token`, derived index buffer (0.04 M) | none in the backbone |
| `Net2D` rest | - | - | - | proj 0.30 M, temporal transformer 7.10 M (4 layers, dim 384), pos 0.03 M; D4RT decoder 3.3 M |
| VideoMAE-B encoder | 85.44 M (160 tensors, 100% of the encoder) | RGB conv3d -> 1 channel (exact: sum_c W_c/std_c, bias absorbs means); sinusoidal pos regenerated on our flat (t,h,w) layout | decoder 8.0 M, `encoder_to_decoder`, `mask_token` | D4RT decoder 3.4 M (kv Linear 768->256) |
| EfficientNet-B1 pair (ImageNet) | 6.58 M (506 tensors incl. BN buffers; all `features`) | first conv 3->2 channels: sum_c W_c / 2 per frame (= pretrained response on the mean frame) | classifier 1.28 M | pose head 11.5 k |
Both arms of the planned USFM comparison use identical code, seed and RNG consumption; only `usfm=True` loads the backbone.

## 4. Feasibility: cost per D4RT step (encoder + decoder, Q=512, bf16, fwd+bwd+AdamW, GPU 4 of a shared node; ~equal for random vs pretrained)
| Config | kv tokens/clip | bs | enc / dec params | VRAM | s/step | h for 3000 steps at bs 8 |
|---|---|---|---|---|---|---|
| ours Net (scratch) 120x160, L64 | 4410 | 8 | 14.4 / 3.3 M | 4.9 GB | 0.128 | 0.11 |
| **USFM-MV 120x160 (7x10), L64** | 4480 | 8 | 93.1 / 3.3 M | 21.5 GB | 0.376 | 0.31 |
| USFM-MV 176x240 (11x15), L64 | 10560 | 4 | 93.1 / 3.3 M | 27.1 GB | 0.569 | 0.95 |
| USFM-MV 240x320 (15x20), L64 | 19200 | 2 | 93.2 / 3.3 M | 27.7 GB | 0.656 | 2.19 |
| VideoMAE-B native 224x224, L16, stride 2 | 1568 | 8 | 85.4 / 3.4 M | 6.2 GB | 0.107 | 0.09 |
| **VideoMAE-B 120x160, L64, stride 2** | 2240 | 8 | 85.4 / 3.4 M | 8.1 GB | 0.155 | 0.13 |
| VideoMAE-B 120x160, L64, stride 1 (one token per frame pair, as our Net) | 4410 | 8 | 85.4 / 3.4 M | 14.1 GB | 0.364 | 0.30 |
| VideoMAE-B 176x240, L64, stride 2 | 5280 | 8 | 85.4 / 3.4 M | 16.6 GB | 0.466 | 0.39 |
Everything fits one 40 GB A100 at bs 8 except USFM-MV at >=176x240 (smaller batch). A comparable pilot (3000 steps, same data/seed/eval) costs 0.1-0.4 h per arm at 120x160.
VideoMAE adaptation choices that are NOT neutral: (i) stride 1 and our 7x10 grid change the token layout the fixed sinusoid was defined on (flat index), (ii) no 90% masking at use time,
(iii) 120 rows give 7 patch rows (last 8 px dropped, as in our current Net). USFM is a per-frame 2D encoder: it gives appearance features, motion has to be computed by the temporal module.

## 5. Resolution (model-free test, `speckle_res.py`, 24 val scans, ~590 pairs per entry)
| Resolution | k | GT in-plane px/step | phase-corr corr x / y | NCC | corr(NCC, elevational dist) |
|---|---|---|---|---|---|
| 480x640 | 1 | 0.667 | 0.62 / 0.38 | 0.981 | -0.38 |
| 240x320 | 1 | 0.333 | 0.91 / 0.69 | 0.987 | -0.42 |
| 120x160 | 1 | 0.167 | 0.89 / 0.58 | 0.993 | -0.50 |
| 120x160 | 4 | 0.654 | 0.97 / 0.57 | 0.979 | -0.44 |
Area-downsampling to 120x160 does NOT remove the in-plane shift or the elevational decorrelation signal for a classical estimator (full resolution is noisier). Resolution is not shown to be the
bottleneck for the learned models; it can still matter for fine ViT patch features (a 16x16 patch at 120x160 spans ~14 mm), which is untested.

## 6. Proposed next pilots (nothing launched; no long training)
1. D4RT on `Net2D`: random vs USFM backbone (same everything), 3000 steps, ~0.3 h each; diagnose with `diagnose.py` (needs a `Net2D` branch).
2. D4RT with VideoMAE-B (stride 2, 120x160, 0.13 h) as encoder; same diagnosis.
3. Repeat the ImageNet-vs-random pairwise with 2-3 seeds and a longer budget to size the effect, since it is currently the strongest signal.
Success criterion before any long run: val skill vs identity clearly > 0 and image-swap degradation, on subjects not used for training.
