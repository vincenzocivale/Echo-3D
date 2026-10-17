# IMPROVEMENT_HYPOTHESES — what limits the current model and what to try next (2026-10-17)

Inputs: `docs/EXPERIMENT_REPORT.md` (§1-19), new dev-only analyses (below), a code audit, and a literature sweep with an independent fact-check pass
(papers marked ✔ were confirmed by the checker, ✱ corrected, ? not independently checked). **No training was launched and `TUS-REC2024_val` was not touched.**
Analysis scripts live in the session scratchpad (not versioned); all numbers needed to reason are copied here.

## 0. TL;DR (ranked)
| # | Hypothesis | Main evidence | Cheapest decisive experiment | Cost |
|---|---|---|---|---|
| 1 | **Long temporal context** (L=16 → 64-256 or whole scan) | Error is low-frequency (§2.1: keeping only the 31-frame moving average of the step error reproduces all of GPE); error correlation time > window; Li et al. TBME 2024: no gain below ~25 frames, 22.8 → 4.0 mm at M=100; DualTrack whole-sweep transformer GPE 4.93 on *our val scans* | Cache the 256-d pair features of the current warm checkpoint for every train scan, train only a temporal transformer over 64-256 steps (k up to 32/64 in the loss); compare with the same head at L=16 | < 1 h, tiny code**Tested 2026-10-17 (EXPERIMENT_REPORT §20): on frozen features GPE −0.5…−0.7 mm (5-6%), saturated at L=64; not the large gain expected. End-to-end untested.** |
| 2 | **Train 4-10× longer, more windows per batch** | SEQ warm curve still falling (18.0/17.2/14.8/13.2/12.9), train ≈ dev (no overfitting), 0.6 epoch of pairs; top teams train 1.2-9.7 GPU-days | 20k-50k steps, 16-24 windows/step, frames pre-resized to 192×256 uint8 (input-bound otherwise) | 2-8 h |
| 3 | **Loss on signal-dominated horizons** | k=1 labels are 31-58% tracker jitter (std ratio, §2.3); GPE driven by drift; inertial odometry supervises integrated displacement | Down-weight/drop k=1 rotation, add k=L-1 (and 16/32 with longer L); optional metric-aligned point loss (corners / pixel grid in mm, as GPE) | code-only, same run as 1/2 |
| 4 | **Label-exact augmentation** | Every top TUS-REC entry augments, we do not | Lateral mirror (exact 4×4 conjugation, not a sign flip), temporal stride s∈{1,2,3} (labels compose exactly), window-consistent gain/TGC jitter; reversal **only** with long context | code-only |
| 5 | **Ensembles / snapshot averaging** | RecuVol 8-model median, FlowNet 3 snapshots + offsets, FiMoNet 2 models | 3 seeds (needed anyway for CIs), average steps | 3× training |
| 6 | **Video encoder end-to-end** (port legacy VideoMAE) | Legacy VideoMAE-B, 64-frame clips: GPE/identity 0.172 vs legacy pair 0.288; new SEQ 0.151 (different split → suggestive only) | Port plan in §3.3, train in the SEQ protocol at L=16 then 64 | 1 day code+GPU |
| 7 | **Sparse global stream** (DualTrack) | local-only 7.36 → dual 4.93 GPE in their ablation | After 1: add every-8th-frame tokens of the whole scan as keys/values | medium |
| – | **Stop spending effort on wx** | oracle wx = −0.1…−0.3 GPE; classical probe: depth slope carries no wx information (§2.2) | none | – |

What matters for GPE is the **elevational translation of the image centre (vz_c) and its slow drift**, plus a per-scan bias spread over all axes — not wx, not per-step precision, not test-time tricks.

## 1. Where we are (reference rows)
dev = 35 scans (every 7th of the 10 dev subjects); val = TUS-REC2024_val, 72 scans, touched once.
| | dev GPE | dev LPE | val GPE / GLE / LPE / LLE |
|---|---|---|---|
| identity | 85.5 | 0.377 | 81.8 / 82.6 / 0.360 / 0.357 |
| per-scan-type prior (oracle metadata) | 23.7 | 0.214 | – |
| pair EffB1-ImageNet, sequential | 21.9 | 0.187 | 15.8 / 15.2 / 0.159 / 0.149 |
| pair + pose graph {1,8} | 16.6 | 0.180 | 12.2 / 11.7 / 0.157 / 0.144 |
| **SEQ warm temporal (current best)** | **12.9** | **0.152** | **9.7 / 9.1 / 0.134 / 0.122** |

External positioning:
- **Same 72 val scans**: DualTrack (arXiv 2509.09530 ✔) GPE 4.93 / LPE 0.122; their re-trained baselines: Hybrid Transformer 6.01 / 0.122, 2-frame CNN 8.90 / 0.140, MoNet 9.19 / 0.128, DCL-Net 10.77 / 0.134. Caveat: their GPE uses 5 points (corners + centre), not all pixels; ~10 A40-days of training. **We sit between DCL-Net and the 2-frame CNN of their table, with ~100× less training.**
- Hidden test TUS-REC2024 (not comparable): FiMoNet 7.19/6.28/0.097/0.084, RecuVol 6.86/5.98/0.101/0.088, FlowNet 5.97/5.17/0.111/0.096, MoGLo-Net 9.39/0.112, baseline 12.49/0.135 (arXiv 2506.21765 ✔).
- 2025-style score (identity-normalised, reproducible on our data: X* = 1 − X/X_identity, mean of the four): ~0.76 on our val (rough, from means). Worth adding to `eval_final.py`.

## 2. New analyses (dev only, SEQ warm checkpoint; sanity: cached predictions reproduce eval_dev.json, GPE 12.865)
### 2.1 Error budget (oracle substitution of steps)
| variant | GPE | LPE | corner end |
|---|---|---|---|
| baseline | 12.87 | 0.152 | 20.5 |
| oracle wx / wy / wz (corner frame) | 12.73 / 12.44 / 12.58 | | |
| oracle vx / vy / vz (corner frame) | 11.44 / 12.65 / 10.21 | | |
| oracle all rotations / all translations | 12.05 / 7.98 | | |
| centred frame: oracle vz_c (elevational motion of image centre) | **8.63** | 0.104 | 15.2 |
| centred frame: oracle wx (vz re-centred) / wy | 12.55 / 11.88 | | |
| centred frame: all centre translations | 5.33 | 0.051 | |
| per-scan constant bias removed, all axes | **6.69** | 0.148 | 7.7 |
| only high-frequency error kept (error − MA31) | **0.36** | 0.127 | 0.4 |
| only low-frequency error kept (MA31) | 12.85 | 0.078 | 20.5 |
- GPE is entirely low-frequency drift; LPE is mostly high-frequency error. ~half of GPE is a constant per-scan bias spread over several axes, the other half slowly varying error.
- Correcting only the 33-frame moving average of the centroid-dz error gives 8.64 (≈ the full vz_c oracle). Oracle sign alone 11.13, oracle magnitude alone 11.71: both matter.
- Single-axis oracles in the corner frame look small because errors cancel across coupled axes (corr(err wz, err vy) = −0.90, the origin is the image corner). **Re-parameterising the output about the image/sector centre** decouples rotation and translation and is a cheap change worth testing with hypothesis 3.
- vz oracle helps PtD much more than DtP (+4.1 vs +1.3 GPE): the elevational sign/magnitude prior is direction-dependent.

Per-axis step statistics (pooled, mrad|mm/frame): r(pred, GT) wx 0.32, wy 0.47, wz 0.65, vx 0.96, vy 0.72, vz 0.88; std ratio pred/GT wx 0.36 (strong shrinkage), wy 0.44, wz 0.64, vx 1.01. Per-scan mean of wx not learned (r of per-scan means 0.44); error ACF lag 8 up to 0.37 (wx, vz).

Test-time variants (no training): reversal TTA hurts (13.50); window stride 1/2/4, medians: ±0.05; moving-average smoothing of steps (MA3/MA5): GPE unchanged, **LPE 0.152 → 0.147, LLE 0.138 → 0.131** (the only free gain, local only).

### 2.2 Why wx is (and stays) hard — physics
- Geometry (plane frame, origin = top-left pixel): per-pixel elevational step dz(x,y) = vz + wx·y − wy·x. Measured on the real sector, the wx part of dz is 0.026 mm RMS at k=1 vs 0.277 mm for dz at the centroid (~9% of the signal, <1% of variance); informative depth only ~60 mm (intensity falls below ~66 mm).
- My own check (48 train scans): adjacent-frame NCC per depth band drops 0.96 → 0.79 from 13 to 93 mm, and a lag-1/lag-2 fit attributes the drop to a **lag-independent (SNR) factor** (0.97 → 0.81), not to motion. A second agent's classical probe (patch NCC after oracle in-plane alignment, per-depth calibrated curves) confirms: **the depth slope of decorrelation has r ≈ 0.00-0.03 with wx**, it is dominated by the translation magnitude; |dz_c| is recoverable (r 0.47 → 0.70 for k = 1 → 8).
- Literature agrees: Housden et al. (Cambridge TR587 / Ultrasonics 2008) report "considerably more error in tilt than yaw"; depth-dependent elevational beam width, attenuation/TGC and shadows make decorrelation per mm depth-dependent, so a calibration error turns vz into spurious wx.
- Symmetry: with a symmetric elevational beam the image sequence is invariant to reflection through the image plane, so (vz, wx, wy) share one sign ambiguity per sequence; the absolute sign comes from anatomy/protocol priors (consistent with reversed-sequence GPE 16.7 vs 12.9).
- Decision: wx is worth ≤0.3 GPE here. Effort goes to vz_c drift.

### 2.3 GT noise floor (train, 80 scans, step autocovariance)
Tracker pose jitter per frame: wx 0.45, wy 0.34, wz 0.37 mrad; vx 0.044, vy 0.035, vz 0.049 mm, i.e. noise/step-std 0.39 / 0.58 / 0.47 / 0.39 / 0.51 / 0.31 (10-34% of k=1 variance). k=1 rotation labels are noisy; k≥4 are signal-dominated. LPE/LLE have an irreducible floor from this; per-step r of rotations is attenuated by it.

## 3. Code / recipe audit (main points; none is a correctness bug)
1. **Under-training** (high): 5000 steps × 4 windows ≈ 0.6 epoch; train ≈ dev, curve still falling (the last steps only flatten because the cosine LR hits 0). Cold-start SEQ sat at identity ~2k steps → 4-scan k=1-only batches are inefficient.
2. **Only 4 windows (4 scans) per step** (high/medium) on a mostly idle GPU; BatchNorm sees 60 highly correlated pairs and is not frozen at warm start; EffB1 stochastic depth (up to 0.19) stays on while under-fitting → ablate frozen BN + no stochastic depth, or PreciseBN recompute.
3. **Input-bound loader**: full 480×640 frames read from HDD each step → pre-resize every scan once to 192×256 uint8 (~24 GB, memmap).
4. **Context 16 frames (~6 mm travel)**, CNN sees only adjacent pairs; loss stops at k=8 and weights noisy k=1 equally (`scripts/train_seq.py:86-90`).
5. Dev evaluation of 35 scans cannot resolve 1-2 mm (per-scan GPE SD ≈ 3.5 in the challenge) → use all 240 dev scans (inference is cheap) + subject bootstrap CIs + 3 seeds for finalists.
6. Minor: bf16 autocast around the head and `se3.exp` (`seq_cnn.py:27`) — run them in fp32; warm-start `load_state_dict(strict=False)` only printed, add an assert (`train_seq.py:20`); scan sampling ignores length (`tusrec.py:96`); crop to the fan bounding box instead of resizing the full frame; edge windows/averaging checked, fine.

### 3.3 Legacy VideoMAE (not apples-to-apples, but promising)
- Metric identical; split different (legacy val 001/014/028/036/043 — 9 of our 10 dev subjects were in its training set: **never score legacy checkpoints on our dev**), 120 vs 35 scans. Normalised by identity GPE: legacy VideoMAE 0.172, legacy pair 0.288, new pair 0.256, new SEQ 0.151.
- Port plan: new `src/usmotion4d/models/videomae.py` (copy `sincos, VBlockMAE, VideoMAEEnc, load_videomae` from `pretrained.py:133-188`), `seq_videomae.py` with the SeqPairCNN contract (temporal stride 1 → token t = step t, mean over patches → LayerNorm + zero-init Linear(768,6) × scale), separate lr for encoder/head (legacy: 2e-5 / 1e-4). Weights: `/mnt/hdd/vcivale/DATASETS/Echo3D_weights/videomae-base/model.safetensors` (verified: 85.44 M params, 0 random init at grid 12×16). Legacy default `/raid/...` no longer exists.

## 4. Literature notes (what transfers)
- **TUS-REC2024 summary** (Li et al., arXiv 2506.21765 ✔): 85 subjects, 20 fps, Ultrasonix 4DC7-3/40 curvilinear, NDI Polaris; top 3 all use ImageNet init, 10 frames-to-whole-scan context, augmentation (flips, frame-order flip, random intervals, TrivialAugment) and ensembles/TTA; organisers: "most approaches struggle to maintain lower global error over long sequences, likely due to lack of long range correlation"; S-shape hardest, Par worse than Per on local metrics, long PtD scans worse on GPE (same ordering as ours). **RecuVol** (rank 2) ≈ our design: ImageNet EfficientNet on concatenated pairs + LSTM over 16 frames, + TrivialAugment, near-native resolution, 8-model median. **FlowNet** (rank 3): EffB6, 10 frames, points + params + warped-image loss, frame-order flip augmentation, 3-checkpoint + forward/reverse + 5-offset TTA. **FiMoNet** (rank 1): ResNet18 + Mamba on whole scans (train clips 60-180 frames), L1 + Pearson loss, 2-model ensemble. Code of FiMoNet/RecuVol not released.
- **TUS-REC2025** (zenodo 15119085 ?): new "rotating" protocol (fanning/rocking around a fixed contact point), 2 scans/subject, identity-normalised score; winner ImFusion/DualTrack 0.622 (GPE 9.25). Rotation-heavy data: wx would matter there, unlike here.
- **DualTrack** (Wilson et al., arXiv 2509.09530 ✔): local encoder (video ResNet18, R(2+1)D stem, 16-frame pretraining) + global encoder (2D ResNet18 on every ~8th frame) fused by a transformer decoder over the full sweep; flip-with-pose-mirror, reverse sweep, 10% frame drop, random crop; USFM/MedSAM/iBOT backbones were worse than CNN (consistent with our USFM failure). Ablation: global only 11.26, local only 7.36, coupled 5.92, dual 4.93. Code available — the only complete top-ranked trainable codebase on this data.
- **Li et al., TBME 2024** (arXiv 2310.10248 ✔): accumulated error 22.75 (M=2) → 16.6 (20) → 11.1 (75) → 4.0 mm (100 frames); CNN ≈ RNN; ≥20k epochs. Caveat from the checker: much of the gain is attributed to protocol dependency → long-context gains must be checked with the reversed-sequence test.
- **Li et al., ISBI 2023** (arXiv 2211.04867 ✔): auxiliary past/future transformations help; intervals <10 frames best at that model scale. **Privileged anatomical/protocol discrimination** (arXiv 2308.10293 ?): subject/protocol discrimination as auxiliary training signal helps.
- **MoGLo-Net** (arXiv 2502.03505 ✱, preprint): correlation volume + global-local attention + ConvGRU; the gain is from the correlation *operation* (not a loss). **UltrasODM** (arXiv 2512.07756 ✱, workshop): optical-flow stream + Mamba, ~8% on TUS-REC, weak evidence (own split).
- **Classical out-of-plane**: Housden/Gee/Prager/Treece (UMB 2007, TR553/587 ✔), Laporte & Arbel MedIA 2011 ✔ (learned correlation length from in-plane statistics), Afsham et al. TMI 2014 ✔ (37% vs 15.5% underestimation at 1 mm). Systematic tissue-dependent elevational scale bias (~25-37% underestimation) is a classical finding — matches our per-scan bias. Idea: tissue-adaptive elevational scale from in-plane speckle statistics as an extra input/head.
- **Learned odometry / VO**: Kendall & Cipolla 2017 ✔ (learned uncertainty weighting, reprojection-style losses), β-NLL (arXiv 2203.09168 ✔) if heteroscedastic heads are revived, TLIO/RoNIN (supervise integrated displacement over ~1 s, learned covariance for fusion), DROID-SLAM/DPVO (iterative refinement + BA; deprioritised: no perspective geometry for the out-of-plane part), CL-VO curriculum on sequence length.
- **Encoders**: ultrasound foundation models (USFM, UltraFedFM, EchoPrime, EchoJEPA…) are semantic, no motion/pose evidence; EchoJEPA explicitly becomes invariant to speckle — wrong inductive bias for decorrelation. Video MAE (VideoMAE / V2 distilled ViT-B) is the plausible pretrained route (legacy evidence). Point trackers (CoTracker3 on echo speckle) only help in-plane, which is already good.

## 5. Experiment protocol going forward
- Reference rows in every table: identity, per-scan-type prior, current SEQ warm (12.9 dev).
- Choices on **all 240 dev scans**, paired per-scan differences with subject-bootstrap CI; 3 seeds (17/42/97) for any finalist; val once per finalist.
- Always report the reversed-sequence GPE and its ratio (long context could learn protocol priors, cf. Li 2024).
- Report per-axis in the centred frame (vz_c) and the low-/high-frequency split from §2.1, plus the 2025 identity-normalised score.
- Suggested order: (1) cached-feature long-context head [<1 h] → (2) pre-resized cache + longer/bigger-batch SEQ training with the new loss horizons and centred output [hours] → (3) augmentation → (4) seeds/ensemble → (5) VideoMAE port or DualTrack-style global stream. Free now: MA3 smoothing of steps for LPE/LLE.
