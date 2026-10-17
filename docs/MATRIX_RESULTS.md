# 2x2 matrix: pretrained encoder x head (3000 steps, internal validation, 120 scans; single seed)

## 1. Whole validation (mm). Primary D4RT mode = anchor (declared before looking at results); adjacent is secondary.

| Model | GPE | GLE | LPE | LLE | lag4 | lag16 | lag64 | enc/new params (M) | VRAM GB | s/step |
|---|---|---|---|---|---|---|---|---|---|---|
| no motion (identity) | 90.22 | 88.25 | 0.391 | 0.388 | 1.55 | 6.14 | 24.41 | | | |
| EfficientNet-B1 pair, ImageNet (reference baseline) | 25.96 | 24.77 | 0.207 | 0.188 | 0.73 | 2.55 | 9.04 | | | |
| EfficientNet-B1 pair, random | 61.40 | 60.83 | 0.313 | 0.303 | 1.20 | 4.55 | 17.43 | | | |
| usfm + pose | 90.23 | 88.17 | 0.391 | 0.387 | 1.55 | 6.14 | 24.41 | 85.7 pretrained / 7.4 new | 21.0 | 0.92 |
| usfm + d4rt (anchor) | 90.23 | 88.27 | 0.391 | 0.388 | 1.55 | 6.14 | 24.41 | 85.7 pretrained / 10.7 new | 21.5 | 0.85 |
| usfm + d4rt (adjacent) | 95.43 | 92.52 | 0.410 | 0.405 | 1.63 | 6.42 | 25.50 | 85.7 pretrained / 10.7 new | 21.5 | 0.85 |
| videomae + pose | 15.56 | 14.40 | 0.176 | 0.156 | 0.63 | 2.05 | 6.41 | 85.4 pretrained / 0.0 new | 13.8 | 0.58 |
| videomae + d4rt (anchor) | 27.45 | 26.60 | 0.252 | 0.238 | 0.92 | 3.07 | 9.55 | 85.4 pretrained / 3.4 new | 14.2 | 0.62 |
| videomae + d4rt (adjacent) | 35.45 | 35.12 | 0.246 | 0.235 | 0.93 | 3.29 | 11.16 | 85.4 pretrained / 3.4 new | 14.2 | 0.62 |

## 2. Learning curves (GPE / LPE on every 7th val scan; last point = all 120 scans)

| Model | step 500 | step 1000 | step 1500 | step 2000 | step 2500 | step 3000 |
|---|---|---|---|---|---|---|
| usfm + pose | 93.7 / 0.422 | 90.3 / 0.414 | 85.3 / 0.403 | 90.3 / 0.395 | 90.2 / 0.395 | 90.2 / 0.391 |
| usfm + d4rt | 89.6 / 0.394 | 89.6 / 0.394 | 89.5 / 0.394 | 89.6 / 0.394 | 89.6 / 0.394 | 90.2 / 0.391 |
| videomae + pose | 54.6 / 0.269 | 43.8 / 0.250 | 33.5 / 0.218 | 21.2 / 0.191 | 16.2 / 0.183 | 15.6 / 0.176 |
| videomae + d4rt | 74.6 / 0.378 | 43.6 / 0.318 | 31.6 / 0.282 | 33.5 / 0.270 | 27.8 / 0.260 | 27.5 / 0.252 |

Image sensitivity during training (mean output distance in mm, true clip vs clip of another scan; 0 = ignores images):

| Model | step 500 | step 1000 | step 1500 | step 2000 | step 2500 | step 3000 |
|---|---|---|---|---|---|---|
| usfm + pose | 0.003 | 0.004 | 0.000 | 0.000 | 0.000 | 0.000 |
| usfm + d4rt | 0.002 | 0.004 | 0.005 | 0.007 | 0.007 | 0.007 |
| videomae + pose | 0.577 | 0.600 | 0.523 | 0.517 | 0.701 | 0.620 |
| videomae + d4rt | 2.918 | 6.010 | 6.700 | 6.400 | 6.243 | 6.353 |

## 3. Per acquisition pattern (val; GPE skill = 1 - GPE/GPE_identity of the same scans; mean over scans)


**arm** (GPE skill / LPE skill)

| Model | LH | RH |
|---|---|---|
| usfm+pose | +0.00 / +0.000 | -0.00 / +0.000 |
| usfm+d4rt/anchor | -0.00 / -0.001 | -0.00 / -0.001 |
| videomae+pose | +0.82 / +0.559 | +0.83 / +0.540 |
| videomae+d4rt/anchor | +0.70 / +0.365 | +0.69 / +0.345 |
| pair-ImageNet | +0.74 / +0.491 | +0.69 / +0.451 |
| pair-random | +0.33 / +0.209 | +0.31 / +0.190 |

**probe orientation** (GPE skill / LPE skill)

| Model | Par | Per |
|---|---|---|
| usfm+pose | +0.00 / +0.000 | -0.00 / +0.000 |
| usfm+d4rt/anchor | -0.00 / -0.001 | -0.00 / -0.001 |
| videomae+pose | +0.83 / +0.527 | +0.82 / +0.571 |
| videomae+d4rt/anchor | +0.70 / +0.341 | +0.69 / +0.370 |
| pair-ImageNet | +0.81 / +0.524 | +0.62 / +0.418 |
| pair-random | +0.47 / +0.302 | +0.17 / +0.097 |

**shape** (GPE skill / LPE skill)

| Model | C | L | S |
|---|---|---|---|
| usfm+pose | -0.00 / +0.000 | +0.00 / -0.000 | -0.00 / +0.000 |
| usfm+d4rt/anchor | -0.00 / -0.001 | -0.00 / -0.001 | -0.00 / -0.001 |
| videomae+pose | +0.82 / +0.545 | +0.86 / +0.618 | +0.80 / +0.485 |
| videomae+d4rt/anchor | +0.68 / +0.347 | +0.72 / +0.422 | +0.69 / +0.297 |
| pair-ImageNet | +0.68 / +0.456 | +0.75 / +0.540 | +0.71 / +0.418 |
| pair-random | +0.29 / +0.186 | +0.36 / +0.248 | +0.30 / +0.166 |

**direction** (GPE skill / LPE skill)

| Model | DtP | PtD |
|---|---|---|
| usfm+pose | -0.01 / -0.004 | +0.01 / +0.004 |
| usfm+d4rt/anchor | -0.00 / -0.000 | -0.00 / -0.002 |
| videomae+pose | +0.82 / +0.559 | +0.83 / +0.539 |
| videomae+d4rt/anchor | +0.68 / +0.360 | +0.71 / +0.351 |
| pair-ImageNet | +0.71 / +0.453 | +0.72 / +0.490 |
| pair-random | +0.32 / +0.197 | +0.32 / +0.202 |

## 4. Diagnosis (runs/diag_*.json)


### usfm + pose

- **train** per-step translation: std ratio pred/GT [0.0019, 0.0052, 0.0206], corr with GT [0.0061, 0.0102, 0.011], step-direction cosine 0.0082; GPE model/identity 85.6875/85.6724, LPE 0.384/0.3839
  - sensitivity (GPE true / other scan / static): scan 0: 69.417 / 69.412 / 69.413; scan 541: 116.529 / 116.532 / 116.532
  - skill vs identity by elevational distance: {'0-1': 0.0006, '1-3': -0.0001, '3-10': 0.0, '10-30': -0.0001, '30-1000000000.0': 0.0003}; by in-plane overlap: {'0.75-1.01': 0.0, '0.25-0.75': -0.0051}
- **val** per-step translation: std ratio pred/GT [0.0018, 0.0066, 0.0202], corr with GT [0.0113, -0.02, 0.0105], step-direction cosine 0.0233; GPE model/identity 91.7577/91.7571, LPE 0.4025/0.4025
  - sensitivity (GPE true / other scan / static): scan 0: 97.199 / 97.207 / 97.224; scan 61: 99.403 / 99.414 / 99.389
  - skill vs identity by elevational distance: {'0-1': 0.001, '1-3': -0.0003, '3-10': 0.0008, '10-30': -0.0003, '30-1000000000.0': 0.0011}; by in-plane overlap: {'0.75-1.01': 0.0002, '0.25-0.75': 0.0004}
- **feature vs output sensitivity (val)**: {'other_scan': {'enc_rel_change': 0.0043, 'enc_token_cosine': 1.0, 'out_change_mm': 0.0002}, 'static': {'enc_rel_change': 0.0025, 'enc_token_cosine': 1.0, 'out_change_mm': 0.0001}, 'time_reversed': {'enc_rel_change': 0.0024, 'enc_token_cosine': 1.0, 'out_change_mm': 0.0001}}
- gradient norm per module (8 batches): {'enc.backbone': 0.0008, 'enc.proj': 0.0152}; cross-batch cosine {'encoder': -0.0168, 'decoder_or_head': 0.0646}

### usfm + d4rt

- **train** per-step translation: std ratio pred/GT [0.0359, 0.0883, 0.019], corr with GT [0.011, -0.0027, 0.0063], step-direction cosine 0.0061; GPE model/identity 85.6751/85.6724, LPE 0.3843/0.3839
  - sensitivity (GPE true / other scan / static): scan 0: 69.085 / 69.071 / 69.07; scan 541: 117.132 / 117.128 / 117.138
  - skill vs identity by elevational distance: {'0-1': -0.0001, '1-3': -0.0001, '3-10': 0.0001, '10-30': 0.0001, '30-1000000000.0': -0.0007}; by in-plane overlap: {'0.75-1.01': 0.0001, '0.25-0.75': -0.0001}
  - D4RT identity queries: error 0.1731 mm (offset std 0.1217); non-identity: error 5.1152 vs no-motion 5.098 (skill -0.0034), corr -0.039, std ratio 0.0338, error with other scan's images 5.1153
  - by |dt| (skill / corr / std ratio): {'1': (-0.1038, 0.0026, 0.4952), '2-3': (-0.0199, 0.0022, 0.2107), '4-15': (-0.0036, -0.0226, 0.0635), '16-31': (-0.0016, -0.0419, 0.0303), '32-63': (-0.0014, -0.0626, 0.0211)}
- **val** per-step translation: std ratio pred/GT [0.0339, 0.1121, 0.0186], corr with GT [0.0036, 0.0045, 0.0064], step-direction cosine 0.0028; GPE model/identity 91.7611/91.7571, LPE 0.4029/0.4025
  - sensitivity (GPE true / other scan / static): scan 0: 96.814 / 96.802 / 96.803; scan 61: 99.966 / 99.959 / 99.978
  - skill vs identity by elevational distance: {'0-1': -0.0002, '1-3': -0.0001, '3-10': 0.0, '10-30': 0.0002, '30-1000000000.0': -0.0009}; by in-plane overlap: {'0.75-1.01': 0.0, '0.25-0.75': 0.0003}
  - D4RT identity queries: error 0.1732 mm (offset std 0.1218); non-identity: error 5.683 vs no-motion 5.6606 (skill -0.004), corr -0.0627, std ratio 0.0306, error with other scan's images 5.6832
  - by |dt| (skill / corr / std ratio): {'1': (-0.093, -0.0141, 0.4531), '2-3': (-0.0176, -0.0037, 0.1906), '4-15': (-0.0041, -0.0346, 0.0574), '16-31': (-0.0027, -0.0841, 0.0278), '32-63': (-0.002, -0.0882, 0.0191)}
- **feature vs output sensitivity (val)**: {'other_scan': {'enc_rel_change': 0.0247, 'enc_token_cosine': 0.9997, 'out_change_mm_feature_only': 0.0007, 'out_change_mm_patch_only': 0.0079, 'out_change_mm_all': 0.008, 'gt_motion_mm': 5.6626, 'out_change_over_gt_motion': 0.0014}, 'static': {'enc_rel_change': 0.0124, 'enc_token_cosine': 0.9999, 'out_change_mm_feature_only': 0.0005, 'out_change_mm_patch_only': 0.0032, 'out_change_mm_all': 0.0033, 'gt_motion_mm': 5.6626, 'out_change_over_gt_motion': 0.0006}, 'time_reversed': {'enc_rel_change': 0.0123, 'enc_token_cosine': 0.9999, 'out_change_mm_feature_only': 0.0004, 'out_change_mm_patch_only': 0.0033, 'out_change_mm_all': 0.0034, 'gt_motion_mm': 5.6626, 'out_change_over_gt_motion': 0.0006}}
- **Kabsch**: {'frames 1-3': {'raw_point_err_mm': 0.8944, 'after_rigid_projection_mm': 0.8821, 'no_motion_err_mm': 0.8619, 'rigid_rms_mm': 0.1618, 'sv_ratio_mean': 0.5227, 'sv_ratio_min': 0.5225}, 'frames 4-15': {'raw_point_err_mm': 3.9608, 'after_rigid_projection_mm': 3.9578, 'no_motion_err_mm': 3.9398, 'rigid_rms_mm': 0.1598, 'sv_ratio_mean': 0.5226, 'sv_ratio_min': 0.5224}, 'frames 16-63': {'raw_point_err_mm': 15.4356, 'after_rigid_projection_mm': 15.435, 'no_motion_err_mm': 15.4203, 'rigid_rms_mm': 0.1573, 'sv_ratio_mean': 0.5224, 'sv_ratio_min': 0.5218}}
- adjacent mode (val): GPE 96.304 (identity 91.7571), LPE 0.4214 (identity 0.4025)
- gradient norm per module (8 batches): {'enc.backbone': 0.0, 'enc.proj': 0.0009, 'dec.kv': 0.0058, 'dec.blocks.0': 0.011, 'dec.blocks.1': 0.0084, 'dec.blocks.2': 0.0104, 'dec.blocks.3': 0.0093}; cross-batch cosine {'encoder': 0.7669, 'decoder_or_head': 0.9613}

### videomae + pose

- **train** per-step translation: std ratio pred/GT [0.8898, 0.4545, 0.8445], corr with GT [0.9164, 0.5374, 0.7891], step-direction cosine 0.8416; GPE model/identity 14.4551/85.6724, LPE 0.1861/0.3839
  - sensitivity (GPE true / other scan / static): scan 0: 19.712 / 23.241 / 82.09; scan 541: 31.055 / 43.622 / 104.77
  - skill vs identity by elevational distance: {'0-1': 0.6807, '1-3': 0.7288, '3-10': 0.7029, '10-30': 0.7332, '30-1000000000.0': 0.8192}; by in-plane overlap: {'0.75-1.01': 0.7212, '0.25-0.75': 0.8192}
- **val** per-step translation: std ratio pred/GT [0.8947, 0.5352, 0.7968], corr with GT [0.9302, 0.5365, 0.8055], step-direction cosine 0.8728; GPE model/identity 17.5747/91.7571, LPE 0.1788/0.4025
  - sensitivity (GPE true / other scan / static): scan 0: 39.955 / 33.908 / 102.547; scan 61: 9.142 / 33.086 / 97.695
  - skill vs identity by elevational distance: {'0-1': 0.7228, '1-3': 0.7487, '3-10': 0.7137, '10-30': 0.7235, '30-1000000000.0': 0.8136}; by in-plane overlap: {'0.75-1.01': 0.7272, '0.25-0.75': 0.7064}
- **feature vs output sensitivity (val)**: {'other_scan': {'enc_rel_change': 0.3901, 'enc_token_cosine': 0.9213, 'out_change_mm': 0.5926}, 'static': {'enc_rel_change': 0.2545, 'enc_token_cosine': 0.9659, 'out_change_mm': 0.3546}, 'time_reversed': {'enc_rel_change': 0.3413, 'enc_token_cosine': 0.9383, 'out_change_mm': 0.6937}}
- gradient norm per module (8 batches): {'enc.embed': 0.3624, 'enc.blocks.0': 0.363, 'enc.blocks.1': 0.3081, 'enc.blocks.2': 0.5429, 'enc.blocks.3': 0.5999, 'enc.blocks.4': 0.3843, 'enc.blocks.5': 0.255, 'enc.blocks.6': 0.1859, 'enc.blocks.7': 0.1511, 'enc.blocks.8': 0.1097, 'enc.blocks.9': 0.1021, 'enc.blocks.10': 0.0934, 'enc.blocks.11': 0.1284}; cross-batch cosine {'encoder': -0.0437, 'decoder_or_head': -0.0169}

### videomae + d4rt

- **train** per-step translation: std ratio pred/GT [0.8522, 0.2295, 0.7907], corr with GT [0.723, 0.194, 0.628], step-direction cosine 0.6578; GPE model/identity 24.5144/85.6724, LPE 0.2599/0.3839
  - sensitivity (GPE true / other scan / static): scan 0: 23.224 / 27.871 / 81.322; scan 541: 55.663 / 65.506 / 104.285
  - skill vs identity by elevational distance: {'0-1': 0.5246, '1-3': 0.5897, '3-10': 0.5641, '10-30': 0.6152, '30-1000000000.0': 0.5373}; by in-plane overlap: {'0.75-1.01': 0.5866, '0.25-0.75': 0.5482}
  - D4RT identity queries: error 0.2053 mm (offset std 0.144); non-identity: error 1.708 vs no-motion 5.098 (skill 0.665), corr 0.9282, std ratio 0.9051, error with other scan's images 2.4457
  - by |dt| (skill / corr / std ratio): {'1': (0.2584, 0.6957, 1.0234), '2-3': (0.4999, 0.8406, 0.8849), '4-15': (0.6282, 0.9076, 0.8999), '16-31': (0.6799, 0.9264, 0.8979), '32-63': (0.6959, 0.9325, 0.9094)}
- **val** per-step translation: std ratio pred/GT [0.853, 0.2825, 0.7266], corr with GT [0.7523, 0.2275, 0.6485], step-direction cosine 0.6957; GPE model/identity 30.3702/91.7571, LPE 0.257/0.4025
  - sensitivity (GPE true / other scan / static): scan 0: 71.889 / 60.846 / 99.009; scan 61: 26.556 / 39.687 / 87.211
  - skill vs identity by elevational distance: {'0-1': 0.5545, '1-3': 0.6101, '3-10': 0.5856, '10-30': 0.5811, '30-1000000000.0': 0.5735}; by in-plane overlap: {'0.75-1.01': 0.5849, '0.25-0.75': 0.5696}
  - D4RT identity queries: error 0.2064 mm (offset std 0.1441); non-identity: error 2.1706 vs no-motion 5.6606 (skill 0.6165), corr 0.9102, std ratio 0.8332, error with other scan's images 3.2854
  - by |dt| (skill / corr / std ratio): {'1': (0.2645, 0.6715, 0.94), '2-3': (0.4712, 0.8137, 0.8042), '4-15': (0.5732, 0.8789, 0.8121), '16-31': (0.6267, 0.9047, 0.836), '32-63': (0.6524, 0.9177, 0.8348)}
- **feature vs output sensitivity (val)**: {'other_scan': {'enc_rel_change': 0.9337, 'enc_token_cosine': 0.5591, 'out_change_mm_feature_only': 8.0479, 'out_change_mm_patch_only': 0.0313, 'out_change_mm_all': 8.0471, 'gt_motion_mm': 5.6626, 'out_change_over_gt_motion': 1.4211}, 'static': {'enc_rel_change': 0.6209, 'enc_token_cosine': 0.8095, 'out_change_mm_feature_only': 4.5433, 'out_change_mm_patch_only': 0.0241, 'out_change_mm_all': 4.5434, 'gt_motion_mm': 5.6626, 'out_change_over_gt_motion': 0.8023}, 'time_reversed': {'enc_rel_change': 0.7969, 'enc_token_cosine': 0.6621, 'out_change_mm_feature_only': 9.0394, 'out_change_mm_patch_only': 0.0245, 'out_change_mm_all': 9.0385, 'gt_motion_mm': 5.6626, 'out_change_over_gt_motion': 1.5962}}
- **Kabsch**: {'frames 1-3': {'raw_point_err_mm': 0.4504, 'after_rigid_projection_mm': 0.4106, 'no_motion_err_mm': 0.8619, 'rigid_rms_mm': 0.1675, 'sv_ratio_mean': 0.5221, 'sv_ratio_min': 0.5219}, 'frames 4-15': {'raw_point_err_mm': 1.6055, 'after_rigid_projection_mm': 1.5912, 'no_motion_err_mm': 3.9398, 'rigid_rms_mm': 0.179, 'sv_ratio_mean': 0.5222, 'sv_ratio_min': 0.5218}, 'frames 16-63': {'raw_point_err_mm': 5.3794, 'after_rigid_projection_mm': 5.3705, 'no_motion_err_mm': 15.4203, 'rigid_rms_mm': 0.271, 'sv_ratio_mean': 0.5226, 'sv_ratio_min': 0.5177}}
- adjacent mode (val): GPE 35.9258 (identity 91.7571), LPE 0.2488 (identity 0.4025)
- gradient norm per module (8 batches): {'enc.embed': 0.3882, 'enc.blocks.0': 0.3815, 'enc.blocks.1': 0.3239, 'enc.blocks.2': 0.491, 'enc.blocks.3': 0.5504, 'enc.blocks.4': 0.3847, 'enc.blocks.5': 0.2715, 'enc.blocks.6': 0.2109, 'enc.blocks.7': 0.1621, 'enc.blocks.8': 0.1226, 'enc.blocks.9': 0.0946, 'enc.blocks.10': 0.0751, 'enc.blocks.11': 0.0612, 'dec.kv': 0.2682, 'dec.blocks.0': 0.2102, 'dec.blocks.1': 0.0929, 'dec.blocks.2': 0.0694, 'dec.blocks.3': 0.0744}; cross-batch cosine {'encoder': 0.0058, 'decoder_or_head': 0.8166}
