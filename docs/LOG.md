# Log of what was done and found (2026-10-13)

## Decisions
1. Task: trackerless 3D freehand reconstruction (TUS-REC2024). Chosen over cardiac/other tasks because the published methods are local frame-to-frame
   estimators with accumulating drift, which a global encoder could address. Echo-only datasets (EchoNet, CAMUS, BUSI...) were surveyed and dropped.
2. Calibration/metric geometry is explicit; no scale is ever learned.
3. Pose-regression arm kept as the fundamental baseline; D4RT-style arm added on the same encoder.
4. 120x160 resolution fixed for the controlled comparison (higher resolution to be tested later).

## Validations (all pass)
- `geom.py` vs the official baseline `LabelTransform` on a real scan: global 2e-4 mm, local 1.4e-3 mm (the baseline is the less precise, float32).
- Oracle tests (`eval.py`, `test_query.py`): feeding ground-truth deltas / exact 3D points through the whole inference path (clip chaining, Kabsch) gives ~0 error
  for clip lengths 7, 64, longer than the scan; no discontinuity at clip boundaries; relative poses are gauge-free; identity queries return the calibrated pixel position.
- Bugs found and fixed: float32 rounding of relative poses (-> float64); `pose_to_T` and `pixel_grid` created tensors on CPU; bf16 head could not resolve ~1e-3 rad
  rotations (-> fp32 head); bf16 autocast fails in the CPU fast path (autocast only on CUDA).
- A single fixed batch of 8 clips is memorised (local 0.44->0.15, global 13.6->1.2 mm in 800 steps at lr 1e-4): the pipeline is learnable, not buggy.

## Reference errors on the 120 val scans (mm)
| | GPE | GLE | LPE | LLE |
|---|---|---|---|---|
| no motion (identity) | 90.22 | 88.25 | 0.391 | 0.388 |
| constant mean training step | 105.6 | 109.1 | 0.455 | 0.452 |
Published (hidden test set, NOT comparable): FlowNet GPE 5.97 / FiMoNet LPE 0.097 / baseline GPE 12.49 / LPE 0.135.

## Pilot A (pose regression, 3000 steps, bs 8, clips of 64, same data/seed)
- global: GPE 90.18, GLE 88.19, LPE 0.39, LLE 0.39 -> identical to the identity floor. Predicted deltas are ~constant and ~0
  (std 0.0007 mm vs 0.19 real; corr with GT 0.03-0.08). Same for `local4` (GPE 90.224 = floor). `local8` / `local1` were running.
- lambda_global = 0.05: local and lambda*global loss terms and gradient norms were of the same order early in training (not re-tuned).
- Cost: 0.18-0.27 s/step, 4.6-5.2 GB, 0.06-0.1 s/scan inference.
- Conclusion: at this budget the pilot cannot say whether global context helps; the model did not learn motion.

## Pilot B (D4RT-style, global and local4 encoder, 3000 steps): launched, results pending.
Smoke test: 0.33 s/step, 4.9 GB, 3.3 M decoder params; Kabsch rigid residual and singular-value ratio are logged.

## Caveats
- The curves of pilot A used every 6th val scan, which always picks the same 4 scan types per subject (biased); fixed to every 7th.
  Final-step numbers use all 120 scans.
- The mean-step "prior" looked useful on that biased subset (GPE 55) but is worse than identity on all scans.
- Official validation zip (4.8 GB) is downloading to `/raid/DATASETS/TUS-REC2024_val`; not yet used.
- Official baseline training was not reproduced (needs pytorch3d); FiMoNet/RecuVol/FlowNet not re-run on our split.
- Decoder predicts (source position + offset) rather than absolute coordinates: a deliberate conditioning choice.

## Next (needs a decision)
Make the model learn motion at all (longer training and/or a motion-friendly inductive bias) before the local-vs-global and A-vs-B questions can be answered.

## 2026-10-13 (later): pilots finished, diagnosis, pretrained feasibility
See `DIAGNOSIS_AND_FEASIBILITY.md`. Headlines: all ViT variants (pose and D4RT-style, global/local) collapsed to identity (GPE ~90 = identity floor);
EfficientNet-B1 pairwise learns, and with ImageNet weights reaches GPE 26.0 / LPE 0.207 (random, same preprocessing: 61.4 / 0.313). USFM/VideoMAE audited and costed; no long training launched.
