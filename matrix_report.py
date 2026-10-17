"""Assemble the 2x2 matrix report (markdown) from run jsons: python matrix_report.py > docs/MATRIX_RESULTS.md
Reads runs/matrix/code/runs/{enc}_{head}.json, runs/diag_{enc}_{head}.json (if present), runs/perscan_pairwise_*.json, runs/floors_val120.txt."""
import json, os
from collections import defaultdict

R = "runs/"; M = R + "matrix/code/runs/"
ARMS = [("usfm", "pose"), ("usfm", "d4rt"), ("videomae", "pose"), ("videomae", "d4rt")]
ld = lambda p: json.load(open(p)) if os.path.exists(p) else None
f = lambda v, n=2: "-" if v is None else f"{v:.{n}f}"
fl = json.loads(open(R + "floors_val120.txt").read().strip().splitlines()[-1])["none"]
runs = {(e, h): ld(f"{M}{e}_{h}.json") for e, h in ARMS}
pw = {n: ld(f"{R}perscan_{n}.json") for n in ("pairwise_imagenet", "pairwise_random_norm")}
final = lambda r, mode: None if r is None else r["eval_log"][-1][mode]
print("# 2x2 matrix: pretrained encoder x head (3000 steps, internal validation, 120 scans; single seed)\n")
print("## 1. Whole validation (mm). Primary D4RT mode = anchor (declared before looking at results); adjacent is secondary.\n")
print("| Model | GPE | GLE | LPE | LLE | lag4 | lag16 | lag64 | enc/new params (M) | VRAM GB | s/step |\n|---|---|---|---|---|---|---|---|---|---|---|")
print(f"| no motion (identity) | {f(fl['GPE'])} | {f(fl['GLE'])} | {f(fl['LPE'],3)} | {f(fl['LLE'],3)} | {f(fl['lag4'])} | {f(fl['lag16'])} | {f(fl['lag64'])} | | | |")
for n, label in (("pairwise_imagenet", "EfficientNet-B1 pair, ImageNet (reference baseline)"), ("pairwise_random_norm", "EfficientNet-B1 pair, random")):
    m = pw[n]["mean"] if pw[n] else None
    if m: print(f"| {label} | {f(m['GPE'])} | {f(m['GLE'])} | {f(m['LPE'],3)} | {f(m['LLE'],3)} | {f(m['lag4'])} | {f(m['lag16'])} | {f(m['lag64'])} | | | |")
for (e, h), r in runs.items():
    if r is None: print(f"| {e} + {h} | (not finished) |"); continue
    for mode in (["pose"] if h == "pose" else ["anchor", "adjacent"]):
        m = final(r, mode); au = r["audit"]
        print(f"| {e} + {h}{'' if h == 'pose' else ' (' + mode + ')'} | {f(m['GPE'])} | {f(m['GLE'])} | {f(m['LPE'],3)} | {f(m['LLE'],3)} | {f(m['lag4'])} | {f(m['lag16'])} | {f(m['lag64'])} | "
              f"{au['loaded_params']/1e6:.1f} pretrained / {r['params_m'] - au['loaded_params']/1e6:.1f} new | {r['peak_mem_gb']:.1f} | {r['train_sec_per_step']:.2f} |")
print("\n## 2. Learning curves (GPE / LPE on every 7th val scan; last point = all 120 scans)\n")
print("| Model | " + " | ".join(f"step {s}" for s in (500, 1000, 1500, 2000, 2500, 3000)) + " |\n|---|" + "---|" * 6)
for (e, h), r in runs.items():
    if r:
        for mode in (["pose"] if h == "pose" else ["anchor"]):
            print(f"| {e} + {h} | " + " | ".join(f"{f(x[mode]['GPE'],1)} / {f(x[mode]['LPE'],3)}" for x in r["eval_log"]) + " |")
print("\nImage sensitivity during training (mean output distance in mm, true clip vs clip of another scan; 0 = ignores images):\n")
print("| Model | " + " | ".join(f"step {s}" for s in (500, 1000, 1500, 2000, 2500, 3000)) + " |\n|---|" + "---|" * 6)
for (e, h), r in runs.items():
    if r: print(f"| {e} + {h} | " + " | ".join(f"{x['img_sensitivity_mm']:.3f}" for x in r["eval_log"]) + " |")
print("\n## 3. Per acquisition pattern (val; GPE skill = 1 - GPE/GPE_identity of the same scans; mean over scans)\n")
recs = {f"{e}+{h}" + ("" if h == "pose" else "/anchor"): r["eval_log"][-1][("pose" if h == "pose" else "anchor") + "_per_scan"] for (e, h), r in runs.items() if r}
for n, lbl in (("pairwise_imagenet", "pair-ImageNet"), ("pairwise_random_norm", "pair-random")):
    if pw[n]: recs[lbl] = pw[n]["per_scan"]
fac = {"arm": lambda s: s.split("/")[1].split("_")[0], "probe orientation": lambda s: s.split("/")[1].split("_")[1], "shape": lambda s: s.split("/")[1].split("_")[2], "direction": lambda s: s.split("/")[1].split("_")[3]}
for fname, fn in fac.items():
    levels = sorted({fn(x["scan"]) for x in next(iter(recs.values()))})
    print(f"\n**{fname}** (GPE skill / LPE skill)\n\n| Model | " + " | ".join(levels) + " |\n|---|" + "---|" * len(levels))
    for k, rs in recs.items():
        cells = []
        for lv in levels:
            sel = [x for x in rs if fn(x["scan"]) == lv]
            cells.append(f"{1 - sum(x['GPE'] for x in sel) / sum(x['GPE_identity'] for x in sel):+.2f} / {1 - sum(x['LPE'] for x in sel) / sum(x['LPE_identity'] for x in sel):+.3f}")
        print(f"| {k} | " + " | ".join(cells) + " |")
print("\n## 4. Diagnosis (runs/diag_*.json)\n")
for (e, h), r in runs.items():
    d = ld(f"{R}diag_{e}_{h}.json")
    if d is None: continue
    print(f"\n### {e} + {h}\n")
    for s in ("train", "val"):
        x = d[s]["deltas_and_lags"]; c = x["components"]
        print(f"- **{s}** per-step translation: std ratio pred/GT {[c[k]['std_ratio'] for k in ('tx','ty','tz')]}, corr with GT {[c[k]['corr'] for k in ('tx','ty','tz')]}, "
              f"step-direction cosine {x['step_cosine_mean']}; GPE model/identity {x['metrics_vs_identity']['GPE']['model']}/{x['metrics_vs_identity']['GPE']['identity']}, LPE {x['metrics_vs_identity']['LPE']['model']}/{x['metrics_vs_identity']['LPE']['identity']}")
        print(f"  - sensitivity (GPE true / other scan / static): " + "; ".join(f"scan {k}: {v['true']['GPE']} / {v['other']['GPE']} / {v['static']['GPE']}" for k, v in d[s]['sensitivity'].items()))
        print("  - skill vs identity by elevational distance: " + str({k: v['skill'] for k, v in d[s]['overlap']['elevational_distance_mm'].items()}) + "; by in-plane overlap: " + str({k: v['skill'] for k, v in d[s]['overlap']['inplane_overlap_fraction'].items()}))
        if "queries" in d[s]:
            q = d[s]["queries"]; i, n = q["identity_all"], q["non_identity_all"]
            print(f"  - D4RT identity queries: error {i['err_model']} mm (offset std {i['pred_offset_std_mm']}); non-identity: error {n['err_model']} vs no-motion {n['err_identity']} (skill {n['skill']}), corr {n['corr_offset']}, std ratio {n['std_ratio']}, error with other scan's images {n['err_image_swap']}")
            print("  - by |dt| (skill / corr / std ratio): " + str({k: (v['skill'], v['corr_offset'], v['std_ratio']) for k, v in q['by_dt'].items() if v and k != 'id'}))
    fs = d.get("feature_sensitivity")
    if fs: print("- **feature vs output sensitivity (val)**: " + str(fs))
    if "kabsch_quality" in d: print("- **Kabsch**: " + str(d["kabsch_quality"]))
    if "val_adjacent_mode" in d:
        m = d["val_adjacent_mode"]["metrics_vs_identity"]; print(f"- adjacent mode (val): GPE {m['GPE']['model']} (identity {m['GPE']['identity']}), LPE {m['LPE']['model']} (identity {m['LPE']['identity']})")
    g = d["grads"]; print("- gradient norm per module (8 batches): " + str({k: v['grad_norm'] for k, v in g['per_module'].items() if k.count('.') <= 2 and v['params_m'] > 0.05}) + f"; cross-batch cosine {g['mean_pairwise_cosine_across_batches']}")
