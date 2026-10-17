"""Image-sensitivity of the 2x2 arms over many val scans: mean GPE/LPE with true frames vs frames of another scan vs a static frame.
python sens_many.py  -> runs/sens_many.json"""
import json, sys, torch
import data, eval as ev, pretrained as pt

dev = "cuda"; tr, va = data.split(5); val = data.Clips(va, full=True); W = "/raid/DATASETS/Echo3D_weights/matrix/"
out = {}
for enc in ("videomae", "usfm"):
    for head in ("pose", "d4rt"):
        m = pt.build(enc, head, val.S).to(dev); m.load_state_dict(torch.load(f"{W}{enc}_{head}.pt", map_location=dev)); m.eval()
        pred = ev.predict if head == "pose" else (lambda mo, x, L: ev.predict_q(mo, x, L, "anchor"))
        rs = [ev.sensitivity(m, val, 64, dev, pred, k) for k in range(0, len(val), 10)]  # 12 scans, spread over subjects and types
        out[f"{enc}+{head}"] = {t: {k: round(sum(r[t][k] for r in rs) / len(rs), 3) for k in ("GPE", "LPE", "lag16", "lag64", "mean_step_mm")} for t in ("true", "other", "static")}
        out[f"{enc}+{head}"]["per_scan_GPE_true_vs_other"] = [(r["true"]["GPE"], r["other"]["GPE"]) for r in rs]
        print(enc, head, {t: out[f"{enc}+{head}"][t]["GPE"] for t in ("true", "other", "static")}, flush=True)
json.dump(out, open("runs/sens_many.json", "w"), indent=1)
