"""Per-scan metrics (+ identity floor) of a saved pairwise checkpoint on the whole validation: python per_scan.py --ckpt runs/pairwise_imagenet.pt --name pairwise_imagenet [--norm]"""
import argparse, json, torch
import data, eval as ev, pairwise

a = argparse.ArgumentParser(); a.add_argument("--ckpt", required=True); a.add_argument("--name", required=True); a.add_argument("--norm", action="store_true"); a = a.parse_args()
tr, va = data.split(5); val = data.Clips(va, full=True)
m = pairwise.Pair(norm=a.norm).cuda(); m.load_state_dict(torch.load(a.ckpt, map_location="cuda"))
mean, recs = ev.evaluate(m, val, 64, "cuda", 1, pred=pairwise.predict_pair, per_scan=True)
json.dump(dict(mean=mean, per_scan=recs), open(f"runs/perscan_{a.name}.json", "w")); print({k: round(v, 3) for k, v in mean.items() if k in ("GPE", "GLE", "LPE", "LLE", "lag16", "lag64")})
