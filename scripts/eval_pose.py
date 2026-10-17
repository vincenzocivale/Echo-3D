"""Reload a checkpoint and evaluate it independently: official GPE/GLE/LPE/LLE (mm), per-offset pose errors, drift, breakdown by
trajectory/orientation/direction, pooling-weight diagnostics. Usage: python scripts/eval_pose.py outputs/pose/<run>/last.pt [--subset dev|train] [--every 1]"""
import argparse, json, os, sys, torch
sys.path.insert(0, os.path.dirname(__file__)); sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from train_pose import build, REPO
from usmotion4d.data import tusrec, split
from usmotion4d.evaluation import pose_eval

ap = argparse.ArgumentParser(); ap.add_argument("ckpt"); ap.add_argument("--subset", default="dev"); ap.add_argument("--every", type=int, default=1); ap.add_argument("--out")
a = ap.parse_args()
ck = torch.load(a.ckpt, map_location="cpu", weights_only=False); cfg = ck["cfg"]
model = build(cfg); missing = model.load_state_dict(ck["state"], strict=False)
assert not [k for k in missing.missing_keys if not k.startswith("encoder.")], missing  # only the (frozen, pretrained) encoder may come from elsewhere
man = split.load_manifest(os.path.join(REPO, cfg["split"])); assert ck["split_sha256"] == man["sha256"], "split manifest changed since training"
res = pose_eval.evaluate(model, tusrec.scan_paths(man[a.subset]), tuple(cfg["size"]), every=a.every, diag=True)
res["floor_identity"] = pose_eval.identity_floor(tusrec.scan_paths(man[a.subset]), every=a.every)
out = a.out or os.path.join(os.path.dirname(a.ckpt), f"eval_{a.subset}.json"); json.dump(res, open(out, "w"), indent=1)
f = lambda d: {k: round(v, 3) for k, v in d.items() if k in ("GPE", "GLE", "LPE", "LLE", "rot_mrad_k1", "trans_mm_k1", "trans_mm_k8", "sec_scan")}
print("all", f(res["all"]), "\nidentity", f(res["floor_identity"]))
for g in ("trajectory", "orientation", "direction"):
    for k, v in res[g].items(): print(f"{g}={k}", f(v))
