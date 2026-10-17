"""Evaluate multi-offset pose-graph fusion for a checkpoint (no training). Per-axis error stds per offset are estimated on training scans,
then each dev scan is reconstructed with offset sets {1} (= plain sequential composition), {1,8}, {1,2,4,8}.
Usage: eval_posegraph.py <ckpt> [--every 7] [--train-every 40]"""
import argparse, json, os, sys, time, torch
sys.path.insert(0, os.path.dirname(__file__)); sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from train_pose import build, REPO
from usmotion4d.data import tusrec, split
from usmotion4d.evaluation import pose_eval
from usmotion4d.geometry import se3, calib
from usmotion4d.optimization import pose_graph

ap = argparse.ArgumentParser(); ap.add_argument("ckpt"); ap.add_argument("--every", type=int, default=7); ap.add_argument("--train-every", type=int, default=40); a = ap.parse_args()
ck = torch.load(a.ckpt, map_location="cpu", weights_only=False); cfg = ck["cfg"]
model = build(cfg); model.load_state_dict(ck["state"], strict=False); model.eval()
man = split.load_manifest(os.path.join(REPO, cfg["split"])); size = tuple(cfg["size"]); OFFS = (1, 2, 4, 8)
SETS = {"seq_k1": (1,), "pg_1_8": (1, 8), "pg_1_2_4_8": OFFS}


def predict(s):
    F = pose_eval.encode_scan(model, s.frames(), size)
    return {k: pose_eval.predict_offset(model, F, k)[0] for k in OFFS}


# 1) per-axis error std per offset, on training scans only
err = {k: [] for k in OFFS}
for e in tusrec.scan_paths(man["train"])[::a.train_every]:
    s = tusrec.Scan(e); P = s.poses.cuda(); D = predict(s)
    for k in OFFS: err[k].append(se3.log(se3.inverse(se3.inverse(P[:-k]) @ P[k:]) @ D[k]))
sigma = {k: torch.cat(v).pow(2).mean(0).sqrt() for k, v in err.items()}
print("sigma (train) rot mrad / trans mm:", {k: [round(x, 3) for x in (v[:3] * 1e3).tolist() + v[3:].tolist()] for k, v in sigma.items()}, flush=True)

# 2) dev reconstruction with each offset set
rows = {n: [] for n in SETS}; t0 = time.time()
for e in tusrec.scan_paths(man["dev"])[::a.every]:
    s = tusrec.Scan(e); Tg = s.poses.cuda(); D = predict(s); n = len(Tg)
    for name, offs in SETS.items():
        Tp = (pose_graph.fuse({k: D[k].cpu() for k in offs}, sigma).cuda() if len(offs) > 1 else calib.sequential(D[1]))  # tiny problem: CPU is ~100x faster than a shared GPU
        r = pose_eval.trajectory_metrics(Tp, Tg, s.S, s.landmarks)
        rows[name].append(r)
out = {"sigma": {k: v.tolist() for k, v in sigma.items()}}
for name, rs in rows.items():
    out[name] = {k: sum(r[k] for r in rs) / len(rs) for k in rs[0]}
    print(name, {k: round(v, 3) for k, v in out[name].items() if k in ("GPE", "GLE", "LPE", "LLE") or k.endswith("_k8") or k.startswith("corner")}, flush=True)
print(f"{len(rows['seq_k1'])} scans, {time.time()-t0:.0f}s")
json.dump(out, open(os.path.join(os.path.dirname(a.ckpt), "eval_posegraph.json"), "w"), indent=1)
