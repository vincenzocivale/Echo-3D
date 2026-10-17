"""Final comparison of trajectory reconstructions on the same scans: identity, pair model (sequential / pose graph {1,8}), sequence model,
and the hybrid pose graph (sequence-model adjacent steps + pair-model direct k=8 estimates). Error stds for the pose-graph weights come
from training scans only (cached in outputs/final_sigma.json). --subset dev (every 7th scan, used for choices) | val (TUS-REC2024_val,
3 subjects, never used for choices: run once with the configuration fixed on dev).
Usage: eval_final.py --pair <pair ckpt> --seq <seq ckpt> [--subset dev|val]"""
import argparse, json, os, sys, time, torch
sys.path.insert(0, os.path.dirname(__file__)); sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import train_pose, train_seq
from usmotion4d.data import tusrec, split
from usmotion4d.evaluation import pose_eval
from usmotion4d.geometry import se3, calib
from usmotion4d.optimization import pose_graph

ap = argparse.ArgumentParser(); ap.add_argument("--pair", required=True); ap.add_argument("--seq", required=True); ap.add_argument("--subset", default="dev"); a = ap.parse_args()
REPO = train_pose.REPO
ckp = torch.load(a.pair, map_location="cpu", weights_only=False); pair = train_pose.build(ckp["cfg"]); pair.load_state_dict(ckp["state"], strict=False); pair.eval()
cks = torch.load(a.seq, map_location="cpu", weights_only=False); seq = train_seq.build(cks["cfg"], init=False); seq.load_state_dict(cks["state"]); seq.eval()
size_p, size_s = tuple(ckp["cfg"]["size"]), tuple(cks["cfg"]["size"])
man = split.load_manifest(os.path.join(REPO, ckp["cfg"]["split"]))


def preds(s):
    F = s.frames(); E = pose_eval.encode_scan(pair, F, size_p)
    return dict(pair1=pose_eval.predict_offset(pair, E, 1)[0], pair8=pose_eval.predict_offset(pair, E, 8)[0], seq1=se3.exp(train_seq.predict_steps(seq, F, size_s)))


sig_path = os.path.join(REPO, "outputs", "final_sigma.json")
if os.path.exists(sig_path):
    sigma = {k: torch.tensor(v, dtype=torch.float64) for k, v in json.load(open(sig_path)).items()}
else:  # per-axis error std of each measurement type, on training scans only
    err = dict(pair1=[], pair8=[], seq1=[])
    for e in tusrec.scan_paths(man["train"])[::40]:
        s = tusrec.Scan(e); P = s.poses.cuda(); D = preds(s)
        for name, k in (("pair1", 1), ("pair8", 8), ("seq1", 1)): err[name].append(se3.log(se3.inverse(se3.inverse(P[:-k]) @ P[k:]) @ D[name]))
    sigma = {k: torch.cat(v).pow(2).mean(0).sqrt().cpu() for k, v in err.items()}
    json.dump({k: v.tolist() for k, v in sigma.items()}, open(sig_path, "w"), indent=1)
print("sigma (train) rot mrad x/y/z, trans mm x/y/z:", {k: [round(x, 3) for x in (v[:3] * 1e3).tolist() + v[3:].tolist()] for k, v in sigma.items()}, flush=True)

if a.subset == "val":
    root = tusrec.VAL_ROOT; entries = tusrec.scan_paths(sorted(os.listdir(root + "frames")), root=root)
else:
    root = tusrec.ROOT; entries = tusrec.scan_paths(man["dev"])[::7]
METHODS = ["identity", "pair_seq", "pair_pg18", "seq", "seq_pg_pair8"]
rows = {m: [] for m in METHODS}; t0 = time.time()
for e in entries:
    s = tusrec.Scan(e, root=root); Tg = s.poses.cuda(); D = preds(s); n = len(Tg)
    pg = lambda k1, s1: pose_graph.fuse({1: D[k1].cpu(), 8: D["pair8"].cpu()}, {1: sigma[s1], 8: sigma["pair8"]}).cuda()  # CPU: tiny problem
    T = dict(identity=torch.eye(4, dtype=torch.float64, device="cuda").expand(n, 4, 4).clone(), pair_seq=calib.sequential(D["pair1"]),
             pair_pg18=pg("pair1", "pair1"), seq=calib.sequential(D["seq1"]), seq_pg_pair8=pg("seq1", "seq1"))
    for m in METHODS: rows[m].append({**s.meta, **pose_eval.trajectory_metrics(T[m], Tg, s.S, s.landmarks)})
out = dict(subset=a.subset, n_scans=len(entries), pair=a.pair, seq=a.seq)
KEYS = ("GPE", "GLE", "LPE", "LLE", "rot_x_mrad_k8", "rot_y_mrad_k8", "rot_z_mrad_k8", "trans_z_mm_k8", "corner_err_end")
for m in METHODS:
    num = [k for k, v in rows[m][0].items() if isinstance(v, float)]
    out[m] = {k: sum(r[k] for r in rows[m]) / len(rows[m]) for k in num}
    out[m]["by_subject_GPE"] = {sub: sum(r["GPE"] for r in rows[m] if r["subject"] == sub) / sum(r["subject"] == sub for r in rows[m]) for sub in sorted({r["subject"] for r in rows[m]})}
    print(f"{m:14s}", {k: round(out[m][k], 3) for k in KEYS}, flush=True)
print("GPE by subject:", {m: {k: round(v, 1) for k, v in out[m]["by_subject_GPE"].items()} for m in METHODS} if a.subset == "val" else "", f"| {len(entries)} scans, {time.time()-t0:.0f}s")
json.dump(dict(out, rows=rows), open(os.path.join(REPO, "outputs", f"final_{a.subset}.json"), "w"), indent=1)
