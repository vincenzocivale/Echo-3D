"""Audit 1: (a) distribution of ground-truth relative poses per offset k and axis (plane frame: x lateral, y depth, z elevational);
(b) image-free baselines: constant mean transform per k (global, and per scan type = oracle protocol metadata) evaluated with the same metrics as the models."""
import json, os, sys, torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from usmotion4d.data import tusrec, split
from usmotion4d.geometry import calib, se3
from usmotion4d.evaluation.metrics import tusrec_metrics

KS = (1, 2, 4, 8)
man = split.load_manifest(); tr = tusrec.scan_paths(man["train"]); dv = tusrec.scan_paths(man["dev"])[::7]
def xis(entries, k):
    out = {}
    for e in entries:
        s = tusrec.Scan(e); P = s.poses; out[e[1]] = (s.meta, se3.log(se3.inverse(P[:-k]) @ P[k:]))  # xi of Delta_{t,t+k} = (omega rad, v mm)
    return out
res = dict(gt={}, baseline={})
mean_xi, type_xi = {}, {}
for k in KS:
    X = xis(tr, k); A = torch.cat([v[1] for v in X.values()])
    mean_xi[k] = A.mean(0)
    g = {}
    for a, nm in enumerate(["wx", "wy", "wz", "vx", "vy", "vz"]):
        x = A[:, a]; u = "mrad" if a < 3 else "mm"; c = 1e3 if a < 3 else 1
        g[nm] = dict(unit=u, mean=x.mean().item() * c, std=x.std().item() * c, rms=x.pow(2).mean().sqrt().item() * c, p99_abs=x.abs().quantile(.99).item() * c, frac_pos=(x > 0).float().mean().item())
    g["rot_norm_mean_mrad"] = A[:, :3].norm(dim=-1).mean().item() * 1e3; g["trans_norm_mean_mm"] = A[:, 3:].norm(dim=-1).mean().item()
    # sign consistency across scans: per-scan mean sign of vz (is the elevational direction constant within a scan, and across scan types?)
    per = {}
    for p, (m, x) in X.items(): per.setdefault((m["orientation"], m["trajectory"], m["direction"]), []).append(x.mean(0))
    sc = torch.cat([torch.stack(v)[:, 5] for v in per.values()]); g["vz_scan_mean_sign_agreement"] = max((sc > 0).float().mean().item(), (sc < 0).float().mean().item())
    g["vz_scan_mean_by_direction"] = {d: torch.cat([torch.stack(v)[:, 5] for kk, v in per.items() if kk[2] == d]).mean().item() for d in ("DtP", "PtD")}
    g["wz_scan_mean_by_direction"] = {d: torch.cat([torch.stack(v)[:, 2] for kk, v in per.items() if kk[2] == d]).mean().item() * 1e3 for d in ("DtP", "PtD")}
    res["gt"][f"k{k}"] = g
    type_xi[k] = {t: torch.stack(v).mean(0) for t, v in per.items()}
# baselines on dev (same 35 scans as the pilots)
def run(name, get):
    rows = []
    for e in dv:
        s = tusrec.Scan(e); m = s.meta; P = s.poses.cuda(); n = len(P)
        D = se3.exp(get(m, 1).double()).cuda().expand(n - 1, 4, 4)
        r = tusrec_metrics(calib.sequential(D), P, s.S, s.landmarks)
        for k in KS:
            gt = se3.inverse(P[:-k]) @ P[k:]; ek = se3.log(se3.inverse(gt) @ se3.exp(get(m, k).double().cuda()))
            r[f"rot_mrad_k{k}"], r[f"trans_mm_k{k}"] = ek[:, :3].norm(dim=-1).mean().item() * 1e3, ek[:, 3:].norm(dim=-1).mean().item()
            for a, ax in enumerate("xyz"): r[f"rot_{ax}_mrad_k{k}"], r[f"trans_{ax}_mm_k{k}"] = ek[:, a].abs().mean().item() * 1e3, ek[:, 3 + a].abs().mean().item()
        rows.append(r)
    res["baseline"][name] = {k: sum(r[k] for r in rows) / len(rows) for k in rows[0]}
run("identity", lambda m, k: torch.zeros(6))
run("mean_per_k_global", lambda m, k: mean_xi[k])
run("mean_per_k_per_scan_type_ORACLE", lambda m, k: type_xi[k][(m["orientation"], m["trajectory"], m["direction"])])
json.dump(res, open("outputs/audit_gt.json", "w"), indent=1)
for k in KS:
    g = res["gt"][f"k{k}"]; print(f"k={k} rot|w| {g['rot_norm_mean_mrad']:.2f} mrad trans|v| {g['trans_norm_mean_mm']:.2f} mm | " + " ".join(f"{a}: {g[a]['mean']:+.3f}±{g[a]['std']:.3f}" for a in ("wx", "wy", "wz", "vx", "vy", "vz")))
    print("   vz sign agreement of scan means", round(g["vz_scan_mean_sign_agreement"], 3), "vz mean by direction", {d: round(v, 3) for d, v in g["vz_scan_mean_by_direction"].items()}, "wz(mrad)", {d: round(v, 3) for d, v in g["wz_scan_mean_by_direction"].items()})
for n, b in res["baseline"].items(): print(n, {k: round(v, 3) for k, v in b.items() if k in ("GPE", "GLE", "LPE", "LLE", "rot_mrad_k1", "trans_mm_k1", "rot_mrad_k8", "trans_mm_k8", "rot_x_mrad_k8", "rot_y_mrad_k8", "rot_z_mrad_k8")})
