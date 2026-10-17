"""Observability of the elevational sign. For a checkpoint: per-axis correlation / sign accuracy of predicted xi vs GT (forward pairs (t,t+k)),
the same for reversed pairs (t+k,t), antisymmetry corr(xi_fwd, -xi_rev) (1 = uses image evidence, ~0 with corr(xi_fwd, xi_rev)~1 = prior only),
split by acquisition direction; reversed-sequence trajectory error; identity test (two identical frames).
Usage: eval_sign.py <ckpt> [--every 7]"""
import argparse, json, os, sys, torch
sys.path.insert(0, os.path.dirname(__file__)); sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from train_pose import build, REPO
from usmotion4d.data import tusrec, split
from usmotion4d.evaluation import pose_eval
from usmotion4d.evaluation.metrics import tusrec_metrics
from usmotion4d.geometry import calib, se3

ap = argparse.ArgumentParser(); ap.add_argument("ckpt"); ap.add_argument("--every", type=int, default=7); a = ap.parse_args()
ck = torch.load(a.ckpt, map_location="cpu", weights_only=False); cfg = ck["cfg"]
model = build(cfg); model.load_state_dict(ck["state"], strict=False); model.eval()
man = split.load_manifest(os.path.join(REPO, cfg["split"])); size = tuple(cfg["size"])
AX = ["wx", "wy", "wz", "vx", "vy", "vz"]
def xi_of(model, fa, fb, k):
    with torch.no_grad(), torch.autocast("cuda", torch.bfloat16): return model(fa, fb, torch.full((len(fa),), k, device=fa.device))["xi"].float()
def run_pairs(F, k, rev=False):  # xi for pairs (t, t+k) for all t, or (t+k, t) if rev
    m, out = len(F) - k, []
    for c in range(0, m, 128):
        b = min(c + 128, m); fa, fb = F[c:b], F[c + k:b + k]
        out.append(xi_of(model, fb, fa, k) if rev else xi_of(model, fa, fb, k))
    return torch.cat(out)
acc = {}; rev_traj = []; ident = []
for e in tusrec.scan_paths(man["dev"])[::a.every]:
    s = tusrec.Scan(e); F = pose_eval.encode_scan(model, s.frames(), size); P = s.poses.cuda(); n = len(P)
    for k in (1, 8):
        i = torch.arange(0, n - k, 1, device="cuda")
        gf = se3.log(se3.inverse(P[i]) @ P[i + k]).float(); gr = se3.log(se3.inverse(P[i + k]) @ P[i]).float()
        pf, pr = run_pairs(F, k), run_pairs(F, k, rev=True)
        acc.setdefault((k, s.meta["direction"]), []).append(torch.stack([pf, pr, gf, gr]).cpu())
    Fr = F.flip(0); Pr = (se3.inverse(P[-1]) @ P.flip(0))
    D = se3.exp(run_pairs(Fr, 1)).double()
    rev_traj.append(tusrec_metrics(calib.sequential(D), Pr, s.S))
    ident.append(xi_of(model, F[:64], F[:64], 1).abs().mean(0).cpu())
cor = lambda x, y: torch.corrcoef(torch.stack([x, y]))[0, 1].item()
out = {}
for k in (1, 8):
    for grp in ("DtP", "PtD", "all"):
        T = torch.cat([torch.cat(v, 1) for kk, v in acc.items() if kk[0] == k and grp in (kk[1], "all")], 1) if any(kk[0] == k for kk in acc) else None
        pf, pr, gf, gr = T
        o = {}
        for ax, nm in enumerate(AX):
            big = gf[:, ax].abs() > gf[:, ax].abs().median()
            o[nm] = dict(corr_fwd=cor(pf[:, ax], gf[:, ax]), corr_rev=cor(pr[:, ax], gr[:, ax]), antisym=cor(pf[:, ax], -pr[:, ax]), same_sign_fr=cor(pf[:, ax], pr[:, ax]),
                         sign_acc_fwd=(torch.sign(pf[big, ax]) == torch.sign(gf[big, ax])).float().mean().item(), sign_acc_rev=(torch.sign(pr[big, ax]) == torch.sign(gr[big, ax])).float().mean().item(),
                         pred_std_over_gt_std=(pf[:, ax].std() / gf[:, ax].std()).item())
        out[f"k{k}_{grp}"] = o
        print(f"k={k} {grp}: " + " ".join(f"{nm}[r={o[nm]['corr_fwd']:+.2f} rev={o[nm]['corr_rev']:+.2f} anti={o[nm]['antisym']:+.2f} sgn={o[nm]['sign_acc_fwd']:.2f}/{o[nm]['sign_acc_rev']:.2f}]" for nm in ("wx", "wz", "vx", "vz")))
out["reversed_sequence"] = {k: sum(r[k] for r in rev_traj) / len(rev_traj) for k in rev_traj[0]}
out["identity_test_mean_abs_xi"] = dict(zip(AX, torch.stack(ident).mean(0).tolist()))
print("reversed sequence", {k: round(v, 3) for k, v in out["reversed_sequence"].items()}); print("identity test |xi|", {k: round(v, 5) for k, v in out["identity_test_mean_abs_xi"].items()})
json.dump(out, open(os.path.join(os.path.dirname(a.ckpt), "eval_sign.json"), "w"), indent=1)
