"""Scan-level evaluation of a PoseEstimator: relative poses at offsets, sequential composition, official metrics, breakdowns, diagnostics.
Inference reads frames only: tforms/poses are used exclusively as ground truth to compute errors (see tests/test_leakage.py)."""
import time, torch
from collections import defaultdict
from ..data import tusrec
from ..geometry import calib, se3
from ..models.pose_support import support_diagnostics
from .metrics import tusrec_metrics


@torch.no_grad()
def encode_scan(model, frames, size, bs=64):
    dev = next(model.parameters()).device
    out = []
    for a in range(0, len(frames), bs):
        x = tusrec.preprocess(frames[a:a + bs], size).to(dev)
        with torch.autocast("cuda", torch.bfloat16): out.append(model.encode(x).to(torch.bfloat16))
    return torch.cat(out)


@torch.no_grad()
def predict_offset(model, feats, k, bs=128, diag=False):
    """Delta_{t,t+k} for t = 0..n-k-1 -> [n-k,4,4] float64 (+ diagnostics of the pooling weights)."""
    n, out, d = len(feats), [], defaultdict(list)
    for a in range(0, n - k, bs):
        b = min(a + bs, n - k)
        with torch.autocast("cuda", torch.bfloat16): o = model(feats[a:b], feats[a + k:b + k], torch.full((b - a,), k, device=feats.device))
        out.append(o["delta"].double())
        if diag:
            for name in ("weights", "match_confidence", "pose_support"):
                if name in o: d[name].append(o[name].float())
    return torch.cat(out), {k_: torch.cat(v) for k_, v in d.items()}


@torch.no_grad()
def evaluate_scan(model, scan, size, offsets=(1, 2, 4, 8), diag=False):
    dev = next(model.parameters()).device
    feats = encode_scan(model, scan.frames(), size)
    Tg = scan.poses.to(dev)
    t0 = time.time()
    D1, dg = predict_offset(model, feats, 1, diag=diag)
    Tp = calib.sequential(D1)  # sequential composition of adjacent predictions
    torch.cuda.synchronize(); sec = time.time() - t0
    lm = scan.landmarks
    r = tusrec_metrics(Tp, Tg, scan.S, lm)
    for k in offsets:  # relative-pose error at each offset (mm of translation part, mrad of rotation part)
        Dk = D1 if k == 1 else predict_offset(model, feats, k)[0]
        gt = se3.inverse(Tg[:-k]) @ Tg[k:]
        e = se3.log(se3.inverse(gt) @ Dk)
        r[f"rot_mrad_k{k}"], r[f"trans_mm_k{k}"] = e[:, :3].norm(dim=-1).mean().item() * 1e3, e[:, 3:].norm(dim=-1).mean().item()
        for a, ax in enumerate("xyz"):  # per-axis mean |error| in the plane frame: x lateral, y depth, z elevational (normal to the plane)
            r[f"rot_{ax}_mrad_k{k}"], r[f"trans_{ax}_mm_k{k}"] = e[:, a].abs().mean().item() * 1e3, e[:, 3 + a].abs().mean().item()
    n = len(Tg)  # drift: error of the accumulated pose at start / middle / end of the sequence (mm, mean over pixels of corner points)
    pts = scan.S.to(dev) @ torch.tensor([[1., 640, 1, 640], [1, 1, 480, 480], [0.] * 4, [1.] * 4], device=dev, dtype=torch.float64)
    ep = ((Tp - Tg) @ pts)[:, :3].norm(dim=1).mean(1)
    for name, sl in (("start", slice(1, n // 3)), ("mid", slice(n // 3, 2 * n // 3)), ("end", slice(2 * n // 3, n))): r[f"corner_err_{name}"] = ep[sl].mean().item()
    if diag and dg:
        tm = model.tmask[None].expand(len(dg["weights"]), -1) if dg["weights"].shape[1] == model.tmask.numel() else torch.cat([model.tmask, model.tmask])[None].expand(len(dg["weights"]), -1)
        err = ((D1 - (se3.inverse(Tg[:-1]) @ Tg[1:])) @ pts)[:, :3].norm(dim=1).mean(1)
        r.update({f"diag_{k}": v for k, v in support_diagnostics(dg["weights"], tm, dg.get("match_confidence"), err).items()})
    r["sec_scan"], r["frames"] = sec, n
    return r, Tp


def evaluate(model, entries, size, every=1, **kw):
    """Mean of each metric over scans, plus breakdown by trajectory / orientation / direction. Scan-level means (a subject's scans are not independent)."""
    model.eval(); rows = []
    for e in entries[::every]:
        s = tusrec.Scan(e); r, _ = evaluate_scan(model, s, size, **kw); rows.append({**s.meta, **r})
    num = [k for k, v in rows[0].items() if isinstance(v, (int, float)) and not isinstance(v, bool)]
    mean = lambda rs: {k: sum(r[k] for r in rs) / len(rs) for k in num}
    out = dict(all=mean(rows), n_scans=len(rows))
    for g in ("trajectory", "orientation", "direction"):
        out[g] = {v: mean([r for r in rows if r[g] == v]) for v in sorted({r[g] for r in rows})}
    out["rows"] = rows
    return out


def identity_floor(entries, every=1):
    """'No motion' reference with the same metrics (and the same scans)."""
    rs = []
    for e in entries[::every]:
        s = tusrec.Scan(e); T = s.poses.cuda()
        rs.append(tusrec_metrics(torch.eye(4, dtype=torch.float64, device=T.device).expand(len(T), 4, 4).clone(), T, s.S, s.landmarks))
    return {k: sum(r[k] for r in rs) / len(rs) for k in rs[0]}


def trajectory_metrics(Tp, Tg, S, landmarks=None):
    """Metrics of a reconstructed trajectory Tp vs GT Tg ([n,4,4] T_ref_from_plane): official GPE/GLE/LPE/LLE, per-axis mean |error| of the
    relative pose (t, t+k) of the *trajectory* for k=1,8 (plane frame x lateral / y depth / z elevational), corner drift start/mid/end (mm)."""
    r = tusrec_metrics(Tp, Tg, S, landmarks)
    for k in (1, 8):
        e = se3.log(se3.inverse(se3.inverse(Tg[:-k]) @ Tg[k:]) @ (se3.inverse(Tp[:-k]) @ Tp[k:]))
        for a, ax in enumerate("xyz"): r[f"rot_{ax}_mrad_k{k}"], r[f"trans_{ax}_mm_k{k}"] = e[:, a].abs().mean().item() * 1e3, e[:, 3 + a].abs().mean().item()
    n = len(Tg)
    pts = S.to(Tg) @ torch.tensor([[1., 640, 1, 640], [1, 1, 480, 480], [0.] * 4, [1.] * 4], device=Tg.device, dtype=torch.float64)
    ep = ((Tp - Tg) @ pts)[:, :3].norm(dim=1).mean(1)
    for name, sl in (("start", slice(1, n // 3)), ("mid", slice(n // 3, 2 * n // 3)), ("end", slice(2 * n // 3, n))): r[f"corner_err_{name}"] = ep[sl].mean().item()
    return r
