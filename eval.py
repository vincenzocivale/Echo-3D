"""TUS-REC metrics (mm): GPE/GLE = displacement error vs frame 0, LPE/LLE = vs previous frame (pixel/landmark).
Extras: lag{k} = point error of the relative pose T_{i+k->i} (consistency at temporal distance k);
bnd/int = local 5-point error on deltas at clip boundaries vs interior (chaining discontinuity)."""
import h5py, torch
from geom import pixel_grid, uv_to_mm, kabsch, Rt_to_T
from model import accumulate

LAGS = (1, 4, 16, 64, 256)
IDX = lambda p: (p.split("/")[-2], p.split("/")[-1][:-3])  # path -> (subject, scan name)


@torch.no_grad()
def predict(model, x, L):
    """Deltas T_{i->i-1} for a whole scan x [n,1,h,w]: clips of L frames sharing 1 frame, chained. Only x is used.
    Returns (deltas [n-1,4,4], indices of the first delta of each clip)."""
    n, d, a, starts = len(x), [], 0, []
    while a < n - 1:
        b = min(a + L - 1, n - 1)
        starts.append(a)
        with torch.autocast(x.device.type, torch.bfloat16, enabled=x.device.type == "cuda"):
            d.append(model(x[a:b + 1][None])[0])
        a = b
    return torch.cat(d), starts


@torch.no_grad()
def predict_q(model, x, L, mode="anchor", grid=(8, 8)):
    """Pose recovery from a D4RT-style model, using only x: uniform `grid` of query pixels per frame (see predict_uv)."""
    dev = x.device
    gv, gu = torch.meshgrid((torch.arange(grid[0], device=dev) + .5) / grid[0], (torch.arange(grid[1], device=dev) + .5) / grid[1], indexing="ij")
    return predict_uv(model, x, L, mode, torch.stack([gu.flatten(), gv.flatten()], -1))


@torch.no_grad()
def predict_uv(model, x, L, mode, uv):
    """Per clip (L frames, 1 shared): query the pixels uv [K,2] of every frame f in the frame of the clip's first frame (mode="anchor") or of
    f-1 ("adjacent"), then rigid Kabsch fit (no scale) of the known planar source points onto the predicted 3D points -> T_{f->ref};
    chained across clips. Returns (deltas, clip starts, info)."""
    dev, n, K = x.device, len(x), len(uv)
    d, starts, rms, sr, a = [], [], [], [], 0
    while a < n - 1:
        b = min(a + L - 1, n - 1); l = b - a + 1; starts.append(a)
        clip = x[a:b + 1][None]
        with torch.autocast(dev.type, torch.bfloat16, enabled=dev.type == "cuda"):
            feats = model.encode(clip)
            src = torch.arange(1, l, device=dev).repeat_interleave(K)[None]
            ref = torch.zeros_like(src) if mode == "anchor" else src - 1
            P = model.decode(feats, clip, uv.repeat(l - 1, 1)[None], src, ref)[0].float().view(l - 1, K, 3)
        R, t, s_, e = kabsch(uv_to_mm(uv, model.S)[None].expand(l - 1, K, 3), P)
        T = Rt_to_T(R, t)  # T_{f->ref}
        if mode == "anchor":  # delta_f = inv(T_{f-1->anchor}) T_{f->anchor} (the clip's pose in frame 0 cancels)
            pose = torch.cat([torch.eye(4, dtype=torch.float64, device=dev)[None], T]); T = torch.linalg.inv(pose[:-1]) @ pose[1:]
        d.append(T.float()); rms.append(e); sr.append(s_[:, 1] / s_[:, 0]); a = b
    return torch.cat(d), starts, dict(rigid_rms=torch.cat(rms).mean().item(), sv_ratio_min=torch.cat(sr).min().item())


@torch.no_grad()
def errors(Td, Tg, lm, S, starts=None, chunk=32):
    """Td [n-1,4,4] predicted deltas, Tg [n,4,4] GT T_{i->0}, lm [K,3] (0-based frame, x, y px) -> dict of floats."""
    dev, S = Td.device, S.to(Td.device)
    Td, Tg = Td.float(), Tg.float()
    Gp, Gl = accumulate(Td[None])[0], torch.linalg.inv(Tg[:-1]) @ Tg[1:]  # predicted T_{i->0}, GT deltas
    P = pixel_grid(S)
    P5 = S @ torch.tensor([[1., 640, 1, 640, 320], [1, 1, 480, 480, 240], [0.] * 5, [1.] * 5], device=dev)
    L = lm.to(dev).float(); li = L[:, 0].long()
    Lp = S @ torch.stack([L[:, 1], L[:, 2], torch.zeros_like(L[:, 0]), torch.ones_like(L[:, 0])])
    pix = lambda A, B: torch.cat([((A[i:i + chunk] - B[i:i + chunk])[:, :3] @ P).norm(dim=1).mean(1) for i in range(0, len(A), chunk)]).mean()
    lmk = lambda A, B, i: torch.einsum("kij,jk->ki", (A[i] - B[i])[:, :3], Lp).norm(dim=1).mean()
    p5 = lambda A, B: ((A - B)[:, :3] @ P5).norm(dim=1).mean(1)  # per-transform 5-point error
    r = dict(GPE=pix(Gp[1:], Tg[1:]).item(), GLE=lmk(Gp, Tg, li).item(), LPE=pix(Td, Gl).item(), LLE=lmk(Td, Gl, li - 1).item())
    for k in LAGS:
        if k < len(Tg):
            r[f"lag{k}"] = p5(torch.linalg.inv(Gp[:-k]) @ Gp[k:], torch.linalg.inv(Tg[:-k]) @ Tg[k:]).mean().item()
    if starts and len(starts) > 1:
        e = p5(Td, Gl); m = torch.zeros(len(e), dtype=torch.bool, device=dev)
        m[torch.tensor(starts[1:], device=dev)] = True  # first delta of every clip but the first
        m[torch.tensor(starts[1:], device=dev) - 1] = True  # and the last delta of the previous clip
        r["bnd"], r["int"] = e[m].mean().item(), e[~m].mean().item()
    return r


def evaluate(model, ds, L, dev="cuda", every=1, pred=predict, per_scan=False):
    """Mean metrics over every `every`-th scan of ds (a Clips(full=True) dataset); also runtime/scan.
    per_scan=True -> (mean, records) with one record per scan: name, metrics and the identity-floor GPE/LPE (acquisition pattern = name)."""
    import time
    model.eval(); rs = []; t = 0; recs = []
    for k in range(0, len(ds), every):
        x, Tg = ds[k]
        sid, name = IDX(ds.paths[k])
        lm = torch.from_numpy(h5py.File(f"{ds.paths[k].rsplit('/', 2)[0]}/landmark_{sid}.h5")[name][()])
        x = x.to(dev); torch.cuda.synchronize(); t0 = time.time()
        Td, st, *info = pred(model, x, L); torch.cuda.synchronize(); t += time.time() - t0
        rs.append({**errors(Td, Tg.to(dev), lm, ds.S, st), **(info[0] if info else {})})
        if per_scan:
            fl = errors(torch.eye(4, device=dev).expand(len(Td), 4, 4), Tg.to(dev), lm, ds.S)
            recs.append(dict(scan=f"{sid}/{name}", frames=len(x), **{k: round(v, 4) for k, v in rs[-1].items()}, GPE_identity=round(fl["GPE"], 4), LPE_identity=round(fl["LPE"], 4)))
    out = {k: sum(r[k] for r in rs if k in r) / sum(k in r for r in rs) for k in {k for r in rs for k in r}}
    out["sec_per_scan"] = t / len(rs)
    return (out, recs) if per_scan else out


if __name__ == "__main__":  # oracle model (reads the frame index from x, returns GT deltas) -> ~0 error through chaining
    import data
    ds = data.Clips(data.split()[1], full=True)
    _, Tg = ds[0]
    n = len(Tg); Gl = torch.linalg.inv(Tg[:-1]) @ Tg[1:]
    x = torch.arange(n, dtype=torch.float32)[:, None, None, None].expand(n, 1, 4, 4)  # frame i filled with value i; no GT visible

    class Oracle(torch.nn.Module):
        def forward(s, x):
            i = x[0, :, 0, 0, 0].long()
            return Gl[i[1:] - 1][None]

    sid, name = IDX(ds.paths[0])
    lm = torch.from_numpy(h5py.File(f"{data.ROOT}landmark_{sid}.h5")[name][()])
    for L in (7, 64, n + 5):  # clip lengths incl. one larger than the scan and one not dividing it
        Td, st = predict(Oracle(), x, L)
        assert Td.shape == (n - 1, 4, 4) and torch.allclose(accumulate(Td[None])[0], Tg, atol=1e-2), L
        e = errors(Td, Tg, lm, ds.S, st)
        assert max(e.values()) < 1e-2, (L, e)
    # identity deltas ("no motion") is the reference floor
    e1 = errors(torch.eye(4).expand(n - 1, 4, 4), Tg, lm, ds.S)
    assert e1["GPE"] > 10 and 0.05 < e1["LPE"] < 2 and e1["lag256"] > e1["lag1"], e1
    print("eval ok; no-motion floor:", {k: round(v, 3) for k, v in e1.items()})


@torch.no_grad()
def sensitivity(model, ds, L, dev="cuda", pred=predict, k=0):
    """Image-sensitivity control on scan k: true frames vs frames of a scan of ANOTHER subject (same length) vs frame 0 repeated.
    A model that learned from images must degrade clearly on `other`/`static`; if not, it only uses trajectory priors."""
    x, Tg = ds[k]; n = len(x)
    xo, _ = ds[(k + len(ds) // 2) % len(ds)]
    sid, name = IDX(ds.paths[k])
    lm = torch.from_numpy(h5py.File(f"{ds.paths[k].rsplit('/', 2)[0]}/landmark_{sid}.h5")[name][()])
    out = {}
    for tag, xv in dict(true=x, other=xo[torch.arange(n) % len(xo)], static=x[:1].expand_as(x)).items():
        Td, st, *_ = pred(model, xv.to(dev), L)
        out[tag] = {**{k_: round(v, 3) for k_, v in errors(Td, Tg.to(dev), lm, ds.S, st).items() if k_ in ("GPE", "LPE", "lag16", "lag64")},
                    "mean_step_mm": round(Td[:, :3, 3].norm(dim=1).mean().item(), 4)}
    return out
