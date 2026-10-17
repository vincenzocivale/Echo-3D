"""Diagnosis of motion learning for a trained arm, on train AND val scans (read-only on checkpoints).
python diagnose.py --arm pose|d4rt|pair --ckpt runs/x.pt [--window -1] --name x     -> runs/diag_x.json + printed summary
Sections: deltas (distribution/collapse/corr/skill vs identity), lags (model vs identity), queries (D4RT: identity vs non-identity,
by |dt|, elevational distance, FOV overlap, image swap), grads (per module norm + cross-batch cosine), sensitivity, overlap (pairs)."""
import argparse, json, math, random, h5py, torch
import data, eval as ev, pairwise
from geom import query_gt, uv_to_mm, load_calib, kabsch, sample_q as _sq, H, W
from model import Net, D4RT, accumulate

dev = "cuda"
a = argparse.ArgumentParser()
a.add_argument("--arm", required=True, choices=["pose", "d4rt", "pair"]); a.add_argument("--enc", default="vit", choices=["vit", "usfm", "videomae"]); a.add_argument("--ckpt", required=True)
a.add_argument("--window", type=int, default=-1); a.add_argument("--name", required=True)
a.add_argument("--pair_norm", action="store_true"); a.add_argument("--grad_bs", type=int, default=4); a.add_argument("--n_val", type=int, default=5); a.add_argument("--L", type=int, default=64); a.add_argument("--n_scans", type=int, default=24)
a = a.parse_args()
torch.manual_seed(0)
tr, va = data.split(a.n_val)
S, _ = load_calib(data.ROOT + "calib_matrix.csv")
w = None if a.window < 0 else a.window
import pretrained as pt
model = (pt.build(a.enc, a.arm, S) if a.enc != "vit" and a.arm != "pair" else
         {"pose": lambda: Net(window=w), "d4rt": lambda: D4RT(Net(window=w), S, dim=256, heads=8), "pair": lambda: pairwise.Pair(norm=a.pair_norm)}[a.arm]()).to(dev)
model.load_state_dict(torch.load(a.ckpt, map_location=dev)); model.eval()
PRED = {"pose": ev.predict, "pair": pairwise.predict_pair,
        "d4rt": lambda m, x, L: ev.predict_q(m, x, L, "anchor")}[a.arm]
P5 = S.to(dev) @ torch.tensor([[1., 640, 1, 640, 320], [1, 1, 480, 480, 240], [0.] * 5, [1.] * 5], device=dev)
SPLITS = {"train": data.Clips(tr, full=True), "val": data.Clips(va, full=True)}
r3 = lambda v: round(float(v), 4)


def lm_of(ds, k):
    sid, name = ev.IDX(ds.paths[k])
    return torch.from_numpy(h5py.File(f"{data.ROOT}landmark_{sid}.h5")[name][()])


def idx_of(ds, n):  # 24 scans with stride coprime to 24 (covers all scan types)
    st = max(1, len(ds) // n)
    while math.gcd(st, 24) != 1: st += 1
    return list(range(0, len(ds), st))[:n]


def vee(R): return .5 * torch.stack([R[:, 2, 1] - R[:, 1, 2], R[:, 0, 2] - R[:, 2, 0], R[:, 1, 0] - R[:, 0, 1]], -1)
def dparams(T): return torch.cat([T[:, :3, 3], vee(T[:, :3, :3])], -1)  # tx ty tz wx wy wz (mm, rad)
def corr(x, y): return torch.corrcoef(torch.stack([x, y]))[0, 1].item() if x.std() > 1e-12 and y.std() > 1e-12 else float("nan")


@torch.no_grad()
def scans(split):
    """per scan: predicted deltas, GT T_{i->0}, landmarks"""
    ds = SPLITS[split]; out = []
    for k in idx_of(ds, a.n_scans):
        x, Tg = ds[k]; Td, st, *_ = PRED(model, x.to(dev), a.L)
        out.append((Td, Tg.to(dev), lm_of(ds, k)))
    return out


def deltas_lags(sc):
    GP = torch.cat([dparams(torch.linalg.inv(Tg[:-1]) @ Tg[1:]) for _, Tg, _ in sc]).cpu(); PP = torch.cat([dparams(Td.float()) for Td, _, _ in sc]).cpu()
    comp = {}
    for i, n in enumerate(["tx", "ty", "tz", "wx", "wy", "wz"]):
        comp[n] = dict(gt_mean=r3(GP[:, i].mean()), gt_std=r3(GP[:, i].std()), pred_mean=r3(PP[:, i].mean()), pred_std=r3(PP[:, i].std()),
                       std_ratio=r3(PP[:, i].std() / GP[:, i].std()), corr=r3(corr(PP[:, i], GP[:, i])),
                       skill_vs_identity=r3(1 - ((PP[:, i] - GP[:, i]) ** 2).sum() / (GP[:, i] ** 2).sum()))
    # translation-vector cosine between predicted and GT steps
    cos = torch.nn.functional.cosine_similarity(PP[:, :3], GP[:, :3], dim=-1)
    lag = {}
    for Td, Tg, lm in sc:
        m = ev.errors(Td, Tg, lm, S); i = ev.errors(torch.eye(4, device=dev).expand(len(Td), 4, 4), Tg, lm, S)
        for k in m:
            if k.startswith(("lag", "GPE", "GLE", "LPE", "LLE")):
                lag.setdefault(k, []).append((m[k], i[k]))
    lags = {k: dict(model=r3(sum(x for x, _ in v) / len(v)), identity=r3(sum(y for _, y in v) / len(v)),
                    skill=r3(1 - sum(x for x, _ in v) / sum(y for _, y in v))) for k, v in lag.items()}
    return dict(components=comp, step_cosine_mean=r3(cos.mean()), step_cosine_median=r3(cos.median()), metrics_vs_identity=lags)


# ---- D4RT queries -------------------------------------------------------------------------------------------------
DT = [("id", 0, 0), ("1", 1, 1), ("2-3", 2, 3), ("4-15", 4, 15), ("16-31", 16, 31), ("32-63", 32, 63)]
ZB = [("0-1mm", 0, 1), ("1-3", 1, 3), ("3-10", 3, 10), ("10-30", 10, 30), (">30", 30, 1e9)]


def sample_q(B, Q, L, g, id_frac=.1): return _sq(B, Q, L, g, dev, id_frac)


def binstats(m, e_mod, e_id, e_sw, off_p, off_g):
    n = int(m.sum())
    if n < 5: return None
    op, og = off_p[m], off_g[m]
    triv = e_id[m].mean() < 1e-6  # identity queries: the "no motion" answer is exact, skill/ratios are undefined
    return dict(n=n, err_model=r3(e_mod[m].mean()), err_identity=r3(e_id[m].mean()), skill=None if triv else r3(1 - e_mod[m].mean() / e_id[m].mean()),
                err_image_swap=r3(e_sw[m].mean()), corr_offset=None if triv else r3(corr(op.flatten(), og.flatten())),
                std_ratio=None if triv else r3(op.std() / max(og.std(), 1e-9)), pred_offset_std_mm=r3(op.std()))


@torch.no_grad()
def queries(split, n_clips=24, Q=2048):
    ds = data.Clips(tr if split == "train" else va, a.L); random.seed(123)
    g = torch.Generator(device=dev); g.manual_seed(7); R = []
    ks = list(range(0, len(ds), max(1, len(ds) // n_clips)))[:n_clips]; xs, Ts = zip(*[ds[k] for k in ks])
    X, T = torch.stack(xs).to(dev), torch.stack(Ts).to(dev)
    for i in range(len(ks)):
        uv, src, ref = sample_q(1, Q, a.L, g)
        with torch.autocast("cuda", torch.bfloat16):
            P = model(X[i:i + 1], uv, src, ref).float(); Psw = model(X[(i + len(ks) // 2) % len(ks)][None], uv, src, ref).float()  # other clip's images
        gt = query_gt(T[i:i + 1], model.S, uv, src, ref); p0 = uv_to_mm(uv, model.S)
        R.append([t[0] for t in ((ref - src).abs(), (P - gt).norm(dim=-1), (p0 - gt).norm(dim=-1), (Psw - gt).norm(dim=-1), P - p0, gt - p0, gt)])
    dt, em, ei, es, op, og, gt = (torch.cat([r[j] for r in R]) for j in range(7))
    fov = (gt[:, 0] >= 0) & (gt[:, 0] <= W * float(S[0, 0])) & (gt[:, 1] >= 0) & (gt[:, 1] <= H * float(S[1, 1]))
    z = gt[:, 2].abs()
    out = dict(by_dt={n: binstats((dt >= lo) & (dt <= hi), em, ei, es, op, og) for n, lo, hi in DT},
               by_elevational_distance={n: binstats((z >= lo) & (z < hi) & (dt > 0), em, ei, es, op, og) for n, lo, hi in ZB},
               by_fov={"inside_ref_fov": binstats(fov & (dt > 0), em, ei, es, op, og), "outside_ref_fov": binstats(~fov & (dt > 0), em, ei, es, op, og)},
               non_identity_all=binstats(dt > 0, em, ei, es, op, og), identity_all=binstats(dt == 0, em, ei, es, op, og))
    return out


# ---- gradients ----------------------------------------------------------------------------------------------------
def gkey(n):
    p = n.split(".")
    for t in ("layers", "blocks"):
        if t in p: return ".".join(p[:p.index(t) + 2])
    if p[0] == "f": return ".".join(p[:2])
    return ".".join(p[:2]) if p[0] in ("enc", "dec") else p[0]


def loss_of(x, Tg, g):
    P = P5
    if a.arm == "pose":
        Gl = torch.linalg.inv(Tg[:, :-1]) @ Tg[:, 1:]; Td = model(x); pt = lambda A, B: ((A - B)[..., :3, :] @ P).norm(dim=-2).mean()
        return pt(Td, Gl) + .05 * pt(accumulate(Td)[:, 1:], Tg[:, 1:])
    if a.arm == "d4rt":
        uv, src, ref = sample_q(len(x), 512, a.L, g)
        return (model(x, uv, src, ref) - query_gt(Tg, model.S, uv, src, ref)).abs().mean()
    B, L = x.shape[:2]; i = torch.randint(1, L, (B, 16), generator=g, device=dev); b = torch.arange(B, device=dev)[:, None]
    T = model(torch.cat([x[b, i - 1], x[b, i]], 2).flatten(0, 1)); gt = (torch.linalg.inv(Tg[b, i - 1]) @ Tg[b, i]).flatten(0, 1)
    return ((T - gt)[:, :3, :] @ P).norm(dim=-2).mean()


def grads(n_batches=8):
    ds = data.Clips(tr, a.L); g = torch.Generator(device=dev); g.manual_seed(3)
    dl = torch.utils.data.DataLoader(ds, a.grad_bs, shuffle=True, num_workers=4, generator=torch.Generator().manual_seed(0), drop_last=True)
    params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    per, flat = {}, {"encoder": [], "decoder_or_head": []}
    for bi, (x, Tg) in enumerate(dl):
        if bi == n_batches: break
        loss = loss_of(x.to(dev), Tg.to(dev), g)
        gr = torch.autograd.grad(loss, [p for _, p in params], allow_unused=True)
        tmp = {"encoder": [], "decoder_or_head": []}
        for (n, p), gg in zip(params, gr):
            gg = torch.zeros_like(p) if gg is None else gg
            per.setdefault(gkey(n), []).append(gg.float().pow(2).sum().item())
            tmp["decoder_or_head" if gkey(n).startswith(("dec", "head")) else "encoder"].append(gg.flatten().float())
        for k in tmp: flat[k].append(torch.cat(tmp[k]))
    wn, cnt = {}, {}
    for n, p in params: wn[gkey(n)] = wn.get(gkey(n), 0) + p.float().pow(2).sum().item(); cnt[gkey(n)] = cnt.get(gkey(n), 0) + p.numel()
    out = {k: dict(params_m=r3(cnt[k] / 1e6), grad_norm=r3(sum(math.sqrt(x) for x in v) / len(v)), weight_norm=r3(math.sqrt(wn[k]))) for k, v in per.items()}
    cos = {}
    for k, v in flat.items():
        if v:
            V = torch.nn.functional.normalize(torch.stack(v), dim=1); C = V @ V.T; cos[k] = r3((C.sum() - len(v)) / (len(v) * (len(v) - 1)))
    return dict(per_module=out, mean_pairwise_cosine_across_batches=cos)


# ---- overlap (pairs from predicted global poses) ------------------------------------------------------------------
@torch.no_grad()
def overlap(sc, lags=(4, 16, 64)):
    gv, gu = torch.meshgrid((torch.arange(8, device=dev) + .5) / 8, (torch.arange(8, device=dev) + .5) / 8, indexing="ij")
    Pg = uv_to_mm(torch.stack([gu.flatten(), gv.flatten()], -1), S.to(dev)); Ph = torch.cat([Pg, torch.ones(64, 1, device=dev)], -1).T  # [4,64]
    rows = []
    for Td, Tg, _ in sc:
        Gp = accumulate(Td.float()[None])[0]
        for k in lags:
            i = torch.arange(0, len(Tg) - k, 5, device=dev)
            Rg = torch.linalg.inv(Tg[i]) @ Tg[i + k]; Rp = torch.linalg.inv(Gp[i]) @ Gp[i + k]
            pts = (Rg @ Ph)[:, :3]  # [n,3,64] where the source grid lands in the reference plane
            em = ((Rp - Rg)[:, :3] @ Ph).norm(dim=1).mean(-1); ei = ((Rg - torch.eye(4, device=dev))[:, :3] @ Ph).norm(dim=1).mean(-1)
            dz = pts[:, 2].abs().mean(-1); ov = ((pts[:, 0] >= 0) & (pts[:, 0] <= W * float(S[0, 0])) & (pts[:, 1] >= 0) & (pts[:, 1] <= H * float(S[1, 1]))).float().mean(-1)
            rows.append(torch.stack([torch.full_like(dz, k), dz, ov, em, ei], -1))
    R = torch.cat(rows); out = {}
    for name, col, bins in (("elevational_distance_mm", 1, [(0, 1), (1, 3), (3, 10), (10, 30), (30, 1e9)]), ("inplane_overlap_fraction", 2, [(.75, 1.01), (.25, .75), (0, .25)])):
        out[name] = {}
        for lo, hi in bins:
            m = (R[:, col] >= lo) & (R[:, col] < hi)
            if m.sum() >= 5: out[name][f"{lo}-{hi}"] = dict(n=int(m.sum()), err_model=r3(R[m, 3].mean()), err_identity=r3(R[m, 4].mean()), skill=r3(1 - R[m, 3].mean() / R[m, 4].mean()))
    return out


@torch.no_grad()
def feature_sensitivity(n=16, Q=512):
    """Encoder-feature change vs output change when the images change (other scan / static / time-reversed). D4RT: output change split into
    feature-only (decoder sees changed features, original patches) and patch-only (original features, changed 9x9 patches)."""
    enc = getattr(model, "encode", None) or getattr(getattr(model, "enc", None), "encode", None)
    if enc is None: return None
    ds = data.Clips(va, a.L); ks = list(range(0, len(ds), max(1, len(ds) // n)))[:n]
    X = torch.stack([ds[k][0] for k in ks]).to(dev); Tg = torch.stack([ds[k][1] for k in ks]).to(dev)
    g = torch.Generator(device=dev); g.manual_seed(7); uv, src, ref = sample_q(len(X), Q, a.L, g)
    nz = (ref != src); gt = query_gt(Tg, model.S, uv, src, ref) if a.arm == "d4rt" else None
    out = {}
    with torch.autocast("cuda", torch.bfloat16):
        z0 = enc(X).float().flatten(1, 2)
        P0 = model.decode(model.encode(X), X, uv, src, ref).float() if a.arm == "d4rt" else None
        D0 = model(X).float() if a.arm == "pose" else None
        for tag, Xv in dict(other_scan=X.roll(1, 0), static=X[:, :1].expand_as(X).contiguous(), time_reversed=X.flip(1)).items():
            z = enc(Xv).float().flatten(1, 2); r = dict(enc_rel_change=r3(((z - z0).flatten(1).norm(dim=1) / z0.flatten(1).norm(dim=1)).mean()),
                                                        enc_token_cosine=r3(torch.nn.functional.cosine_similarity(z, z0, dim=-1).mean()))
            if a.arm == "d4rt":
                p0 = uv_to_mm(uv, model.S); mv = (gt - p0).norm(dim=-1)[nz].mean()
                Pf = model.decode(model.encode(Xv), X, uv, src, ref).float(); Pp = model.decode(model.encode(X), Xv, uv, src, ref).float(); Pa = model(Xv, uv, src, ref).float()
                ch = lambda Pv: (Pv - P0).norm(dim=-1)[nz].mean()
                r.update(out_change_mm_feature_only=r3(ch(Pf)), out_change_mm_patch_only=r3(ch(Pp)), out_change_mm_all=r3(ch(Pa)), gt_motion_mm=r3(mv),
                         out_change_over_gt_motion=r3(ch(Pa) / mv))
            if a.arm == "pose": r.update(out_change_mm=r3(((model(Xv).float() - D0)[..., :3, :] @ P5).norm(dim=-2).mean()))
            out[tag] = r
    return out


@torch.no_grad()
def kabsch_quality(n=16, K=8):
    """D4RT anchor mode: error of the raw predicted 3D points vs after the rigid (Kabsch) projection, per frame distance to the anchor; rigid residual."""
    ds = data.Clips(va, a.L); ks = list(range(0, len(ds), max(1, len(ds) // n)))[:n]; rows = []
    gv, gu = torch.meshgrid((torch.arange(K, device=dev) + .5) / K, (torch.arange(K, device=dev) + .5) / K, indexing="ij"); uvg = torch.stack([gu.flatten(), gv.flatten()], -1)
    for k in ks:
        x, Tg = ds[k]; x, Tg = x[None].to(dev), Tg[None].to(dev); f = torch.arange(1, a.L, device=dev)
        uv = uvg.repeat(a.L - 1, 1)[None]; src = f.repeat_interleave(K * K)[None]; ref = torch.zeros_like(src)
        with torch.autocast("cuda", torch.bfloat16): P = model(x, uv, src, ref).float().view(a.L - 1, K * K, 3)
        gt = query_gt(Tg, model.S, uv, src, ref).view(a.L - 1, K * K, 3); p0 = uv_to_mm(uvg, model.S)[None].expand(a.L - 1, -1, -1)
        R, t, sv, rms = kabsch(p0, P); Pr = (p0.double() @ R.transpose(-1, -2) + t[:, None]).float()
        rows.append(torch.stack([f.float(), (P - gt).norm(dim=-1).mean(-1), (Pr - gt).norm(dim=-1).mean(-1), (p0 - gt).norm(dim=-1).mean(-1), rms.float(), (sv[:, 1] / sv[:, 0]).float()], -1))
    R = torch.cat(rows); out = {}
    for lo, hi in ((1, 3), (4, 15), (16, 63)):
        m = (R[:, 0] >= lo) & (R[:, 0] <= hi); out[f"frames {lo}-{hi}"] = dict(raw_point_err_mm=r3(R[m, 1].mean()), after_rigid_projection_mm=r3(R[m, 2].mean()), no_motion_err_mm=r3(R[m, 3].mean()),
                                                                              rigid_rms_mm=r3(R[m, 4].mean()), sv_ratio_mean=r3(R[m, 5].mean()), sv_ratio_min=r3(R[m, 5].min()))
    return out


res = dict(arm=a.arm, enc=a.enc, ckpt=a.ckpt, window=a.window)
for split in ("train", "val"):
    sc = scans(split)
    res[split] = dict(deltas_and_lags=deltas_lags(sc), overlap=overlap(sc))
    if a.arm == "d4rt": res[split]["queries"] = queries(split)
    res[split]["sensitivity"] = {k: ev.sensitivity(model, SPLITS[split], a.L, dev, PRED, k) for k in (0, len(SPLITS[split]) // 2 + 1)}
res["feature_sensitivity"] = feature_sensitivity()
if a.arm == "d4rt":
    res["kabsch_quality"] = kabsch_quality()
    PRED = lambda m, x, L: ev.predict_q(m, x, L, "adjacent")  # secondary mode, reported next to the primary (anchor) one; never used for selection
    res["val_adjacent_mode"] = deltas_lags(scans("val"))
res["grads"] = grads()
json.dump(res, open(f"runs/diag_{a.name}.json", "w"), indent=1)
for s in ("train", "val"):
    d = res[s]["deltas_and_lags"]
    print(s, "std_ratio(t)", [d["components"][c]["std_ratio"] for c in ("tx", "ty", "tz")], "corr(t)", [d["components"][c]["corr"] for c in ("tx", "ty", "tz")],
          "GPE model/id", d["metrics_vs_identity"]["GPE"]["model"], d["metrics_vs_identity"]["GPE"]["identity"], "LPE", d["metrics_vs_identity"]["LPE"]["model"], d["metrics_vs_identity"]["LPE"]["identity"])
    if a.arm == "d4rt": print("  queries identity:", res[s]["queries"]["identity_all"], "\n  queries non-identity:", res[s]["queries"]["non_identity_all"])
print("grad cosine:", res["grads"]["mean_pairwise_cosine_across_batches"])
