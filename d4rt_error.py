"""Where does D4RT lose accuracy? No retraining; checkpoint read-only (analysis may use GT, inference path does not).
1. 3D error of the predicted points before / after the rigid (Kabsch) projection, per frame distance to the reference, anchor and adjacent mode
2. rigidity of the predicted points (pair-distance distortion, planarity, Procrustes scale, Kabsch residual) + rotation/translation error of the recovered pose
3. local relative-transform error BEFORE chaining: systematic bias vs random error, implied drift, vs the direct pose-regression arm
4. number / distribution of queried points (2x2 .. 16x16, random, central, border): pose quality with the same checkpoint
python d4rt_error.py [--ckpt ...] -> runs/d4rt_error.json"""
import argparse, json, math, torch
import data, eval as ev, pretrained as pt
from geom import query_gt, uv_to_mm, kabsch, Rt_to_T, load_calib

a = argparse.ArgumentParser()
a.add_argument("--ckpt", default="/raid/DATASETS/Echo3D_weights/matrix/videomae_d4rt.pt"); a.add_argument("--pose_ckpt", default="/raid/DATASETS/Echo3D_weights/matrix/videomae_pose.pt")
a.add_argument("--n_scans", type=int, default=24); a.add_argument("--L", type=int, default=64); a.add_argument("--out", default="runs/d4rt_error.json")
a = a.parse_args(); dev = "cuda"
tr, va = data.split(5); val = data.Clips(va, full=True); S, _ = load_calib(data.ROOT + "calib_matrix.csv")
r3 = lambda v: round(float(v), 4)
ks = list(range(0, len(val), 5))[:a.n_scans]  # 5 is coprime with 24: all scan types


def load(head, ck):
    m = pt.build("videomae", head, val.S).to(dev); m.load_state_dict(torch.load(ck, map_location=dev)); return m.eval()


d4 = load("d4rt", a.ckpt)
vee = lambda R: .5 * torch.stack([R[:, 2, 1] - R[:, 1, 2], R[:, 0, 2] - R[:, 2, 0], R[:, 1, 0] - R[:, 0, 1]], -1)
par = lambda T: torch.cat([T[:, :3, 3], vee(T[:, :3, :3])], -1)  # tx ty tz (mm), wx wy wz (rad)
uvg = lambda K: torch.stack(torch.meshgrid((torch.arange(K, device=dev) + .5) / K, (torch.arange(K, device=dev) + .5) / K, indexing="ij")[::-1], -1).view(-1, 2)
cen = torch.tensor([320 * float(S[0, 0]), 240 * float(S[1, 1]), 0, 1.], device=dev, dtype=torch.float64)
BINS = [("1", 1, 1), ("2-3", 2, 3), ("4-15", 4, 15), ("16-31", 16, 31), ("32-63", 32, 63)]


@torch.no_grad()
def clip_analysis(mode, uv):
    """rows: one per (clip, frame): lag, raw/after/no-motion point error, kabsch rms, pair distortion, planarity, scale, rot err (deg), centre translation err (mm)
    + local delta error vectors (pred - GT) of the steps, per scan."""
    rows, derr, gstep = [], [], []
    K = len(uv); p0 = uv_to_mm(uv, d4.S)
    for k in ks:
        x, Tg = val[k]; x, Tg = x.to(dev), Tg.to(dev).double(); n = len(x); st = 0; perscan = []
        while st < n - 1:
            b = min(st + a.L - 1, n - 1); l = b - st + 1; clip = x[st:b + 1][None]
            f = torch.arange(1, l, device=dev); src = f.repeat_interleave(K)[None]; ref = torch.zeros_like(src) if mode == "anchor" else src - 1
            with torch.autocast("cuda", torch.bfloat16):
                P = d4.decode(d4.encode(clip), clip, uv.repeat(l - 1, 1)[None], src, ref)[0].float().view(l - 1, K, 3)
            G = query_gt(Tg[st:b + 1][None].float(), d4.S, uv.repeat(l - 1, 1)[None], src, ref)[0].view(l - 1, K, 3)
            R, t, sv, rms = kabsch(p0[None].expand(l - 1, K, 3), P); Tp = Rt_to_T(R, t)
            Pr = (p0.double()[None] @ R.transpose(-1, -2) + t[:, None]).float()
            refi = torch.zeros_like(f) if mode == "anchor" else f - 1
            Tgt = torch.linalg.inv(Tg[st + refi]) @ Tg[st + f]  # GT T_{f->ref}
            cos = ((R.transpose(-1, -2) @ Tgt[:, :3, :3]).diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2
            rot = torch.rad2deg(torch.acos(cos.clamp(-1, 1))); tr_ = ((Tp - Tgt) @ cen).norm(dim=-1)
            d_p = torch.cdist(P, P); d_s = torch.cdist(p0, p0)[None]; m = d_s > 1e-6  # pair-distance distortion
            dist = ((d_p - d_s).abs() / d_s.clamp_min(1e-6) * m).sum((1, 2)) / m.sum((1, 2))
            Pc = P - P.mean(1, keepdim=True); plan = torch.linalg.svdvals(Pc.double())[:, -1] / math.sqrt(K)  # out-of-plane rms (source is planar)
            pc = (p0 - p0.mean(0)).double(); scale = ((Pc.double() * (pc @ R.transpose(-1, -2))).sum((1, 2)) / (pc ** 2).sum()).float()
            lag = f if mode == "anchor" else torch.ones_like(f)
            rows.append(torch.stack([lag.float(), (P - G).norm(dim=-1).mean(-1), (Pr - G).norm(dim=-1).mean(-1), (p0[None] - G).norm(dim=-1).mean(-1), rms.float(), dist, plan.float(), scale, rot.float(), tr_.float()], -1))
            if mode == "anchor":  # local delta from anchor poses
                pose = torch.cat([torch.eye(4, device=dev, dtype=torch.float64)[None], Tp]); dl = torch.linalg.inv(pose[:-1]) @ pose[1:]
            else:
                dl = Tp
            gl = torch.linalg.inv(Tg[st + f - 1]) @ Tg[st + f]
            perscan.append(torch.cat([par(dl.float()) - par(gl.float())], 0)); gstep.append(par(gl.float()))
            st = b
        derr.append(torch.cat(perscan))
    return torch.cat(rows), derr, torch.cat(gstep)


def summarize(R):
    out = {}
    for name, lo, hi in BINS:
        m = (R[:, 0] >= lo) & (R[:, 0] <= hi)
        if m.sum() < 5: continue
        raw, aft, nom = R[m, 1], R[m, 2], R[m, 3]
        out[f"lag {name}"] = dict(n_frames=int(m.sum()), point_err_raw_mm=r3(raw.mean()), point_err_after_kabsch_mm=r3(aft.mean()), no_motion_err_mm=r3(nom.mean()),
                                  share_of_sq_err_left_after_rigid=r3((aft ** 2).mean() / (raw ** 2).mean()), kabsch_rms_mm=r3(R[m, 4].mean()),
                                  pair_dist_distortion_pct=r3(100 * R[m, 5].mean()), out_of_plane_rms_mm=r3(R[m, 6].mean()), scale_vs_1_pct=r3(100 * (R[m, 7].mean() - 1)),
                                  pose_rot_err_deg=r3(R[m, 8].mean()), pose_centre_trans_err_mm=r3(R[m, 9].mean()))
    return out


def bias_std(derr, gstep):
    """local step error vector (pred - GT): pooled bias/std, consistency of the bias across scans, implied drift over N=64 steps."""
    E = torch.cat(derr); out = {}
    for i, c in enumerate(["tx_mm", "ty_mm", "tz_mm", "wx_rad", "wy_rad", "wz_rad"]):
        per_scan = torch.stack([e[:, i].mean() for e in derr]); bias = E[:, i].mean(); sd = E[:, i].std()
        out[c] = dict(bias=r3(bias), std=r3(sd), bias_share_of_mse=r3(bias ** 2 / (bias ** 2 + sd ** 2)), scans_with_same_sign_bias=r3((torch.sign(per_scan) == torch.sign(bias)).float().mean()),
                      bias_over_64_steps=r3(64 * bias), random_walk_over_64_steps=r3(8 * sd), gt_step_mean_abs=r3(gstep[:, i].abs().mean()))
    return out


res = {"scans": len(ks), "ckpt": a.ckpt}
# ---- 1+2: 8x8 grid, both modes
for mode in ("anchor", "adjacent"):
    R, derr, gs = clip_analysis(mode, uvg(8)); res[f"grid8x8_{mode}"] = dict(by_lag=summarize(R), local_step_error_before_chaining=bias_std(derr, gs))
    print(mode, json.dumps({k: (v["point_err_raw_mm"], v["point_err_after_kabsch_mm"], v["pose_rot_err_deg"], v["pose_centre_trans_err_mm"]) for k, v in res[f"grid8x8_{mode}"]["by_lag"].items()}), flush=True)
# ---- 3: same control on the direct pose-regression arm
pose = load("pose", a.pose_ckpt); derr, gs = [], []
with torch.no_grad():
    for k in ks:
        x, Tg = val[k]; Td, _ = ev.predict(pose, x.to(dev), a.L); Tg = Tg.to(dev)
        derr.append(par(Td.float()) - par(torch.linalg.inv(Tg[:-1]) @ Tg[1:])); gs.append(par(torch.linalg.inv(Tg[:-1]) @ Tg[1:]))
res["pose_arm_local_step_error_before_chaining"] = bias_std(derr, torch.cat(gs)); del pose
# ---- 4: number / distribution of the queried points (full chained inference, whole scans)
g = torch.Generator(device=dev); g.manual_seed(0)
ang = torch.arange(64, device=dev) / 64 * 4  # perimeter parameter in [0,4)
border = torch.stack([torch.where(ang < 1, .05 + .9 * ang, torch.where(ang < 2, torch.full_like(ang, .95), torch.where(ang < 3, .95 - .9 * (ang - 2), torch.full_like(ang, .05)))),
                      torch.where(ang < 1, torch.full_like(ang, .05), torch.where(ang < 2, .05 + .9 * (ang - 1), torch.where(ang < 3, torch.full_like(ang, .95), .95 - .9 * (ang - 3))))], -1)
CFG = {"grid 2x2": uvg(2), "grid 4x4": uvg(4), "grid 8x8": uvg(8), "grid 16x16": uvg(16), "random 64 (uniform)": torch.rand(64, 2, generator=g, device=dev),
       "central 8x8 (u,v in .25-.75)": .25 + .5 * uvg(8), "border ring 64": border}
res["points_study"] = {}
for name, uv in CFG.items():
    res["points_study"][name] = {}
    for mode in ("anchor", "adjacent"):
        rs, t = [], 0
        for k in ks:
            x, Tg = val[k]; sid, nm = ev.IDX(val.paths[k]); lm = torch.from_numpy(__import__("h5py").File(f"{data.ROOT}landmark_{sid}.h5")[nm][()])
            Td, st, info = ev.predict_uv(d4, x.to(dev), a.L, mode, uv); rs.append({**ev.errors(Td, Tg.to(dev), lm, val.S, st), **info})
        res["points_study"][name][mode] = {k: r3(sum(r[k] for r in rs) / len(rs)) for k in ("GPE", "GLE", "LPE", "LLE", "lag16", "lag64", "rigid_rms")}
    print(name, {m: (v["GPE"], v["LPE"], v["rigid_rms"]) for m, v in res["points_study"][name].items()}, flush=True)
json.dump(res, open(a.out, "w"), indent=1)
