"""Audit 2: (a) scales and gradients of the rotation vs translation loss components (untrained model and the A1 pilot checkpoint);
(b) can rotation be learned on a small subset? 8 train scans, pairs from frames [0,0.7), evaluated on the held-out frames [0.7,1.0) of the SAME scans.
Usage: audit_loss.py grads | subset <pool> <parts>"""
import json, os, sys, numpy as np, torch, yaml
sys.path.insert(0, os.path.dirname(__file__)); sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from train_pose import build, REPO
from usmotion4d.data import tusrec, split
from usmotion4d.models.losses import pose_loss, pose_error
from usmotion4d.geometry import se3

cfg = yaml.safe_load(open("configs/pose_base.yaml")); cfg["arm"] = "A1"
man = split.load_manifest(); tr = tusrec.scan_paths(man["train"])
sig = (cfg["loss"]["sigma_rot"], cfg["loss"]["sigma_trans"])


def batch(ds, n=64, off=0):
    b = [ds[off + i] for i in range(n)]
    return {k: (torch.stack([x[k] for x in b]) if torch.is_tensor(b[0][k]) else torch.tensor([x[k] for x in b])) for k in b[0]}


def forward(model, b):
    x = b["x"].cuda(); B = len(x); k = b["k"].cuda().float()
    f = model.encode(x.flatten(0, 1)).view(B, 2, -1, model.encoder.channels * len(model.encoder.layers))
    with torch.autocast("cuda", torch.bfloat16): o = model(f[:, 0], f[:, 1], k)
    return o, k


if sys.argv[1] == "grads":
    ds = tusrec.PairDataset(tr, cfg["offsets"], tuple(cfg["size"]), length=64, seed=1); b = batch(ds)
    out = {}
    for tag, ck in (("untrained", None), ("pilot_A1", "outputs/pose/pilot_A1_s17/last.pt")):
        model = build(cfg)
        if ck: model.load_state_dict(torch.load(ck, weights_only=False)["state"], strict=False)
        P = model.trainable_parameters(); g = {}
        for parts in ("rot", "trans", "both"):
            o, k = forward(model, b); loss, d = pose_loss(o["xi"], b["delta"].cuda(), k, *sig, parts=parts)
            gr = torch.autograd.grad(loss, P, allow_unused=True); g[parts] = torch.cat([x.flatten() for x in gr if x is not None])
        er, et = pose_error(o["xi"].detach(), b["delta"].cuda())
        nr, nt = er.norm(dim=-1) / (sig[0] * k), et.norm(dim=-1) / (sig[1] * k)
        out[tag] = dict(norm_rot_err_mean=nr.mean().item(), norm_trans_err_mean=nt.mean().item(), frac_rot_in_quadratic_zone=(nr < 1).float().mean().item(), frac_trans_in_quadratic_zone=(nt < 1).float().mean().item(),
                        grad_norm_rot=g["rot"].norm().item(), grad_norm_trans=g["trans"].norm().item(), grad_norm_both=g["both"].norm().item(), cos_rot_trans=torch.nn.functional.cosine_similarity(g["rot"], g["trans"], dim=0).item())
        print(tag, {k_: round(v, 4) for k_, v in out[tag].items()})
    json.dump(out, open("outputs/audit_loss_grads.json", "w"), indent=1)
else:
    pool, parts = sys.argv[2], sys.argv[3]
    cfg["model"]["pool"] = pool; torch.manual_seed(17); np.random.seed(17)
    sub = tr[:: len(tr) // 8][:8]
    dtr = tusrec.PairDataset(sub, cfg["offsets"], tuple(cfg["size"]), length=1500 * 32, seed=17, frac=(0, .7))
    dte = tusrec.PairDataset(sub, cfg["offsets"], tuple(cfg["size"]), length=512, seed=99, frac=(.7, 1.0))
    model = build(cfg); P = model.trainable_parameters(); opt = torch.optim.AdamW(P, 3e-4, weight_decay=.01)
    dl = torch.utils.data.DataLoader(dtr, 32, num_workers=6, drop_last=True)
    for step, b in enumerate(dl, 1):
        model.train(); o, k = forward(model, b); loss, d = pose_loss(o["xi"], b["delta"].cuda(), k, *sig, parts=parts)
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(P, 1.0); opt.step()
        if step % 250 == 0: print(step, {a: round(v, 4) for a, v in d.items()}, flush=True)
    model.eval(); res = {}
    with torch.no_grad():
        for kk in (1, 8):
            E, E0 = [], []
            for a in range(0, 512, 64):
                bb = batch(dte, 64, a); m = bb["k"] == kk
                if not m.any(): continue
                o, k = forward(model, bb); gt = bb["delta"].cuda().double()
                E.append(se3.log(se3.inverse(gt) @ o["delta"].double())[m.cuda()]); E0.append(se3.log(se3.inverse(gt))[m.cuda()])
            E, E0 = torch.cat(E), torch.cat(E0)
            res[f"k{kk}"] = dict(n=len(E), rot_axis_mrad=(E[:, :3].abs().mean(0) * 1e3).tolist(), rot_axis_mrad_identity=(E0[:, :3].abs().mean(0) * 1e3).tolist(),
                                 trans_axis_mm=E[:, 3:].abs().mean(0).tolist(), trans_axis_mm_identity=E0[:, 3:].abs().mean(0).tolist())
    json.dump(res, open(f"outputs/audit_subset_{pool}_{parts}.json", "w"), indent=1); print(pool, parts, json.dumps(res))
