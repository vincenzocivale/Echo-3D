"""2x2 matrix: encoder {usfm, videomae} x head {pose, d4rt}; pretrained encoder, full fine-tuning.
python train_m.py --enc usfm --head d4rt --name usfm_d4rt
Same seed/data stream/eval for all four arms. Losses unchanged from the pilots: pose = point loss local + 0.05*global (train.py);
d4rt = L1 on 3D points only (train_q.py, same query sampler). Optimiser: AdamW (wd 0.05), lr_enc for the pretrained encoder, lr_new for everything else,
OneCycleLR (5% linear warm-up from max/25, cosine decay), grad-clip 1.0, bf16 autocast. Checkpoints go to /raid (small /home)."""
import argparse, json, math, random, time, torch
import data, eval as ev, pretrained as pt
from geom import query_gt, sample_q
from model import accumulate

a = argparse.ArgumentParser()
a.add_argument("--enc", required=True, choices=["usfm", "videomae"]); a.add_argument("--head", required=True, choices=["pose", "d4rt"]); a.add_argument("--name", required=True)
a.add_argument("--L", type=int, default=64); a.add_argument("--bs", type=int, default=8); a.add_argument("--steps", type=int, default=3000)
a.add_argument("--lr_enc", type=float, default=2e-5); a.add_argument("--lr_new", type=float, default=1e-4); a.add_argument("--lam", type=float, default=0.05)
a.add_argument("--Q", type=int, default=512); a.add_argument("--id_frac", type=float, default=0.1)
a.add_argument("--n_val", type=int, default=5); a.add_argument("--eval_every", type=int, default=500); a.add_argument("--log_every", type=int, default=50)
a = a.parse_args(); dev = "cuda"
CK = "/raid/DATASETS/Echo3D_weights/matrix/"

tr, va = data.split(a.n_val)
val = data.Clips(va, full=True)
model = pt.build(a.enc, a.head, val.S).to(dev)
pre = {id(p) for p in pt.pretrained_params(model)}
groups = [dict(params=[p for p in model.parameters() if id(p) in pre], lr=a.lr_enc), dict(params=[p for p in model.parameters() if id(p) not in pre], lr=a.lr_new)]
opt = torch.optim.AdamW(groups, weight_decay=0.05)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, [a.lr_enc, a.lr_new], total_steps=a.steps, pct_start=0.05)
torch.manual_seed(0)  # same clip stream in all four arms (encoder construction consumed a different amount of RNG)
dl = torch.utils.data.DataLoader(data.Clips(tr, a.L), a.bs, shuffle=True, num_workers=8, drop_last=True, persistent_workers=True)
P = val.S.to(dev) @ torch.tensor([[1., 640, 1, 640, 320], [1, 1, 480, 480, 240], [0.] * 5, [1.] * 5], device=dev)
pt_loss = lambda A, B: ((A - B)[..., :3, :] @ P).norm(dim=-2).mean()
gen = torch.Generator(device=dev); gen.manual_seed(1234)  # query sampler: own stream
BINS = [(0, 0), (1, 3), (4, 15), (16, a.L)]


def coarse(n):  # parameter name -> reporting group
    p = n.split(".")
    if "backbone" in p:
        return "enc_patch_embed" if "patch_embed" in p else (f"enc_blocks_{int(p[p.index('blocks') + 1]) // 4 * 4}-{int(p[p.index('blocks') + 1]) // 4 * 4 + 3}" if "blocks" in p else "enc_other")
    if p[0] == "enc" and a.enc == "videomae":
        return "enc_patch_embed" if p[1] == "embed" else (f"enc_blocks_{int(p[2]) // 4 * 4}-{int(p[2]) // 4 * 4 + 3}" if p[1] == "blocks" else "enc_norm")
    if p[:2] == ["enc", "enc"]: return "new_temporal"
    if p[0] == "enc": return "new_proj_pos"
    return "dec_" + p[1] if p[0] == "dec" else "pose_head"


def losses(x, Tg):
    if a.head == "pose":
        Gl = torch.linalg.inv(Tg[:, :-1]) @ Tg[:, 1:]
        with torch.autocast("cuda", torch.bfloat16): Td = model(x)
        ll, lg = pt_loss(Td, Gl), pt_loss(accumulate(Td)[:, 1:], Tg[:, 1:])
        return ll + a.lam * lg, dict(loss_local=ll.item(), loss_global=lg.item())
    uv, src, ref = sample_q(len(x), a.Q, a.L, gen, dev, a.id_frac)
    gt = query_gt(Tg, model.S, uv, src, ref)  # training-only use of the tracked poses
    with torch.autocast("cuda", torch.bfloat16): Pp = model(x, uv, src, ref)
    err = (Pp.float() - gt).abs(); d = (ref - src).abs(); e = err.mean(-1)
    return err.mean(), dict(loss_l1_mm=err.mean().item(), **{f"l1_dt{lo}-{hi}": round(e[(d >= lo) & (d <= hi)].mean().item(), 4) for lo, hi in BINS if ((d >= lo) & (d <= hi)).any()})


@torch.no_grad()
def query_eval(n_clips=24, Q=2048):
    model.eval(); vc = data.Clips(va, a.L); random.seed(123); g = torch.Generator(device=dev); g.manual_seed(7); E, D = [], []
    for k in list(range(0, len(vc), max(1, len(vc) // n_clips)))[:n_clips]:
        x, Tg = vc[k]; x, Tg = x[None].to(dev), Tg[None].to(dev); uv, src, ref = sample_q(1, Q, a.L, g, dev)
        with torch.autocast("cuda", torch.bfloat16): Pp = model(x, uv, src, ref)
        E.append((Pp.float() - query_gt(Tg, model.S, uv, src, ref)).norm(dim=-1)); D.append((ref - src).abs())
    E, D = torch.cat(E).flatten(), torch.cat(D).flatten()
    return {f"dt{lo}-{hi}": round(E[(D >= lo) & (D <= hi)].mean().item(), 3) for lo, hi in BINS}


@torch.no_grad()
def img_sens(n=8):
    """Mean 3D distance (mm) between outputs on a clip and on a clip of ANOTHER scan (same queries): is the output image-dependent? 0 = ignores the images."""
    model.eval(); vc = data.Clips(va, a.L); g = torch.Generator(device=dev); g.manual_seed(7)
    xs = torch.stack([vc[k][0] for k in list(range(0, len(vc), max(1, len(vc) // n)))[:n]]).to(dev); xo = xs.roll(1, 0)
    with torch.autocast("cuda", torch.bfloat16):
        if a.head == "pose": return pt_loss(model(xs).float(), model(xo).float()).item()
        uv, src, ref = sample_q(len(xs), 512, a.L, g, dev)
        return (model(xs, uv, src, ref).float() - model(xo, uv, src, ref).float()).norm(dim=-1).mean().item()


MODES = ["anchor", "adjacent"] if a.head == "d4rt" else [None]
PRED = lambda mode: ev.predict if mode is None else (lambda m, x_, L_: ev.predict_q(m, x_, L_, mode))
au = model.audit
print(json.dumps(vars(a)), "val", va, flush=True)
print("AUDIT loaded %.2fM of encoder (%d tensors), ckpt tensors unused %s, random-init in encoder %.2fM; pretrained-lr params %.2fM, new %.2fM" % (
    au["loaded_params"] / 1e6, au["loaded_tensors"], list(au["ckpt_not_used"])[:2], au["random_init_params"] / 1e6, sum(p.numel() for g in groups[:1] for p in g["params"]) / 1e6,
    sum(p.numel() for g in groups[1:] for p in g["params"]) / 1e6), flush=True)
log, ev_log, step, t_train = [], [], 0, 0.
torch.cuda.reset_peak_memory_stats()
while step < a.steps:
    for x, Tg in dl:
        t0 = time.time(); step += 1
        x, Tg = x.to(dev, non_blocking=True), Tg.to(dev)
        if step == 1: print("first-batch checksum", round(x.sum().item(), 3), round(Tg.sum().item(), 3), flush=True)
        model.train(); loss, parts = losses(x, Tg)
        opt.zero_grad(set_to_none=True); loss.backward()
        if step % a.log_every == 0:  # gradient norm per module group (before clipping)
            gn = {}
            for n, p in model.named_parameters():
                if p.grad is not None: gn[coarse(n)] = gn.get(coarse(n), 0.) + p.grad.float().pow(2).sum().item()
            rec = dict(step=step, loss=round(loss.item(), 5), **{k: round(v, 5) for k, v in parts.items()}, lr_enc=sched.get_last_lr()[0], lr_new=sched.get_last_lr()[1],
                       grad_norm={k: round(math.sqrt(v), 5) for k, v in sorted(gn.items())}); log.append(rec); print(json.dumps(rec), flush=True)
        gtot = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
        torch.cuda.synchronize(); t_train += time.time() - t0
        if step % a.eval_every == 0 or step == a.steps:
            last = step == a.steps; m = dict(step=step, grad_norm_total=round(float(gtot), 4), img_sensitivity_mm=round(img_sens(), 5))
            if a.head == "d4rt": m["query_err_mm"] = query_eval()
            for mode in MODES:
                r = ev.evaluate(model, val, a.L, dev, 1 if last else 7, pred=PRED(mode), per_scan=last)
                m[mode or "pose"] = r[0] if last else r
                if last: m[(mode or "pose") + "_per_scan"] = r[1]
            ev_log.append(m)
            print("EVAL", json.dumps({k: ({kk: round(vv, 3) for kk, vv in v.items()} if isinstance(v, dict) else v) for k, v in m.items() if not k.endswith("per_scan")}), flush=True)
        if step == a.steps: break
import os
os.makedirs(CK, exist_ok=True)
res = dict(args=vars(a), val_subjects=va, train_log=log, eval_log=ev_log, peak_mem_gb=torch.cuda.max_memory_allocated() / 1e9, train_sec_per_step=t_train / a.steps,
           params_m=sum(p.numel() for p in model.parameters()) / 1e6, audit={k: v for k, v in au.items() if k != "ckpt_not_used"})
json.dump(res, open(f"runs/{a.name}.json", "w")); torch.save(model.state_dict(), f"{CK}{a.name}.pt")
print("DONE peak_mem_gb %.1f sec/step %.3f" % (res["peak_mem_gb"], res["train_sec_per_step"]), flush=True)
