"""D4RT-style arm: same encoder/data/seed/steps as train.py, but a point-query decoder trained with L1 on 3D points only.
python train_q.py --name d4rt --window -1"""
import argparse, json, math, random, time, torch
import data, eval as ev
from geom import query_gt
from model import Net, D4RT

a = argparse.ArgumentParser()
a.add_argument("--name", required=True); a.add_argument("--window", type=int, default=-1)  # encoder window, -1 = global
a.add_argument("--L", type=int, default=64); a.add_argument("--stride", type=int, default=1)
a.add_argument("--bs", type=int, default=8); a.add_argument("--steps", type=int, default=3000)
a.add_argument("--lr", type=float, default=1e-4); a.add_argument("--Q", type=int, default=512)  # queries per clip
a.add_argument("--id_frac", type=float, default=0.1)  # fraction of identity queries (t_src == t_ref)
a.add_argument("--n_val", type=int, default=5); a.add_argument("--eval_every", type=int, default=500)
a.add_argument("--dim", type=int, default=384); a.add_argument("--depth", type=int, default=8); a.add_argument("--dec_dim", type=int, default=256)
a.add_argument("--log_every", type=int, default=50)
a = a.parse_args()
torch.manual_seed(0); dev = "cuda"

tr, va = data.split(a.n_val)
enc = Net(a.dim, a.depth, a.dim // 64, window=None if a.window < 0 else a.window)  # same init as the pose-regression baseline
val = data.Clips(va, full=True)
with torch.random.fork_rng(devices=[]):  # decoder init must not shift the global RNG stream (=> identical clip sampling)
    torch.manual_seed(1)
    model = D4RT(enc, val.S, dim=a.dec_dim, heads=a.dec_dim // 32).to(dev)
dl = torch.utils.data.DataLoader(data.Clips(tr, a.L, a.stride), a.bs, shuffle=True, num_workers=8, drop_last=True, persistent_workers=True)
opt = torch.optim.AdamW(model.parameters(), a.lr, weight_decay=0.05)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.steps, pct_start=0.05)
g = torch.Generator(device=dev); g.manual_seed(1234)  # query sampler has its own stream
BINS = [(0, 0), (1, 3), (4, 15), (16, a.L)]  # |t_ref - t_src|


def sample_q(B, Q, L):
    """Uniform pixels over the whole image; src uniform; |dt| log-uniform in [1,L-1] (short..long), random sign; id_frac identity."""
    r = lambda *s: torch.rand(*s, generator=g, device=dev)
    uv, src = r(B, Q, 2), torch.randint(0, L, (B, Q), generator=g, device=dev)
    dt = torch.exp(r(B, Q) * math.log(L - 1)).round().long().clamp(1, L - 1)
    sg = torch.randint(0, 2, (B, Q), generator=g, device=dev) * 2 - 1
    ref = src + sg * dt
    ref = torch.where((ref < 0) | (ref >= L), src - sg * dt, ref).clamp(0, L - 1)
    return uv, src, torch.where(r(B, Q) < a.id_frac, src, ref)


def by_bin(err, src, ref, fn=lambda e: e.mean().item()):
    d = (ref - src).abs()
    return {f"dt{lo}-{hi}": fn(err[(d >= lo) & (d <= hi)]) for lo, hi in BINS if ((d >= lo) & (d <= hi)).any()}


@torch.no_grad()
def query_eval(n_clips=24):
    """Euclidean 3D error (mm) of queries on fixed val clips, per temporal distance."""
    model.eval(); random.seed(123); vc = data.Clips(va, a.L); E, S_, R_ = [], [], []
    for k in range(0, len(vc), max(1, len(vc) // n_clips)):
        x, Tg = vc[k]; x, Tg = x[None].to(dev), Tg[None].to(dev)
        uv, src, ref = sample_q(1, 2048, a.L)
        with torch.autocast("cuda", torch.bfloat16):
            P = model(x, uv, src, ref)
        E.append((P.float() - query_gt(Tg, model.S, uv, src, ref)).norm(dim=-1)); S_.append(src); R_.append(ref)
    return by_bin(torch.cat(E), torch.cat(S_), torch.cat(R_))


log, ev_log, t_train, step = [], [], 0., 0
torch.cuda.reset_peak_memory_stats()
n_enc, n_dec = sum(p.numel() for p in model.enc.parameters()) - sum(p.numel() for p in model.enc.head.parameters()), sum(p.numel() for p in model.dec.parameters())
print(json.dumps(vars(a)), "train subjects", len(tr), "val", va, "params enc/dec (M)", n_enc / 1e6, n_dec / 1e6, flush=True)
while step < a.steps:
    for x, Tg in dl:
        t0 = time.time(); step += 1
        x, Tg = x.to(dev, non_blocking=True), Tg.to(dev)
        if step == 1:
            print("first-batch checksum", round(x.sum().item(), 3), round(Tg.sum().item(), 3), flush=True)
        uv, src, ref = sample_q(len(x), a.Q, a.L)
        gt = query_gt(Tg, model.S, uv, src, ref)  # training-only use of the tracked poses
        model.train()
        with torch.autocast("cuda", torch.bfloat16):
            P = model(x, uv, src, ref)
        err = (P.float() - gt).abs()
        loss = err.mean()  # L1 on 3D coordinates (mm), the only loss
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
        torch.cuda.synchronize(); t_train += time.time() - t0
        if step % a.log_every == 0:
            rec = dict(step=step, loss_l1_mm=loss.item(), **{"l1_" + k: round(v, 4) for k, v in by_bin(err.mean(-1), src, ref).items()}); log.append(rec)
            print(json.dumps(rec), flush=True)
        if step % a.eval_every == 0 or step == a.steps:
            ev_every = 1 if step == a.steps else 7
            m = dict(step=step, query_err_mm=query_eval())
            for mode in ("anchor", "adjacent"):
                m[mode] = ev.evaluate(model, val, a.L, dev, ev_every, pred=lambda mo, x_, L_: ev.predict_q(mo, x_, L_, mode))
            ev_log.append(m); print("EVAL", json.dumps(m, default=lambda v: round(v, 3)), flush=True)
        if step == a.steps:
            break
res = dict(args=vars(a), val_subjects=va, train_log=log, eval_log=ev_log, peak_mem_gb=torch.cuda.max_memory_allocated() / 1e9,
           train_sec_per_step=t_train / a.steps, params_enc_m=n_enc / 1e6, params_dec_m=n_dec / 1e6)
res["sensitivity"] = {k: ev.sensitivity(model, val, a.L, dev, lambda mo, x_, L_: ev.predict_q(mo, x_, L_, "anchor"), k) for k in (0, len(val) // 2)}
json.dump(res, open(f"runs/{a.name}.json", "w")); torch.save(model.state_dict(), f"runs/{a.name}.pt")
print("DONE peak_mem_gb %.1f sec/step %.3f" % (res["peak_mem_gb"], res["train_sec_per_step"]), json.dumps(res["sensitivity"]), flush=True)
