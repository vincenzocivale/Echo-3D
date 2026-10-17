"""python train.py --name global --window -1   |   --name local4 --window 4
Same seed => same init, same clip sampling, same val scans; only the attention window differs."""
import argparse, json, time, torch
import data, eval as ev
from model import Net, accumulate

a = argparse.ArgumentParser()
a.add_argument("--name", required=True); a.add_argument("--window", type=int, default=-1)  # -1 = global
a.add_argument("--L", type=int, default=64); a.add_argument("--stride", type=int, default=1)
a.add_argument("--bs", type=int, default=8); a.add_argument("--steps", type=int, default=3000)
a.add_argument("--lr", type=float, default=1e-4); a.add_argument("--lam", type=float, default=0.05)  # weight of global point loss
a.add_argument("--n_val", type=int, default=5); a.add_argument("--eval_every", type=int, default=500)
a.add_argument("--dim", type=int, default=384); a.add_argument("--depth", type=int, default=8)
a.add_argument("--log_every", type=int, default=50)
a = a.parse_args()
torch.manual_seed(0); dev = "cuda"

tr, va = data.split(a.n_val)
model = Net(a.dim, a.depth, a.dim // 64, window=None if a.window < 0 else a.window).to(dev)
dl = torch.utils.data.DataLoader(data.Clips(tr, a.L, a.stride), a.bs, shuffle=True, num_workers=8, drop_last=True, persistent_workers=True)
val = data.Clips(va, full=True)  # whole scans, stride 1 (the real metric)
opt = torch.optim.AdamW(model.parameters(), a.lr, weight_decay=0.05)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.steps, pct_start=0.05)
P = val.S.to(dev) @ torch.tensor([[1., 640, 1, 640, 320], [1, 1, 480, 480, 240], [0.] * 5, [1.] * 5], device=dev)  # corners+centre, mm
pt = lambda A, B: ((A - B)[..., :3, :] @ P).norm(dim=-2).mean()
gnorm = lambda l: torch.sqrt(sum(g.pow(2).sum() for g in torch.autograd.grad(l, model.parameters(), retain_graph=True, allow_unused=True) if g is not None)).item()

log, ev_log, t_train, step = [], [], 0., 0
torch.cuda.reset_peak_memory_stats()
print(json.dumps(vars(a)), "train subjects", len(tr), "val", va, flush=True)
while step < a.steps:
    for x, Tg in dl:
        t0 = time.time(); step += 1
        x, Tg = x.to(dev, non_blocking=True), Tg.to(dev)
        if step == 1:
            print("first-batch checksum", round(x.sum().item(), 3), round(Tg.sum().item(), 3), flush=True)
        Gl = torch.linalg.inv(Tg[:, :-1]) @ Tg[:, 1:]  # GT deltas T_{i->i-1}
        model.train()
        with torch.autocast("cuda", torch.bfloat16):
            Td = model(x)
        ll, lg = pt(Td, Gl), pt(accumulate(Td)[:, 1:], Tg[:, 1:])
        loss = ll + a.lam * lg
        if step % a.log_every == 0:  # per-term loss and gradient norm (global term scaled by lam)
            rec = dict(step=step, loss_local=ll.item(), loss_global=lg.item(), lam_loss_global=a.lam * lg.item(),
                       gn_local=gnorm(ll), gn_global_lam=a.lam * gnorm(lg)); log.append(rec)
            print(json.dumps(rec), flush=True)
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
        torch.cuda.synchronize(); t_train += time.time() - t0
        if step % a.eval_every == 0 or step == a.steps:
            m = ev.evaluate(model, val, a.L, dev, every=1 if step == a.steps else 6); m["step"] = step; ev_log.append(m)
            print("EVAL", json.dumps({k: round(v, 3) for k, v in m.items()}), flush=True)
        if step == a.steps:
            break
res = dict(args=vars(a), val_subjects=va, train_log=log, eval_log=ev_log, peak_mem_gb=torch.cuda.max_memory_allocated() / 1e9,
           train_sec_per_step=t_train / a.steps, params_m=sum(p.numel() for p in model.parameters()) / 1e6)
json.dump(res, open(f"runs/{a.name}.json", "w")); torch.save(model.state_dict(), f"runs/{a.name}.pt")
print("DONE peak_mem_gb %.1f sec/step %.3f" % (res["peak_mem_gb"], res["train_sec_per_step"]), flush=True)
