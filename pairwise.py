"""Pairwise control close to the official TUS-REC baseline: EfficientNet-B1 on 2 stacked frames -> 6-DoF T_{i->i-1},
point-coordinate loss on the local transform only, deltas chained at inference. Same data/split/metrics/geometry as our arms.
python pairwise.py --name pairwise --steps 3000"""
import argparse, json, time, torch, torch.nn as nn
from torchvision.models import efficientnet_b1
import data, eval as ev
from geom import pose_to_T
from model import ID
import pretrained

EFF = pretrained.CK + "effnet/efficientnet_b1_rwightman-bac287d4.pth"  # torchvision IMAGENET1K_V1 EfficientNet-B1

class Pair(nn.Module):
    def __init__(s, norm=False, imagenet=False):
        super().__init__()
        s.norm = norm or imagenet  # gray ImageNet normalisation (same preprocessing in both arms of a controlled comparison)
        s.f = efficientnet_b1(weights=None).features
        s.f[0][0] = nn.Conv2d(2, 32, 3, 2, 1, bias=False)  # frames (i-1, i) as channels
        s.head = nn.Linear(1280, 9)
        nn.init.zeros_(s.head.weight); nn.init.zeros_(s.head.bias)  # starts at identity
        s.audit = None
        if imagenet:  # all `features` tensors from ImageNet; first conv RGB->2 frames: W_gray/2 per frame (== pretrained response on the mean frame)
            ck = torch.load(EFF, map_location="cpu"); m = {"f." + k[9:]: v for k, v in ck.items() if k.startswith("features.")}
            m["f.0.0.weight"] = (m["f.0.0.weight"].sum(1, keepdim=True) / 2).repeat(1, 2, 1, 1)
            s.audit = pretrained.load_with_audit(s, m, {"f.0.0.weight": "3->2 channels: sum_c W_c / 2 for each of the two frames", "_src": {k for k in ck if k.startswith("features.")}}, ck)

    def forward(s, p):  # p [N,2,h,w] -> T_{i->i-1} [N,4,4]
        if s.norm: p = (p - .449) / .226
        z = s.f(p).mean((2, 3))
        with torch.autocast(z.device.type, enabled=False):
            return pose_to_T(s.head(z.float()) + ID.to(z.device))


@torch.no_grad()
def predict_pair(model, x, L=None, bs=128):
    """All adjacent pairs of a scan x [n,1,h,w] -> deltas [n-1,4,4]; only x is used."""
    d = []
    for a in range(0, len(x) - 1, bs):
        m = min(bs, len(x) - 1 - a)
        with torch.autocast(x.device.type, torch.bfloat16, enabled=x.device.type == "cuda"):
            d.append(model(torch.cat([x[a:a + m], x[a + 1:a + 1 + m]], 1)))
    return torch.cat(d), [0]


if __name__ == "__main__":
    a = argparse.ArgumentParser()
    a.add_argument("--name", required=True); a.add_argument("--steps", type=int, default=3000)
    a.add_argument("--lr", type=float, default=1e-4); a.add_argument("--bs", type=int, default=8)
    a.add_argument("--K", type=int, default=16)  # random adjacent pairs per clip (8*16=128 pairs/step)
    a.add_argument("--L", type=int, default=64); a.add_argument("--n_val", type=int, default=5); a.add_argument("--eval_every", type=int, default=500)
    a.add_argument("--init", default="random", choices=["random", "imagenet"]); a.add_argument("--norm", action="store_true")
    a = a.parse_args(); torch.manual_seed(0); dev = "cuda"
    tr, va = data.split(a.n_val)
    model = Pair(a.norm, a.init == "imagenet").to(dev)
    if model.audit: print("AUDIT", json.dumps({k: (v if k != "ckpt_not_used" else list(v)) for k, v in model.audit.items()}, default=str), flush=True)
    dl = torch.utils.data.DataLoader(data.Clips(tr, a.L), a.bs, shuffle=True, num_workers=8, drop_last=True, persistent_workers=True)
    val = data.Clips(va, full=True)
    opt = torch.optim.AdamW(model.parameters(), a.lr, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.steps, pct_start=0.05)
    P = val.S.to(dev) @ torch.tensor([[1., 640, 1, 640, 320], [1, 1, 480, 480, 240], [0.] * 5, [1.] * 5], device=dev)
    g = torch.Generator(device=dev); g.manual_seed(1234)
    log, ev_log, step, t_train = [], [], 0, 0.
    torch.cuda.reset_peak_memory_stats()
    print(json.dumps(vars(a)), "params (M)", sum(p.numel() for p in model.parameters()) / 1e6, flush=True)
    while step < a.steps:
        for x, Tg in dl:
            t0 = time.time(); step += 1
            x, Tg = x.to(dev), Tg.to(dev)
            if step == 1:
                print("first-batch checksum", round(x.sum().item(), 3), round(Tg.sum().item(), 3), flush=True)
            B, L = x.shape[:2]
            i = torch.randint(1, L, (B, a.K), generator=g, device=dev); b = torch.arange(B, device=dev)[:, None]
            pairs = torch.cat([x[b, i - 1], x[b, i]], 2).flatten(0, 1)  # [B*K,2,h,w]
            gt = (torch.linalg.inv(Tg[b, i - 1]) @ Tg[b, i]).flatten(0, 1)
            model.train()
            with torch.autocast("cuda", torch.bfloat16):
                T = model(pairs)
            loss = ((T - gt)[:, :3, :] @ P).norm(dim=-2).mean()
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
            torch.cuda.synchronize(); t_train += time.time() - t0
            if step % 50 == 0:
                log.append(dict(step=step, loss_local=loss.item())); print(json.dumps(log[-1]), flush=True)
            if step % a.eval_every == 0 or step == a.steps:
                m = ev.evaluate(model, val, a.L, dev, 1 if step == a.steps else 7, pred=predict_pair); m["step"] = step; ev_log.append(m)
                print("EVAL", json.dumps({k: round(v, 3) for k, v in m.items()}), flush=True)
            if step == a.steps:
                break
    res = dict(args=vars(a), val_subjects=va, train_log=log, eval_log=ev_log, peak_mem_gb=torch.cuda.max_memory_allocated() / 1e9,
               train_sec_per_step=t_train / a.steps, params_m=sum(p.numel() for p in model.parameters()) / 1e6)
    json.dump(res, open(f"runs/{a.name}.json", "w")); torch.save(model.state_dict(), f"runs/{a.name}.pt")
    print("DONE peak_mem_gb %.1f sec/step %.3f" % (res["peak_mem_gb"], res["train_sec_per_step"]), flush=True)
