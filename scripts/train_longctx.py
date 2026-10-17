"""Hypothesis 1 (docs/IMPROVEMENT_HYPOTHESES.md): temporal head (same transformer as SeqPairCNN) trained on CACHED pair features (cache_feats.py),
window length L varied. Isolates the effect of temporal context: features are frozen and identical for every L.
Usage: train_longctx.py --L 64 --name L64 [--offsets 1 2 4 8] [--steps 4000] [--batch 16] [--seed 17]
       train_longctx.py --sanity <seq ckpt>   # head from the checkpoint on cached features must reproduce its eval_dev.json (dev every 7th, L=16)"""
import argparse, json, os, random, sys, time, numpy as np, torch, torch.nn as nn
sys.path.insert(0, os.path.dirname(__file__)); sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from train_seq import window_products
from usmotion4d.data import tusrec, split
from usmotion4d.geometry import se3, calib
from usmotion4d.models.losses import pose_loss
from usmotion4d.evaluation import pose_eval

REPO = os.path.join(os.path.dirname(__file__), "..")
SIG_ROT, SIG_TR = 0.002, 0.4


class Head(nn.Module):
    """Temporal head of SeqPairCNN (pos + transformer + out), same init."""

    def __init__(self, L, dim=256, depth=2, heads=4, arch="tf"):
        super().__init__()
        self.L, self.arch = L, arch; self.register_buffer("scale", torch.tensor([SIG_ROT] * 3 + [SIG_TR] * 3))
        if arch == "tf":  # learned absolute positions (as SeqPairCNN)
            self.pos = nn.Parameter(torch.zeros(L - 1, dim)); nn.init.normal_(self.pos, std=.02)
        elif arch == "tf_sin":  # fixed sinusoidal positions
            t = torch.arange(L - 1)[:, None] * torch.exp(-np.log(1e4) * torch.arange(0, dim, 2) / dim)[None]
            self.register_buffer("pos", torch.cat([t.sin(), t.cos()], 1))
        if arch.startswith("tf"):
            self.tf = nn.TransformerEncoder(nn.TransformerEncoderLayer(dim, heads, 4 * dim, 0., "gelu", batch_first=True, norm_first=True), depth, enable_nested_tensor=False)
        else:  # "gru": bidirectional, no positions
            self.gru = nn.GRU(dim, dim // 2, 2, batch_first=True, bidirectional=True)
        self.out = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 6)); nn.init.zeros_(self.out[-1].weight); nn.init.zeros_(self.out[-1].bias)

    def forward(self, f):
        """f [B,m,dim] with m <= L-1 -> xi [B,m,6]."""
        h = self.gru(f)[0] if self.arch == "gru" else self.tf(f + self.pos[:f.shape[1]])
        return self.out(h).float() * self.scale


def key(e): return f"{e[0]}_{os.path.basename(e[1])[:-3]}"


def load(entries, cache):
    return [(e, torch.load(os.path.join(cache, key(e) + ".pt"))) for e in entries]


@torch.no_grad()
def predict_steps(head, feats, bs=16):
    """feats [n-1,dim] -> averaged adjacent steps [n-1,6] from windows of L frames (L-1 features), stride L/2, last window aligned to the end.
    Scans shorter than L use one window of the whole scan."""
    L, nf = head.L, len(feats); w = min(L - 1, nf); st = max(w // 2, 1)
    starts = sorted(set(list(range(0, nf - w + 1, st)) + [nf - w]))
    acc, cnt = torch.zeros(nf, 6, device="cuda"), torch.zeros(nf, 1, device="cuda")
    for a in range(0, len(starts), bs):
        ss = starts[a:a + bs]; xi = head(torch.stack([feats[s:s + w] for s in ss]).float())
        for s, v in zip(ss, xi): acc[s:s + w] += v; cnt[s:s + w] += 1
    return (acc / cnt).double()


def evaluate(head, data):
    head.eval(); rows = []
    for e, f in data:
        s = tusrec.Scan(e); Tg = s.poses.cuda()
        Tp = calib.sequential(se3.exp(predict_steps(head, f.cuda())))
        rows.append({**s.meta, **pose_eval.trajectory_metrics(Tp, Tg, s.S, s.landmarks)})
    num = [k for k, v in rows[0].items() if isinstance(v, float)]
    return dict(all={k: sum(r[k] for r in rows) / len(rows) for k in num}, n_scans=len(rows), rows=rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--L", type=int, default=16); ap.add_argument("--name"); ap.add_argument("--offsets", type=int, nargs="+", default=[1, 2, 4, 8])
    ap.add_argument("--steps", type=int, default=4000); ap.add_argument("--batch", type=int, default=16); ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=17); ap.add_argument("--cache", default=os.path.join(REPO, "outputs/feat_cache")); ap.add_argument("--dev-every", type=int, default=1)
    ap.add_argument("--sanity"); ap.add_argument("--arch", default="tf")
    a = ap.parse_args(); man = split.load_manifest(os.path.join(REPO, "configs/splits/split_v1.json"))
    dev_entries = tusrec.scan_paths(man["dev"])[::a.dev_every]
    if a.sanity:
        ck = torch.load(a.sanity, map_location="cpu", weights_only=False); head = Head(ck["cfg"]["L"]).cuda()
        head.load_state_dict({k: v for k, v in ck["state"].items() if k.startswith(("pos", "tf.", "out."))}, strict=False)
        res = evaluate(head, load(tusrec.scan_paths(man["dev"])[::7], a.cache)); print("SANITY", {k: round(v, 3) for k, v in res["all"].items() if k in ("GPE", "GLE", "LPE", "LLE")}, "(expected 12.865 / 11.148 / 0.152 / 0.138)"); return
    name = a.name or f"L{a.L}"; out = os.path.join(REPO, "outputs", "longctx", name); os.makedirs(out, exist_ok=True)
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    S, C = calib.load_calib(tusrec.ROOT + "calib_matrix.csv")
    train = []
    for e, f in load(tusrec.scan_paths(man["train"]), a.cache):
        if len(f) + 1 >= a.L: train.append((f.cuda(), calib.plane_poses(tusrec.Scan(e).tforms, C).cuda()))  # scans shorter than L are not sampled
    print(f"{name}: L={a.L} offsets={a.offsets} | {len(train)} train scans usable", flush=True)
    dev = load(dev_entries, a.cache); head = Head(a.L, arch=a.arch).cuda(); params = list(head.parameters())
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / 200) * 0.5 * (1 + np.cos(np.pi * min(1, s / a.steps))))
    rng = random.Random(a.seed); t0 = time.time(); hist = []
    for step in range(1, a.steps + 1):
        head.train(); fs, Ts = [], []
        for _ in range(a.batch):
            f, T = train[rng.randrange(len(train))]; s = rng.randrange(0, len(T) - a.L + 1)
            fs.append(f[s:s + a.L - 1].float()); Ts.append(se3.inverse(T[s]) @ T[s:s + a.L])
        D = se3.exp(head(torch.stack(fs))).double(); T = torch.stack(Ts); loss, d = 0., {}
        for k in a.offsets:
            Pk = window_products(D, k); m = Pk.shape[1]; gt = se3.inverse(T[:, :m]) @ T[:, k:k + m]
            lk, dk = pose_loss(se3.log(Pk).flatten(0, 1), gt.flatten(0, 1), torch.full((Pk.shape[0] * m,), k, device="cuda"), SIG_ROT, SIG_TR)
            loss = loss + lk / len(a.offsets); d.update({f"{n_}_k{k}": v for n_, v in dk.items() if n_ in ("rot_mrad", "trans_mm")})
        opt.zero_grad(set_to_none=True); loss.backward(); gn = torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); sched.step(); hist.append(loss.item())
        if step % 250 == 0: print(f"step {step} loss={np.mean(hist[-250:]):.4g} gnorm={gn.item():.3g} | {time.time()-t0:.0f}s", flush=True)
        if step % 1000 == 0 or step == a.steps:
            ev = evaluate(head, dev[::7] if step < a.steps else dev)
            print("DEV" + ("(every7)" if step < a.steps else f"(n={len(dev)})"), {k: round(v, 4) for k, v in ev["all"].items() if k in ("GPE", "GLE", "LPE", "LLE", "corner_err_end")}, flush=True)
    torch.save(dict(state=head.state_dict(), L=a.L, args=vars(a)), f"{out}/head.pt"); json.dump(dict(ev, args=vars(a)), open(f"{out}/eval_dev.json", "w"), indent=1)


if __name__ == "__main__":
    main()
