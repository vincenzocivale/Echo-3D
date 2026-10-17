"""Train/evaluate SeqPairCNN (option B) on windows of L consecutive frames. Loss: adjacent steps plus steps composed inside the window
(k in loss_offsets), same normalised Huber as the pair models. Inference: overlapping windows (stride L/2), per-step xi averaged, then
sequential composition. Final eval writes eval_dev.json incl. the reversed-sequence check.
Usage: python scripts/train_seq.py configs/seq_effb1.yaml [--name X] [--steps N] [--overfit N] [--eval-only ckpt]"""
import argparse, json, os, random, sys, time, numpy as np, torch, yaml
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from usmotion4d.data import tusrec, split
from usmotion4d.geometry import se3, calib
from usmotion4d.models.seq_cnn import SeqPairCNN
from usmotion4d.models.losses import pose_loss
from usmotion4d.evaluation import pose_eval

REPO = os.path.join(os.path.dirname(__file__), "..")


def build(cfg, dev="cuda", init=True):
    m = SeqPairCNN(tuple(cfg["size"]), cfg["L"], scale_rot=cfg["loss"]["sigma_rot"], scale_trans=cfg["loss"]["sigma_trans"], **cfg["model"])
    if init and cfg.get("init_pair"):  # warm start: pair CNN trunk + feature layer from a trained PairCNN checkpoint (its pose layer is dropped)
        st = torch.load(os.path.join(REPO, cfg["init_pair"]), map_location="cpu", weights_only=False)["state"]
        print("init_pair:", m.pair.load_state_dict({k: v for k, v in st.items() if k.startswith(("net.", "head.1."))}, strict=False), flush=True)
    return m.to(dev)


def window_products(D, k):
    """D [B,m,4,4] adjacent steps -> [B,m-k+1,4,4] products of k consecutive steps (k power of two, by doubling)."""
    P, j = D, 1
    while j < k: P, j = P[:, :-j] @ P[:, j:], 2 * j
    return P


@torch.no_grad()
def predict_steps(model, frames, size, bs=8):
    """frames uint8 [n,H,W] -> averaged adjacent steps xi [n-1,6] from windows with stride L/2 (last window aligned to the end)."""
    L, n = model.L, len(frames)
    x = tusrec.preprocess(frames, size).cuda()
    starts = sorted(set(list(range(0, n - L + 1, L // 2)) + [n - L]))
    acc, cnt = torch.zeros(n - 1, 6, device="cuda"), torch.zeros(n - 1, 1, device="cuda")
    for a in range(0, len(starts), bs):
        ss = starts[a:a + bs]
        with torch.autocast("cuda", torch.bfloat16): xi = model(torch.stack([x[s:s + L] for s in ss]))["xi"]
        for s, v in zip(ss, xi): acc[s:s + L - 1] += v; cnt[s:s + L - 1] += 1
    return (acc / cnt).double()


def evaluate(model, entries, size, reverse=False):
    model.eval(); rows = []
    for e in entries:
        s = tusrec.Scan(e); F = s.frames(); Tg = s.poses.cuda()
        Tp = calib.sequential(se3.exp(predict_steps(model, F, size)))
        r = {**s.meta, **pose_eval.trajectory_metrics(Tp, Tg, s.S, s.landmarks)}
        if reverse:  # same scan played backwards: an image-driven model should be about as good; a direction prior breaks
            Tr = calib.sequential(se3.exp(predict_steps(model, F.flip(0), size)))
            r["rev_GPE"] = pose_eval.tusrec_metrics(Tr, se3.inverse(Tg[-1]) @ Tg.flip(0), s.S)["GPE"]
        rows.append(r)
    num = [k for k, v in rows[0].items() if isinstance(v, float)]
    mean = lambda rs: {k: sum(r[k] for r in rs) / len(rs) for k in num}
    out = dict(all=mean(rows), n_scans=len(rows))
    for g in ("trajectory", "orientation", "direction"): out[g] = {v: mean([r for r in rows if r[g] == v]) for v in sorted({r[g] for r in rows})}
    out["rows"] = rows
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("config"); ap.add_argument("--name"); ap.add_argument("--steps", type=int); ap.add_argument("--overfit", type=int); ap.add_argument("--eval-only")
    a = ap.parse_args(); cfg = yaml.safe_load(open(a.config))
    if a.steps: cfg["train"]["steps"] = a.steps
    man = split.load_manifest(os.path.join(REPO, cfg["split"])); dev_entries = tusrec.scan_paths(man["dev"])[::cfg["train"]["eval_scans_every"]]
    if a.eval_only:
        ck = torch.load(a.eval_only, map_location="cpu", weights_only=False); model = build(ck["cfg"], init=False); model.load_state_dict(ck["state"])
        res = evaluate(model, dev_entries, tuple(ck["cfg"]["size"]), reverse=True); json.dump(res, open(os.path.join(os.path.dirname(a.eval_only), "eval_dev.json"), "w"), indent=1)
        print("FINAL", {k: round(v, 3) for k, v in res["all"].items()}); return
    name = a.name or f"{cfg['arm']}_s{cfg['seed']}"; out = os.path.join(cfg["out_dir"], "pose", name); os.makedirs(out, exist_ok=True)
    random.seed(cfg["seed"]); np.random.seed(cfg["seed"]); torch.manual_seed(cfg["seed"])
    meta = dict(split_sha256=man["sha256"], encoder="SeqPairCNN", commit="no-git", seed=cfg["seed"]); json.dump({**meta, "cfg": cfg}, open(f"{out}/run.json", "w"), indent=1)
    t = cfg["train"]; size = tuple(cfg["size"]); model = build(cfg); params = model.trainable_parameters()
    print(f"{name}: {sum(p.numel() for p in params)/1e6:.2f}M trainable | split {man['sha256'][:8]} | L={cfg['L']} temporal={cfg['model'].get('temporal', True)}", flush=True)
    ds = tusrec.SeqDataset(tusrec.scan_paths(man["train"])[:2 if a.overfit else None], L=cfg["L"], size=size, length=t["steps"] * t["batch"], seed=cfg["seed"], n_unique=a.overfit)
    dl = torch.utils.data.DataLoader(ds, t["batch"], num_workers=t["workers"], drop_last=True, persistent_workers=True, prefetch_factor=4)
    opt = torch.optim.AdamW(params, lr=t["lr"], weight_decay=t["wd"])
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / t["warmup"]) * 0.5 * (1 + np.cos(np.pi * min(1, s / t["steps"]))))
    t0, hist = time.time(), []
    for step, b in enumerate(dl, 1):
        model.train(); x, T = b["x"].cuda(non_blocking=True), b["T"].cuda().double()
        with torch.autocast("cuda", torch.bfloat16): o = model(x)
        D, loss, d = o["delta"].double(), 0., {}
        for k in cfg["loss_offsets"]:  # adjacent (k=1) and composed-in-window steps, each normalised by its own k
            Pk = window_products(D, k); m = Pk.shape[1]
            gt = se3.inverse(T[:, :m]) @ T[:, k:k + m]
            lk, dk = pose_loss(se3.log(Pk).flatten(0, 1), gt.flatten(0, 1), torch.full((Pk.shape[0] * m,), k, device="cuda"), cfg["loss"]["sigma_rot"], cfg["loss"]["sigma_trans"])
            loss = loss + lk / len(cfg["loss_offsets"]); d.update({f"{n_}_k{k}": v for n_, v in dk.items() if n_ in ("rot_mrad", "trans_mm")})
        opt.zero_grad(set_to_none=True); loss.backward(); gn = torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); sched.step()
        hist.append((loss.item(), d))
        if step % 50 == 0:
            m_ = {k: float(np.mean([h[1][k] for h in hist[-50:]])) for k in hist[-1][1]}; m_.update(loss=float(np.mean([h[0] for h in hist[-50:]])), gnorm=gn.item())
            print(f"step {step} " + " ".join(f"{k}={v:.4g}" for k, v in m_.items()) + f" | {time.time()-t0:.0f}s {torch.cuda.max_memory_allocated()/2**30:.1f}GB", flush=True)
        if step % t["eval_every"] == 0 or step == t["steps"]:
            ev = evaluate(model, dev_entries, size)
            print("DEV", {k: round(v, 4) for k, v in ev["all"].items() if k in ("GPE", "GLE", "LPE", "LLE", "rot_x_mrad_k8", "rot_y_mrad_k8", "rot_z_mrad_k8", "trans_z_mm_k8", "corner_err_end")}, flush=True)
            torch.save(dict(state=model.state_dict(), cfg=cfg, step=step, **meta), f"{out}/last.pt")
    json.dump(dict(train_sec=time.time() - t0, peak_gb=torch.cuda.max_memory_allocated() / 2**30), open(f"{out}/train_stats.json", "w"))


if __name__ == "__main__":
    main()
