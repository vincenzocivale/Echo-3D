"""Train a PoseEstimator arm (A0..A3). Usage: python scripts/train_pose.py configs/pose_base.yaml [--arm A1] [--seed 17] [--steps N] [--name X]"""
import argparse, hashlib, json, os, random, subprocess, sys, time, numpy as np, torch, yaml
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from usmotion4d.data import tusrec, split
from usmotion4d.geometry import calib
from usmotion4d.models.encoder import USFMEncoder, CKPT
from usmotion4d.models.pose_estimator import PoseEstimator
from usmotion4d.models.pair_cnn import PairCNN
from usmotion4d.models.losses import pose_loss
from usmotion4d.evaluation import pose_eval

REPO = os.path.join(os.path.dirname(__file__), "..")


def build(cfg, dev="cuda"):
    if cfg["encoder"].get("type") == "cnn":  # trained from scratch, early fusion (no frozen encoder)
        return PairCNN(tuple(cfg["size"]), scale_rot=cfg["loss"]["sigma_rot"], scale_trans=cfg["loss"]["sigma_trans"], **cfg["model"]).to(dev)
    S, _ = calib.load_calib(tusrec.ROOT + "calib_matrix.csv")
    enc = USFMEncoder(tuple(cfg["size"]), tuple(cfg["encoder"]["layers"]), cfg["encoder"]["pretrained"], cfg["encoder"]["frozen"])
    vm = torch.from_numpy(np.load(os.path.join(REPO, "assets", "valid_mask_480x640.npy")))
    return PoseEstimator(
        enc, S, vm, cfg["arm"], scale_rot=cfg["loss"]["sigma_rot"], scale_trans=cfg["loss"]["sigma_trans"], **cfg["model"]).to(dev)


def save(model, cfg, path, extra):
    torch.save(dict(state={k: v for k, v in model.state_dict().items() if not k.startswith("encoder.")}, cfg=cfg, **extra), path)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("config"); ap.add_argument("--arm"); ap.add_argument("--seed", type=int); ap.add_argument("--steps", type=int); ap.add_argument("--name"); ap.add_argument("--overfit", type=int, help="train on N fixed pairs of 2 scans (diagnostic)")
    a = ap.parse_args(); cfg = yaml.safe_load(open(a.config))
    for k in ("arm", "seed"):
        if getattr(a, k) is not None: cfg[k] = getattr(a, k)
    if a.steps: cfg["train"]["steps"] = a.steps
    name = a.name or f"{cfg['arm']}_s{cfg['seed']}"; out = os.path.join(cfg["out_dir"], "pose", name); os.makedirs(out, exist_ok=True)
    random.seed(cfg["seed"]); np.random.seed(cfg["seed"]); torch.manual_seed(cfg["seed"]); torch.backends.cudnn.deterministic = True
    man = split.load_manifest(os.path.join(REPO, cfg["split"]))
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=REPO).stdout.strip() or "no-git"
    cnn = cfg["encoder"].get("type") == "cnn"
    ck_sha = None if cnn else hashlib.sha256(open(CKPT, "rb").read()).hexdigest()
    meta = dict(split_sha256=man["sha256"], encoder="PairCNN (scratch)" if cnn else "USFM ViT-B/16 (USFM_latest.pth)", encoder_sha256=ck_sha, commit=commit, seed=cfg["seed"])
    json.dump({**meta, "cfg": cfg}, open(f"{out}/run.json", "w"), indent=1)
    wb = None
    if cfg["wandb"]["enabled"]:
        import wandb; wb = wandb.init(project=cfg["wandb"]["project"], name=name, config={**cfg, **meta})
    dev = "cuda"; t = cfg["train"]
    model = build(cfg, dev); params = model.trainable_parameters()
    print(f"{name}: {sum(p.numel() for p in params)/1e6:.2f}M trainable | split {man['sha256'][:8]} | encoder: {cfg['encoder']}", flush=True)
    ds = tusrec.PairDataset(tusrec.scan_paths(man["train"])[:2 if a.overfit else None], cfg["offsets"], tuple(cfg["size"]), length=t["steps"] * t["batch"], seed=cfg["seed"], n_unique=a.overfit, reverse_prob=cfg["train"].get("reverse_prob", 0.0))
    dl = torch.utils.data.DataLoader(ds, t["batch"], num_workers=t["workers"], shuffle=False, drop_last=True, persistent_workers=True, prefetch_factor=4)
    opt = torch.optim.AdamW(params, lr=t["lr"], weight_decay=t["wd"])
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / t["warmup"]) * 0.5 * (1 + np.cos(np.pi * min(1, s / t["steps"]))))
    dev_entries = tusrec.scan_paths(man["dev"]); t0 = time.time(); hist = []
    for step, b in enumerate(dl, 1):
        model.train(); x = b["x"].to(dev, non_blocking=True); B = len(x); k = b["k"].to(dev).float()
        with torch.autocast("cuda", torch.bfloat16): f = model.encode(x.flatten(0, 1)); f = f.view(B, 2, *f.shape[1:])
        with torch.autocast("cuda", torch.bfloat16): o = model(f[:, 0], f[:, 1], k)
        loss, d = pose_loss(o["xi"], b["delta"].to(dev), k, cfg["loss"]["sigma_rot"], cfg["loss"]["sigma_trans"], o["logvar"] if cfg["loss"]["nll"] else None)
        opt.zero_grad(set_to_none=True); loss.backward(); gn = torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); sched.step()
        hist.append((loss.item(), d))
        if step % 50 == 0:
            m = {k_: float(np.mean([h[1][k_] for h in hist[-50:]])) for k_ in hist[-1][1]}; m.update(loss=float(np.mean([h[0] for h in hist[-50:]])), gnorm=gn.item(), lr=sched.get_last_lr()[0], step=step)
            print(f"step {step} " + " ".join(f"{k_}={v:.4g}" for k_, v in m.items()) + f" | {time.time()-t0:.0f}s {torch.cuda.max_memory_allocated()/2**30:.1f}GB", flush=True)
            if wb: wb.log(m)
        if step % t["eval_every"] == 0 or step == t["steps"]:
            ev = pose_eval.evaluate(model, dev_entries, tuple(cfg["size"]), every=t["eval_scans_every"], diag=True)
            print("DEV", {k_: round(v, 4) for k_, v in ev["all"].items() if k_ in ("GPE", "GLE", "LPE", "LLE", "rot_mrad_k1", "trans_mm_k1", "trans_mm_k8")}, flush=True)
            if wb: wb.log({f"dev/{k_}": v for k_, v in ev["all"].items()})
            save(model, cfg, f"{out}/last.pt", dict(step=step, **meta))
    json.dump(dict(train_sec=time.time() - t0, peak_gb=torch.cuda.max_memory_allocated() / 2**30), open(f"{out}/train_stats.json", "w"))


if __name__ == "__main__":
    main()
