"""Cache the 256-d per-pair features (pair.head(pair.net(.)) of adjacent frames, eval-mode BN) of a trained SeqPairCNN for every train/dev scan.
Output: <out>/<subject>_<scan>.pt = fp16 [n-1,256]. Used by train_longctx.py (hypothesis 1: temporal context longer than L=16 on frozen features).
Usage: cache_feats.py <seq ckpt> [--out outputs/feat_cache] [--splits train dev]"""
import argparse, os, sys, time, torch
sys.path.insert(0, os.path.dirname(__file__)); sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import train_seq
from usmotion4d.data import tusrec, split

ap = argparse.ArgumentParser(); ap.add_argument("ckpt"); ap.add_argument("--out", default="outputs/feat_cache"); ap.add_argument("--splits", nargs="+", default=["dev", "train"]); ap.add_argument("--bs", type=int, default=128)
a = ap.parse_args()
ck = torch.load(a.ckpt, map_location="cpu", weights_only=False); model = train_seq.build(ck["cfg"], init=False); model.load_state_dict(ck["state"]); model.eval()
size = tuple(ck["cfg"]["size"]); man = split.load_manifest(os.path.join(train_seq.REPO, ck["cfg"]["split"])); os.makedirs(a.out, exist_ok=True)


@torch.no_grad()
def feats(frames):
    x = tusrec.preprocess(frames, size).cuda(); out = []
    for i in range(0, len(x) - 1, a.bs):
        xx = x[i:i + a.bs + 1]; p = torch.cat([xx[:-1], xx[1:]], 1)
        with torch.autocast("cuda", torch.bfloat16): out.append(model.pair.head(model.pair.net((p - 0.449) / 0.226)).half().cpu())
    return torch.cat(out)


t0 = time.time()
for sp in a.splits:
    ents = tusrec.scan_paths(man[sp])
    for j, e in enumerate(ents):
        f = os.path.join(a.out, f"{e[0]}_{os.path.basename(e[1])[:-3]}.pt")
        if os.path.exists(f): continue
        torch.save(feats(tusrec.Scan(e).frames()), f + ".tmp"); os.replace(f + ".tmp", f)
        if j % 20 == 0: print(sp, j, len(ents), f"{time.time()-t0:.0f}s", flush=True)
print("done", f"{time.time()-t0:.0f}s")
