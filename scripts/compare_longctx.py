"""Paired per-scan differences vs a reference run, subject-bootstrap 95% CI (10 dev subjects). Usage: compare_longctx.py ref run1 run2 ..."""
import json, sys, numpy as np
ld = lambda n: {(r["subject"], r["scan"]): r for r in json.load(open(f"outputs/longctx/{n}/eval_dev.json"))["rows"]}
ref = ld(sys.argv[1]); rng = np.random.default_rng(0)
for n in sys.argv[2:]:
    r = ld(n); ks = sorted(ref); subs = sorted({k[0] for k in ks}); out = []
    for m in ("GPE", "LPE"):
        d = {s: np.mean([r[k][m] - ref[k][m] for k in ks if k[0] == s]) for s in subs}; v = np.array(list(d.values()))
        bs = [rng.choice(v, len(v)).mean() for _ in range(5000)]
        out.append(f"{m} {v.mean():+.3f} [{np.percentile(bs,2.5):+.3f},{np.percentile(bs,97.5):+.3f}] subj<0: {(v<0).sum()}/{len(v)}")
    print(f"{n:12s}", " | ".join(out))
