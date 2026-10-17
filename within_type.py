"""Do the models explain motion beyond the pattern mean? Per-step translation/rotation of all 120 val scans:
 corr(pred, GT) overall vs corr(pred - mu_type, GT - mu_type) within pattern; skill vs the type-mean prior = 1 - SSE(model)/SSE(prior).
python within_type.py -> runs/within_type.json"""
import json, collections, torch
import data, eval as ev, pairwise, pretrained as pt
from geom import T_to_pose

dev = "cuda"; tr, va = data.split(5); val = data.Clips(va, full=True); trd = data.Clips(tr, full=True, size=(8, 8)); W = "/raid/DATASETS/Echo3D_weights/matrix/"
typ = lambda p: p.split("/")[-1][:-3]
vee = lambda R: .5 * torch.stack([R[:, 2, 1] - R[:, 1, 2], R[:, 0, 2] - R[:, 2, 0], R[:, 1, 0] - R[:, 0, 1]], -1)
par = lambda T: torch.cat([T[:, :3, 3], vee(T[:, :3, :3])], -1)  # tx ty tz wx wy wz
acc = collections.defaultdict(list)
for k in range(len(trd)):
    _, Tg = trd[k]; acc[typ(trd.paths[k])].append(par(torch.linalg.inv(Tg[:-1]) @ Tg[1:]).mean(0))
mu = {t: torch.stack(v).mean(0) for t, v in acc.items()}
GT, MU, names = [], [], []
for k in range(len(val)):
    _, Tg = val[k]; g = par(torch.linalg.inv(Tg[:-1]) @ Tg[1:]); GT.append(g); MU.append(mu[typ(val.paths[k])].expand_as(g)); names += [k] * len(g)
GT, MU = torch.cat(GT), torch.cat(MU)
models = {}
for head in ("pose", "d4rt"):
    m = pt.build("videomae", head, val.S).to(dev); m.load_state_dict(torch.load(f"{W}videomae_{head}.pt", map_location=dev)); m.eval()
    models[f"videomae+{head}"] = (m, ev.predict if head == "pose" else (lambda mo, x, L: ev.predict_q(mo, x, L, "anchor")))
for n, norm, ck in (("pair-ImageNet", True, "runs/pairwise_imagenet.pt"), ("pair-random", True, "runs/pairwise_random_norm.pt")):
    m = pairwise.Pair(norm=norm).to(dev); m.load_state_dict(torch.load(ck, map_location=dev)); m.eval(); models[n] = (m, pairwise.predict_pair)
out = {}
corr = lambda a, b: torch.corrcoef(torch.stack([a, b]))[0, 1].item()
for n, (m, pred) in models.items():
    P = torch.cat([par(pred(m, val[k][0].to(dev), 64)[0].float().cpu()) for k in range(len(val))])
    comp = {}
    for i, c in enumerate(["tx", "ty", "tz", "wx", "wy", "wz"]):
        comp[c] = dict(corr_overall=round(corr(P[:, i], GT[:, i]), 3), corr_within_pattern=round(corr(P[:, i] - MU[:, i], GT[:, i] - MU[:, i]), 3),
                       skill_vs_type_prior=round(1 - ((P[:, i] - GT[:, i]) ** 2).sum().item() / ((MU[:, i] - GT[:, i]) ** 2).sum().item(), 3),
                       skill_vs_zero=round(1 - ((P[:, i] - GT[:, i]) ** 2).sum().item() / (GT[:, i] ** 2).sum().item(), 3),
                       prior_corr_overall=round(corr(MU[:, i], GT[:, i]), 3))
    out[n] = comp; print(n, {c: (v["corr_overall"], v["corr_within_pattern"], v["skill_vs_type_prior"]) for c, v in comp.items() if c.startswith("t")}, flush=True)
json.dump(out, open("runs/within_type.json", "w"), indent=1)
