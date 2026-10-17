"""Reference that knows the acquisition pattern but sees NO image: for each val scan, the mean training step of scans with the same name
(arm_orientation_shape_direction), repeated. How much of a model's skill could come from knowing the typical motion of the pattern?
python type_prior.py -> runs/type_prior.json"""
import json, h5py, torch, collections
import data, eval as ev
from geom import T_to_pose, pose_to_T

tr, va = data.split(5); dev = "cuda"
trd = data.Clips(tr, full=True, size=(8, 8)); val = data.Clips(va, full=True, size=(8, 8))
typ = lambda p: p.split("/")[-1][:-3]
acc = collections.defaultdict(list)
for k in range(len(trd)):
    _, Tg = trd[k]; acc[typ(trd.paths[k])].append(T_to_pose(torch.linalg.inv(Tg[:-1]) @ Tg[1:]).mean(0))
mean = {t: torch.stack(v).mean(0) for t, v in acc.items()}
rs = []
for k in range(len(val)):
    _, Tg = val[k]; sid, name = ev.IDX(val.paths[k]); n = len(Tg)
    lm = torch.from_numpy(h5py.File(f"{data.ROOT}landmark_{sid}.h5")[name][()])
    rs.append(ev.errors(pose_to_T(mean[typ(val.paths[k])]).expand(n - 1, 4, 4).to(dev), Tg.to(dev), lm, val.S))
out = {k: round(sum(r[k] for r in rs if k in r) / sum(k in r for r in rs), 3) for k in rs[0]}
print(json.dumps(out)); json.dump(out, open("runs/type_prior.json", "w"))
