"""Trivial references on the val scans: 'none' = identity deltas, 'prior' = the mean training delta repeated (constant velocity).
Any learned model must beat 'prior' to claim it uses the images. python floors.py [n_val] [every]"""
import json, sys, h5py, torch
import data, eval as ev
from geom import T_to_pose, pose_to_T

n_val, every = int(sys.argv[1]) if len(sys.argv) > 1 else 5, int(sys.argv[2]) if len(sys.argv) > 2 else 1
tr, va = data.split(n_val); dev = "cuda"
trd = data.Clips(tr, full=True, size=(8, 8))  # frames unused here (tiny), only the poses
ps = []
for k in range(0, len(trd), 8):
    _, Tg = trd[k]; ps.append(T_to_pose(torch.linalg.inv(Tg[:-1]) @ Tg[1:]))
mean = torch.cat(ps).mean(0)
print("mean train step (mm):", mean[6:].tolist())
val = data.Clips(va, full=True, size=(8, 8)); rs = {"none": [], "prior": []}
for k in range(0, len(val), every):
    _, Tg = val[k]; n = len(Tg); sid, name = ev.IDX(val.paths[k])
    lm = torch.from_numpy(h5py.File(f"{data.ROOT}landmark_{sid}.h5")[name][()])
    for tag, Td in dict(none=torch.eye(4).expand(n - 1, 4, 4), prior=pose_to_T(mean).expand(n - 1, 4, 4)).items():
        rs[tag].append(ev.errors(Td.to(dev), Tg.to(dev), lm, val.S))
print(json.dumps({t: {k: round(sum(r[k] for r in v if k in r) / sum(k in r for r in v), 3) for k in v[0]} for t, v in rs.items()}))
