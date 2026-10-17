"""Unit tests for the D4RT-style query pipeline (CPU, real calibration/scan). python test_query.py"""
import inspect, h5py, torch
import data, eval as ev
from geom import *
from geom import rel
from model import Net, D4RT, accumulate, crop

torch.manual_seed(0)
ds = data.Clips(data.split()[1], full=True)
x, Tg = ds[0]; n, S = len(Tg), ds.S
C = ds.C
with h5py.File(ds.paths[0]) as f:
    T = torch.from_numpy(f["tforms"][()])
sid, name = ev.IDX(ds.paths[0])
lm = torch.from_numpy(h5py.File(f"{data.ROOT}landmark_{sid}.h5")[name][()])
B, Q = 2, 64
uv = torch.rand(B, Q, 2); src = torch.randint(0, n, (B, Q)); ref = torch.randint(0, n, (B, Q))
ph = lambda u: torch.cat([uv_to_mm(u, S), torch.ones_like(u[..., :1])], -1)

# 1. GT 3D points == calibrated relative pose applied to the pixel (rel() was validated against the baseline)
P = query_gt(Tg[None].expand(B, -1, -1, -1), S, uv, src, ref)
for b in range(B):
    want = (rel(T, C, src[b], ref[b]) @ ph(uv[b])[..., None])[:, :3, 0]
    assert (P[b] - want).abs().max() < 2e-3, (P[b] - want).abs().max()
a_, b_ = 37, 211  # uv <-> pixel_grid (1-based pixel centres)
assert torch.allclose(uv_to_mm(torch.tensor([(a_ - .5) / W, (b_ - .5) / H]), S), pixel_grid(S)[:3, (b_ - 1) * W + a_ - 1], atol=1e-4)
# resizing keeps pixel<->metric correspondence: a 4x4 full-res block at pixels x=4j+1..4j+4 -> low-res pixel j; crop centre hits it
img = torch.zeros(1, 1, 1, 480, 640); img[..., 4 * 30:4 * 30 + 4, 4 * 50:4 * 50 + 4] = 1  # rows 30, cols 50 (0-based low-res)
lo = torch.nn.functional.interpolate(img[0], (120, 160), mode="area")[None]
u_c = torch.tensor([[[(4 * 50 + 2) / 640, (4 * 30 + 2) / 480]]])  # centre of that block
pc = crop(lo, u_c, torch.zeros(1, 1, dtype=torch.long))[0, 0]
assert pc[40] == 1 and pc.sum() == 1, pc.view(9, 9)
# 2. gauge: a global change of the world frame must not change anything
G = torch.eye(4); G[:3, :3] = torch.linalg.qr(torch.randn(3, 3))[0] * torch.tensor([1., 1, 1]); G[:3, 3] = torch.tensor([50., -20, 300])
if torch.linalg.det(G[:3, :3]) < 0: G[:3, :3] *= -1
Tg2 = rel(G @ T, C, torch.arange(n), torch.zeros(n, dtype=torch.long))
assert (query_gt(Tg2[None].expand(B, -1, -1, -1), S, uv, src, ref) - P).abs().max() < 2e-3
# 3. identity queries: same source/reference frame -> the known calibrated pixel position
Pi = query_gt(Tg[None].expand(B, -1, -1, -1), S, uv, src, src)
assert (Pi - uv_to_mm(uv, S)).abs().max() < 1e-3

# 4. Kabsch: exact on planar points, proper rotation even for a reflection, no scale, flags collinear sets
p = torch.cat([torch.rand(64, 2) * 100, torch.zeros(64, 1)], -1)
R0 = torch.linalg.qr(torch.randn(3, 3))[0]; R0 = R0 * torch.linalg.det(R0).sign(); t0 = torch.randn(3) * 50
R, t, s, e = kabsch(p, p @ R0.T + t0)
assert (R - R0.double()).abs().max() < 1e-4 and e < 1e-4 and s[1] / s[0] > 1e-2
Rr, _, _, er = kabsch(p, p * torch.tensor([1., 1, -1]) @ R0.T)  # planar: reflection through the plane is a proper rotation of it
assert torch.allclose(torch.linalg.det(Rr), torch.tensor(1., dtype=torch.float64))
Rf, _, _, ef = kabsch(p, (p * torch.tensor([-1., 1, 1])) + torch.tensor([0., 0, 5]) + torch.rand(64, 3) * 5)
assert torch.allclose(torch.linalg.det(Rf), torch.tensor(1., dtype=torch.float64), atol=1e-6) and ef > 1  # no reflection, residual shows misfit
assert kabsch(p, 1.1 * p @ R0.T)[3] > 1  # scale is not absorbed
line = torch.stack([torch.linspace(0, 100, 64), torch.zeros(64), torch.zeros(64)], -1)
sl = kabsch(line, line @ R0.T)[2]; assert sl[1] / sl[0] < 1e-6  # degenerate (rank 1) is detectable


# 5. oracle decoder: exact 3D points from the tracked poses -> exact poses (anchor/adjacent modes, clip lengths, chaining)
class Oracle(torch.nn.Module):
    def __init__(s):
        super().__init__(); s.register_buffer("S", S.float())
    def encode(s, clip): return clip
    def decode(s, feats, clip, uv_, src_, ref_):
        i = clip[0, :, 0, 0, 0].long()  # global frame ids hidden in the "image": only the oracle may look up GT
        return query_gt(Tg[None], s.S, uv_.double(), i[src_[0]][None], i[ref_[0]][None]).float()


xi = torch.arange(n, dtype=torch.float32)[:, None, None, None].expand(n, 1, 4, 4)
for mode in ("anchor", "adjacent"):
    for L in (7, 64, n + 3):
        Td, st, info = ev.predict_q(Oracle(), xi, L, mode)
        e = ev.errors(Td, Tg, lm, S, st)
        assert max(e.values()) < 2e-2 and info["rigid_rms"] < 1e-3 and info["sv_ratio_min"] > 1e-2, (mode, L, e, info)
        assert abs(e.get("bnd", 0) - e.get("int", 0)) < 1e-2  # no discontinuity at clip boundaries

# 6. a real (random-weight) model: queries are independent of each other (subset / permutation / presence of others)
enc = Net(dim=64, depth=2, heads=2)
m = D4RT(enc, S, dim=64, depth=2, heads=4).eval()
torch.nn.init.normal_(m.dec.out[1].weight, std=.05)
xc = x[:12][None]
with torch.no_grad():
    f = m.encode(xc); u1, s1, r1 = uv[:1, :40], src[:1, :40] % 12, ref[:1, :40] % 12
    full = m.decode(f, xc, u1, s1, r1)
    perm = torch.randperm(40)
    assert (m.decode(f, xc, u1[:, perm], s1[:, perm], r1[:, perm]) - full[:, perm]).abs().max() < 1e-3
    assert (m.decode(f, xc, u1[:, :5], s1[:, :5], r1[:, :5]) - full[:, :5]).abs().max() < 1e-3
    # sensitive to the images (not only to the query): different frames -> different output
    full2 = m.decode(m.encode(torch.rand_like(xc)), torch.rand_like(xc), u1, s1, r1)
    assert (full2 - full).abs().mean() > 1e-3
# 7. no GT in the inference path: neither predict_q nor the model's forward can receive poses
assert list(inspect.signature(ev.predict_q).parameters) == ["model", "x", "L", "mode", "grid"]
assert list(inspect.signature(D4RT.forward).parameters) == ["s", "x", "uv", "src", "ref"]
Td, st, info = ev.predict_q(m, x[:30], 12); assert Td.shape == (29, 4, 4) and torch.isfinite(Td).all()
assert torch.allclose(torch.linalg.det(Td[:, :3, :3]), torch.ones(29), atol=1e-4)  # proper rotations
# 8. the zero-initialised decoder predicts "no motion" (identity poses), the same starting point as the pose-regression baseline
m0 = D4RT(Net(dim=64, depth=2, heads=2), S, dim=64, depth=2, heads=4).eval()
Td0 = ev.predict_q(m0, x[:30], 12)[0]; assert (Td0 - torch.eye(4)).abs().max() < 1e-4
print("test_query ok")
