"""Pose <-> metric points <-> the 4 TUS-REC displacement fields (GP/GL/LP/LL), all in mm.
Convention (baseline utils/transform.py): T_i->j(mm) = C^-1 inv(T_j) T_i C, C = image_mm->tool, tforms = tool->world."""
import csv, torch

H, W = 480, 640


def load_calib(path):
    """-> S (pixel->mm, 4x4), C (image_mm->tool, 4x4)"""
    r = list(csv.reader(open(path)))
    f = lambda a, b: torch.tensor([[float(x) for x in row] for row in r[a:b]])
    return f(1, 5), f(6, 10)


def pixel_grid(S, h=H, w=W):
    """Homogeneous 4xN pixel coords (1-based, x fastest, z=0) mapped to mm."""
    y, x = torch.meshgrid(torch.arange(1., h + 1, device=S.device), torch.arange(1., w + 1, device=S.device), indexing="ij")
    return S @ torch.stack([x.flatten(), y.flatten(), torch.zeros(h * w, device=S.device), torch.ones(h * w, device=S.device)])


def rel(T, C, src, dst):
    """Rigid transform image_mm(src) -> image_mm(dst) for index tensors src,dst; T: [N,4,4] tool->world."""
    T, C = T.double(), C.double()  # tracker translations are ~100s of mm: float32 loses ~1e-3
    return (torch.linalg.inv(C) @ torch.linalg.inv(T[dst]) @ T[src] @ C).float()


def ddf(Tij, P):
    """Displacement (mm) of points P [4,N] under transforms Tij [M,4,4] -> [M,3,N]."""
    return (Tij @ P)[:, :3] - P[:3]


def four_ddf(T, C, S, landmark, P=None):
    """GP, GL, LP, LL from tforms T [N,4,4] and landmark [K,3] (0-based frame, x, y pixels).
    GP/LP: [N-1,3,H*W] for frames 1..N-1; GL/LL: [K,3]."""
    P = pixel_grid(S) if P is None else P
    n = len(T); f = torch.arange(1, n); z = torch.zeros(len(f), dtype=torch.long)
    L = landmark.float(); li = L[:, 0].long()
    Lp = S @ torch.stack([L[:, 1], L[:, 2], torch.zeros(len(L)), torch.ones(len(L))])
    gl = ddf(rel(T, C, li, torch.zeros_like(li)), Lp)  # [K,3,K]; keep each landmark's own transform
    ll = ddf(rel(T, C, li, li - 1), Lp)
    k = torch.arange(len(L))
    return ddf(rel(T, C, f, z), P), gl[k, :, k], ddf(rel(T, C, f, f - 1), P), ll[k, :, k]


def uv_to_mm(uv, S, h=H, w=W):
    """uv [...,2] in [0,1] over the full image extent (pixel centres at (x-.5)/w, x in 1..w as in pixel_grid) -> [...,3] mm, z=0."""
    x, y = uv[..., 0] * w + 0.5, uv[..., 1] * h + 0.5
    return (torch.stack([x, y, torch.zeros_like(x), torch.ones_like(x)], -1) @ S.T.to(uv))[..., :3]


def query_gt(Tg, S, uv, src, ref):
    """Tg [B,L,4,4] = T_{i->0} (image_mm->image_mm of the clip's frame 0); uv [B,Q,2], src/ref [B,Q] long.
    Returns the 3D position (mm) of source pixel (u,v) in the image-mm coordinate system of frame `ref`:
    P = T_{src->ref} p = inv(T_{ref->0}) T_{src->0} p  (the calibration is already inside Tg via rel())."""
    b = torch.arange(len(Tg), device=Tg.device)[:, None]
    Tsr = torch.linalg.inv(Tg[b, ref].double()) @ Tg[b, src].double()
    p = torch.cat([uv_to_mm(uv.double(), S.double()), torch.ones_like(uv[..., :1]).double()], -1)
    return (Tsr @ p[..., None])[..., :3, 0].to(uv.dtype)


def sample_q(B, Q, L, gen, dev, id_frac=.1):
    """Training/eval queries: uv uniform over the whole image, t_src uniform, |t_ref - t_src| log-uniform in [1,L-1] (random sign,
    reflected/clamped into the clip), a fraction id_frac of identity queries (t_ref = t_src)."""
    import math
    r = lambda *s: torch.rand(*s, generator=gen, device=dev)
    uv, src = r(B, Q, 2), torch.randint(0, L, (B, Q), generator=gen, device=dev)
    dt = torch.exp(r(B, Q) * math.log(L - 1)).round().long().clamp(1, L - 1)
    sg = torch.randint(0, 2, (B, Q), generator=gen, device=dev) * 2 - 1
    ref = torch.where(((src + sg * dt) < 0) | ((src + sg * dt) >= L), src - sg * dt, src + sg * dt).clamp(0, L - 1)
    return uv, src, torch.where(r(B, Q) < id_frac, src, ref)


def kabsch(p, q):
    """Rigid fit (no scale): R,t minimising sum|R p + t - q|^2, det R = +1. p,q [...,N,3] -> R, t, singular values of the
    covariance (s[1]/s[0] ~ 0 flags a degenerate / collinear fit), rms residual. Planar p (rank-2 covariance) is fine."""
    p, q = p.double(), q.double()
    pc, qc = p.mean(-2, keepdim=True), q.mean(-2, keepdim=True)
    U, s, Vh = torch.linalg.svd((p - pc).transpose(-1, -2) @ (q - qc))
    V = Vh.transpose(-1, -2)
    d = torch.sign(torch.linalg.det(V @ U.transpose(-1, -2)))
    D = torch.diag_embed(torch.stack([torch.ones_like(d), torch.ones_like(d), d], -1))
    R = V @ D @ U.transpose(-1, -2)
    t = qc[..., 0, :] - (R @ pc[..., 0, :, None])[..., 0]
    return R, t, s, ((p @ R.transpose(-1, -2) + t[..., None, :] - q) ** 2).sum(-1).mean(-1).sqrt()


def Rt_to_T(R, t):
    T = torch.eye(4, dtype=R.dtype, device=R.device).expand(*R.shape[:-2], 4, 4).clone()
    T[..., :3, :3], T[..., :3, 3] = R, t
    return T


def pose_to_T(p):
    """p [...,9] = 6D rotation (Zhou et al.) + translation (mm) -> [...,4,4]. Identity at p=(1,0,0,0,1,0,0,0,0)."""
    a, b = p[..., :3], p[..., 3:6]
    e1 = torch.nn.functional.normalize(a, dim=-1)
    e2 = torch.nn.functional.normalize(b - (e1 * b).sum(-1, keepdim=True) * e1, dim=-1)
    R = torch.stack([e1, e2, torch.cross(e1, e2, dim=-1)], -1)
    T = torch.eye(4, device=p.device, dtype=p.dtype).expand(*p.shape[:-1], 4, 4).clone()
    T[..., :3, :3], T[..., :3, 3] = R, p[..., 6:9]
    return T


def T_to_pose(T):
    return torch.cat([T[..., :3, 0], T[..., :3, 1], T[..., :3, 3]], -1)


if __name__ == "__main__":  # self-check: composition, identity, pose round-trip
    torch.manual_seed(0)
    S, C = load_calib("/raid/DATASETS/TUS-REC2024/calib_matrix.csv")
    q = torch.linalg.qr(torch.randn(8, 3, 3))[0]
    q = q * torch.linalg.det(q)[:, None, None].sign()
    T = torch.eye(4).repeat(8, 1, 1); T[:, :3, :3] = q; T[:, :3, 3] = torch.randn(8, 3) * 10
    i, j, k = torch.tensor([5]), torch.tensor([2]), torch.tensor([0])
    assert torch.allclose(rel(T, C, i, k), rel(T, C, j, k) @ rel(T, C, i, j), atol=1e-4)  # chain rule
    assert torch.allclose(rel(T, C, i, i)[0], torch.eye(4), atol=1e-4)
    assert torch.allclose(pose_to_T(T_to_pose(rel(T, C, i, k))), rel(T, C, i, k), atol=1e-4)
    G = torch.eye(4); G[:3, :3] = q[0]; G[:3, 3] = torch.tensor([50., -20, 300])  # arbitrary world-frame (gauge) change
    assert torch.allclose(rel(G @ T, C, i, k), rel(T, C, i, k), atol=1e-3)  # relative poses are gauge-free
    lm = torch.tensor([[3, 100., 200.], [7, 50., 60.]])
    gp, gl, lp, ll = four_ddf(T, C, S, lm)
    assert gp.shape == (7, 3, H * W) and gl.shape == (2, 3) and lp.shape == gp.shape and ll.shape == (2, 3)
    print("geom ok")
