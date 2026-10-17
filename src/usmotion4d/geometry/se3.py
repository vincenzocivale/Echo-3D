"""SE(3) with a Lie-algebra parametrisation. xi = (omega, v) in R^6; T = exp(xi) has R = exp([omega]x), t = V(omega) v.
Convention everywhere: T_A_from_B maps points expressed in B to coordinates in A."""
import torch

_EPS = 1e-4


def hat(w):
    z = torch.zeros_like(w[..., 0])
    return torch.stack([z, -w[..., 2], w[..., 1], w[..., 2], z, -w[..., 0], -w[..., 1], w[..., 0], z], -1).view(*w.shape[:-1], 3, 3)


def _coeffs(th2):
    """sin(th)/th, (1-cos th)/th^2, (th-sin th)/th^3 with Taylor series near 0 (stable for the ~1e-3 rad rotations of this task)."""
    small = th2 < _EPS
    th = torch.sqrt(th2.clamp_min(1e-30))
    a = torch.where(small, 1 - th2 / 6 + th2 ** 2 / 120, torch.sin(th) / th)
    b = torch.where(small, 0.5 - th2 / 24 + th2 ** 2 / 720, (1 - torch.cos(th)) / th2.clamp_min(1e-30))
    c = torch.where(small, 1 / 6 - th2 / 120 + th2 ** 2 / 5040, (th - torch.sin(th)) / (th * th2).clamp_min(1e-30))
    return a, b, c


def so3_exp(w):
    th2 = (w * w).sum(-1)[..., None, None]
    a, b, _ = _coeffs(th2)
    K = hat(w)
    return torch.eye(3, dtype=w.dtype, device=w.device) + a * K + b * (K @ K)


def so3_log(R):
    cos = ((R.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2).clamp(-1, 1)
    vee = torch.stack([R[..., 2, 1] - R[..., 1, 2], R[..., 0, 2] - R[..., 2, 0], R[..., 1, 0] - R[..., 0, 1]], -1) / 2  # = sin(th) * axis
    sin = vee.norm(dim=-1)
    th = torch.atan2(sin, cos)  # accurate for small and moderate angles
    k = torch.where(sin < 1e-6, 1 + th ** 2 / 6, th / sin.clamp_min(1e-30))
    return vee * k[..., None]  # ponytail: angles within ~1e-6 of pi are not handled (never occurs for consecutive freehand frames)


def exp(xi):
    w, v = xi[..., :3], xi[..., 3:]
    th2 = (w * w).sum(-1)[..., None, None]
    _, b, c = _coeffs(th2)
    K = hat(w)
    V = torch.eye(3, dtype=xi.dtype, device=xi.device) + b * K + c * (K @ K)
    T = torch.eye(4, dtype=xi.dtype, device=xi.device).expand(*xi.shape[:-1], 4, 4).clone()
    T[..., :3, :3], T[..., :3, 3] = so3_exp(w), (V @ v[..., None])[..., 0]
    return T


def log(T):
    w = so3_log(T[..., :3, :3])
    th2 = (w * w).sum(-1)[..., None, None]
    _, b, _ = _coeffs(th2)
    a, _, _ = _coeffs(th2)
    K = hat(w)
    # V^-1 = I - K/2 + (1 - a/(2b)) / th^2 * K^2 ; coefficient -> 1/12 as th -> 0
    coef = torch.where(th2 < _EPS, 1 / 12 + th2 / 720, (1 - a / (2 * b)) / th2.clamp_min(1e-30))
    Vinv = torch.eye(3, dtype=T.dtype, device=T.device) - K / 2 + coef * (K @ K)
    return torch.cat([w, (Vinv @ T[..., :3, 3, None])[..., 0]], -1)


def inverse(T):
    R, t = T[..., :3, :3], T[..., :3, 3]
    Ti = torch.eye(4, dtype=T.dtype, device=T.device).expand(*T.shape[:-2], 4, 4).clone()
    Ti[..., :3, :3] = R.transpose(-1, -2)
    Ti[..., :3, 3] = -(R.transpose(-1, -2) @ t[..., None])[..., 0]
    return Ti


def compose(*Ts):
    out = Ts[0]
    for T in Ts[1:]: out = out @ T
    return out
