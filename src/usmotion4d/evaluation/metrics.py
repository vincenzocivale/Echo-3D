"""Official TUS-REC2024 metrics in mm (definitions of the baseline utils/evaluation.py, verified in tests/test_official.py).
GPE/GLE: displacement of pixels/landmarks mapped to frame 0; LPE/LLE: mapped to the previous frame. No scale alignment."""
import torch
from ..geometry import se3, calib


def _disp(T, P):  # T [M,4,4] plane_dst_from_plane_src, P [4,N] (mm) -> displacement [M,3,N]
    return (T @ P)[:, :3] - P[:3]


@torch.no_grad()
def tusrec_metrics(T_pred, T_gt, S, landmarks=None, chunk=32):
    """T_pred, T_gt: T_ref_from_plane [n,4,4] (frame 0 = identity). S: official 4x4 pixel->mm. landmarks [K,3] = (0-based frame, x, y px).
    Local transforms are the adjacent Delta_{i-1,i}. Returns dict of floats (mm)."""
    dev = T_pred.device
    S, Tp, Tg = S.to(dev).double(), T_pred.double(), T_gt.double()
    yy, xx = torch.meshgrid(torch.arange(1., calib.H + 1, device=dev, dtype=torch.float64), torch.arange(1., calib.W + 1, device=dev, dtype=torch.float64), indexing="ij")
    P = S @ torch.stack([xx.flatten(), yy.flatten(), torch.zeros(calib.H * calib.W, device=dev, dtype=torch.float64), torch.ones(calib.H * calib.W, device=dev, dtype=torch.float64)])
    loc = lambda T: se3.inverse(T[:-1]) @ T[1:]  # Delta_{i-1,i}
    Lp, Lg = loc(Tp), loc(Tg)
    pix = lambda A, B: torch.cat([(_disp(A[i:i + chunk], P) - _disp(B[i:i + chunk], P)).norm(dim=1).mean(1) for i in range(0, len(A), chunk)]).mean()
    out = dict(GPE=pix(Tp[1:], Tg[1:]).item(), LPE=pix(Lp, Lg).item())
    if landmarks is not None and len(landmarks):
        L = landmarks.to(dev).double(); li = L[:, 0].long()
        pts = S @ torch.stack([L[:, 1], L[:, 2], torch.zeros_like(L[:, 0]), torch.ones_like(L[:, 0])])  # [4,K]
        lm = lambda A, B: torch.einsum("kij,jk->ki", (A - B)[:, :3], pts).norm(dim=1).mean().item()
        out["GLE"] = lm(Tp[li], Tg[li])  # frame 0 -> identity, T_{0<-l} = Tp[l]
        out["LLE"] = lm(Lp[li - 1], Lg[li - 1])
    return out
