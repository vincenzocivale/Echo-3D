"""Multi-offset pose-graph fusion (Phase 2, minimal): fuse the network's direct estimates Delta_{t,t+k} (k in offsets) into one trajectory.
Variables are the adjacent steps xi_t (so the problem is local and well conditioned along long scans); residual for each measurement
Z_{t,t+k}: vee(Z^-1 @ step_t @ ... @ step_{t+k-1}) whitened by the per-axis error std sigma[k] (estimated on training scans, never on dev).
Overlapping measurements share image content, so their errors are correlated: the weights are a heuristic, not a calibrated likelihood."""
import torch
from ..geometry import se3, calib


def _vee(E):  # small-error SE(3) residual: (rotation vee, translation); smooth everywhere, ~= log(E) near identity
    R = E[..., :3, :3]
    return torch.cat([torch.stack([R[..., 2, 1] - R[..., 1, 2], R[..., 0, 2] - R[..., 2, 0], R[..., 1, 0] - R[..., 0, 1]], -1) / 2, E[..., :3, 3]], -1)


def products(steps, offsets):
    """steps [m,4,4] -> {k: [m-k+1,4,4] products of k consecutive steps} for power-of-two offsets (doubling)."""
    P, out, k = steps, {}, 1
    while k <= max(offsets):
        if k in offsets: out[k] = P
        P, k = P[:-k] @ P[k:], 2 * k
    return out


def fuse(meas, sigma, iters=100):
    """meas {k: [n-k,4,4] predicted Delta_{t,t+k}} (must contain k=1), sigma {k: [6] error std per axis} -> T_ref_from_plane [n,4,4]."""
    offs = sorted(meas)
    assert all(k & (k - 1) == 0 for k in offs) and offs[0] == 1, "offsets must be powers of two including 1"
    Zinv = {k: se3.inverse(meas[k].double()) for k in offs}
    w = {k: 1 / sigma[k].double().to(meas[1].device) for k in offs}
    xi = se3.log(meas[1].double()).clone().requires_grad_(True)
    opt = torch.optim.LBFGS([xi], lr=1, max_iter=iters, tolerance_grad=1e-12, tolerance_change=1e-14, history_size=50, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        P = products(se3.exp(xi), offs)
        loss = sum(((_vee(Zinv[k] @ P[k]) * w[k]) ** 2).sum() for k in offs)
        loss.backward()
        return loss
    opt.step(closure)
    return calib.sequential(se3.exp(xi.detach()))
