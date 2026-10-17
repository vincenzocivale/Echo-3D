import torch
from ..geometry import se3


def pose_error(xi_pred, delta_gt):
    """Log(inverse(Delta_GT) @ Delta_pred) -> (rot [B,3] rad, trans [B,3] mm)."""
    e = se3.log(se3.inverse(delta_gt) @ se3.exp(xi_pred))
    return e[:, :3], e[:, 3:]


def huber(r, delta=1.0):
    return torch.where(r < delta, 0.5 * r ** 2, delta * (r - 0.5 * delta))


def pose_loss(xi_pred, delta_gt, k, sigma_rot=0.002, sigma_trans=0.4, nll_logvar=None, parts="both"):
    """Translation and rotation are normalised separately by their typical per-frame-step size times the offset k, then Huber on the norm;
    both terms have unit weight (the scales, not an ad-hoc lambda, set the balance). nll_logvar [B,2]: optional heteroscedastic NLL."""
    er, et = pose_error(xi_pred, delta_gt)
    nr, nt = er.norm(dim=-1) / (sigma_rot * k), et.norm(dim=-1) / (sigma_trans * k)
    lr, lt = huber(nr), huber(nt)
    if nll_logvar is not None:  # N(0, exp(logvar)) on the normalised errors, logvar clamped for stability
        lv = nll_logvar.clamp(-6, 6)
        lr, lt = 0.5 * (nr ** 2 * (-lv[:, 0]).exp() + lv[:, 0]), 0.5 * (nt ** 2 * (-lv[:, 1]).exp() + lv[:, 1])
    if parts != "both": lr, lt = (lr, lt * 0) if parts == "rot" else (lr * 0, lt)  # diagnostic: single component
    return (lr + lt).mean(), dict(rot=lr.mean().item(), trans=lt.mean().item(), rot_mrad=(er.norm(dim=-1)).mean().item() * 1e3, trans_mm=et.norm(dim=-1).mean().item())
