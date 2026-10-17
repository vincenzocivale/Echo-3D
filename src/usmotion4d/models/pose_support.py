"""PoseSupportHead (idea of the Flow4R pose weights). Output `pose_support` is a learned weight of each token in the probe-motion estimate,
trained only through the pose loss. It is NOT a calibrated probability of static tissue (TUS-REC has no rigidity labels) and it is kept
separate from `match_confidence` (reliability of the matching)."""
import torch, torch.nn as nn


class PoseSupportHead(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim // 2), nn.GELU(), nn.Linear(dim // 2, 1))
        nn.init.zeros_(self.net[-1].weight); nn.init.constant_(self.net[-1].bias, 0.)  # starts at support = 0.5 everywhere

    def forward(self, tok, match_confidence):
        """tok [B,P,D] per-token features of both frames + correspondence info; match_confidence [B,P] (detached: support must not game it)."""
        s = torch.sigmoid(self.net(tok)[..., 0])
        return dict(pose_support=s, match_confidence=match_confidence)


def support_diagnostics(w, mask=None, conf=None, err=None):
    """w [B,P] pooling weights (any non-negative per-token map). Flags degenerate solutions: uniform map, mass on very few tokens,
    confidence unrelated to the error. Returns floats; `eff_tokens` = exp(entropy) of the normalised weights."""
    w = w.float(); mask = torch.ones_like(w, dtype=torch.bool) if mask is None else mask
    p = (w * mask) / (w * mask).sum(1, keepdim=True).clamp_min(1e-12)
    n = mask.sum(1).float()
    eff = (-(p * p.clamp_min(1e-12).log()).sum(1)).exp()
    out = dict(eff_tokens=eff.mean().item(), eff_frac=(eff / n).mean().item(), top5_mass=p.topk(5, dim=1).values.sum(1).mean().item(),
               spatial_std=torch.stack([wi[m].std() for wi, m in zip(w, mask)]).mean().item())
    out["uniform"] = out["spatial_std"] < 1e-3 * w.mean().item()
    out["collapsed"] = out["eff_tokens"] < 3
    if conf is not None and err is not None and len(err) > 2:  # corr(mean confidence, pose error) over the batch
        c = torch.stack([ci[m].mean() for ci, m in zip(conf.float(), mask)])
        out["conf_err_corr"] = torch.corrcoef(torch.stack([c, err.float()]))[0, 1].item()
    return out
