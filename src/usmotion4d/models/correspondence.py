"""CorrespondenceMatcher (principle of CAT, MICCAI 2026; own implementation, not a reproduction): bidirectionally normalised token similarity.
Token displacements are only *visual evidence* for the pose, never a 3D-3D rigid constraint: two B-mode planes need not share anatomical points."""
import math, torch, torch.nn as nn, torch.nn.functional as F


class CorrespondenceMatcher(nn.Module):
    def __init__(self, sinkhorn_iters=3, tau_init=0.07):
        super().__init__()
        self.iters, self.log_scale = sinkhorn_iters, nn.Parameter(torch.tensor(math.log(1 / tau_init)))

    def forward(self, za, zb, pos, mask_a=None, mask_b=None):
        """za, zb [B,P,D]; pos [P,2] token centres (mm); masks [B,P] bool (token inside the acquired sector).
        Returns dict: aligned [B,P,D] (frame-b features aligned to a), disp [B,P,2] mm (a -> b), conf [B,P] (1-H_norm)*mutual,
        entropy, mutual, P_ab [B,P,P], P_ba [B,P,P]."""
        B, P, _ = za.shape
        S = self.log_scale.exp().clamp(max=100) * F.normalize(za.float(), dim=-1) @ F.normalize(zb.float(), dim=-1).transpose(1, 2)
        neg = -1e4
        if mask_b is not None: S = S.masked_fill(~mask_b[:, None, :], neg)
        if mask_a is not None: S = S.masked_fill(~mask_a[:, :, None], neg)
        L = S
        for _ in range(self.iters):  # bidirectional normalisation: alternate row / column log-normalisation
            L = L - L.logsumexp(2, keepdim=True)
            L = L - L.logsumexp(1, keepdim=True)
        Pab = (L - L.logsumexp(2, keepdim=True)).exp()  # each a-token: distribution over b-tokens
        Pba = (L - L.logsumexp(1, keepdim=True)).exp().transpose(1, 2)  # each b-token: distribution over a-tokens, [B,Pb,Pa]
        aligned = Pab.to(zb.dtype) @ zb
        disp = Pab @ pos.to(Pab) - pos.to(Pab)
        n = mask_b.sum(1).clamp(min=2).float() if mask_b is not None else torch.full((B,), float(P), device=za.device)
        ent = -(Pab * Pab.clamp_min(1e-12).log()).sum(-1) / n.log()[:, None]
        mutual = (Pab * Pba.transpose(1, 2)).sum(-1)  # prob. that a -> b -> a returns to the same token
        conf = ((1 - ent).clamp(0, 1) * mutual.clamp(0, 1)).clamp_min(1e-4)
        if mask_a is not None: conf = conf * mask_a
        return dict(aligned=aligned, disp=disp, conf=conf, entropy=ent, mutual=mutual, P_ab=Pab, P_ba=Pba)
