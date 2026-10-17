"""Relative-pose estimator for ablations A0..A3 on a frozen USFM encoder (identical code path; flags only):
A0 joint token transformer over both frames, no explicit matching | A1 + CorrespondenceMatcher | A2 + match confidence in the pooling
| A3 + learned PoseSupportHead. Output: xi in se(3) (omega [rad], v [mm]) of Delta_ij = T_plane_i_from_plane_j, optional (log-var rot, log-var trans)."""
import math, torch, torch.nn as nn
from ..geometry import se3, calib
from .correspondence import CorrespondenceMatcher
from .pose_support import PoseSupportHead

ARMS = dict(A0=dict(match=False, conf=False, support=False), A1=dict(match=True, conf=False, support=False),
            A2=dict(match=True, conf=True, support=False), A3=dict(match=True, conf=True, support=True))


def token_geometry(S, size, valid_mask, patch=16):
    """Token centres in plane mm [P,2] and token validity [P] (>50% of the patch inside the acquired sector), for frames resized to `size`."""
    g = calib.PixelCalib.from_official(S).resize(*size).grid_mm()
    gh, gw = size[0] // patch, size[1] // patch
    pos = g.reshape(gh, patch, gw, patch, 2).mean((1, 3)).reshape(-1, 2).float()
    m = torch.nn.functional.interpolate(valid_mask[None, None].float(), size, mode="area")[0, 0]
    return pos, (m.reshape(gh, patch, gw, patch).mean((1, 3)) > .5).reshape(-1)


class PoseEstimator(nn.Module):
    def __init__(self, encoder, S, valid_mask, arm="A3", dim=256, depth=2, heads=8, in_ch=None, sinkhorn_iters=3,
                 scale_rot=0.002, scale_trans=0.4, pool="attn", cells=(4, 4)):
        super().__init__()
        self.cfg, self.arm, self.encoder, self.pool, self.cells = ARMS[arm], arm, encoder, pool, cells
        assert pool in ("attn", "mean", "spatial")
        assert pool == "attn" or arm == "A1", "mean/spatial pooling are defined for A1 only (A2/A3 modulate the attention logits)"
        pos, m = token_geometry(S, encoder.size, valid_mask)
        self.register_buffer("pos", pos); self.register_buffer("tmask", m)
        self.register_buffer("scale", torch.tensor([scale_rot] * 3 + [scale_trans] * 3))  # per-frame-step magnitudes; xi_hat * scale * k
        c = in_ch or encoder.channels * len(encoder.layers)
        self.adapt = nn.Sequential(nn.Linear(c, dim), nn.LayerNorm(dim))
        self.pos_emb = nn.Sequential(nn.Linear(2, dim), nn.GELU(), nn.Linear(dim, dim))
        if self.cfg["match"]:
            self.matcher = CorrespondenceMatcher(sinkhorn_iters)
            self.mix = nn.Sequential(nn.Linear(3 * dim + 2, dim), nn.GELU(), nn.Linear(dim, dim))
        else:
            self.seg = nn.Parameter(torch.zeros(2, dim)); nn.init.normal_(self.seg, std=.02)
        if self.cfg["support"]: self.support = PoseSupportHead(dim)
        self.tf = nn.TransformerEncoder(nn.TransformerEncoderLayer(dim, heads, 4 * dim, 0., "gelu", batch_first=True, norm_first=True), depth, enable_nested_tensor=False)
        self.att = nn.Linear(dim, 1)
        d_in = dim * cells[0] * cells[1] if pool == "spatial" else dim
        self.head = nn.Sequential(nn.LayerNorm(d_in), nn.Linear(d_in, dim), nn.GELU(), nn.Linear(dim, 8))
        nn.init.zeros_(self.head[-1].weight); nn.init.zeros_(self.head[-1].bias)  # identity at init, unit variance

    def trainable_parameters(self): return [p for p in self.parameters() if p.requires_grad]

    @torch.no_grad()
    def encode(self, x):
        """x [N,1,h,w] -> frozen-encoder tokens [N,P,C*len(layers)] (cacheable per scan; only images are used)."""
        maps = self.encoder(x)
        return torch.cat([m.flatten(2).transpose(1, 2) for m in maps], -1)

    def forward(self, fa, fb, k=None):
        """fa, fb [B,P,C] encoder tokens of frames i and j -> predicts Delta_ij. k [B] frame offset (scales xi); defaults to 1."""
        B, P, _ = fa.shape
        m = self.tmask[None].expand(B, -1)
        za, zb = self.adapt(fa.float()), self.adapt(fb.float())
        pe = self.pos_emb(self.pos / 100)[None]
        info = {}
        if self.cfg["match"]:
            c = self.matcher(za, zb, self.pos, m, m)
            tok = self.mix(torch.cat([za, c["aligned"], za - c["aligned"], c["disp"] / 10], -1)) + pe
            info.update(match_confidence=c["conf"], disp=c["disp"], P_ab=c["P_ab"])
            logit_pad = ~m
        else:
            tok = torch.cat([za + pe + self.seg[0], zb + pe + self.seg[1]], 1)
            logit_pad = ~torch.cat([m, m], 1)
        h = self.tf(tok, src_key_padding_mask=logit_pad)
        logit = self.att(h)[..., 0]
        if self.cfg["conf"]: logit = logit + info["match_confidence"].log()
        if self.cfg["support"]:
            sup = self.support(tok, info["match_confidence"].detach())
            logit = logit + sup["pose_support"].clamp_min(1e-4).log(); info["pose_support"] = sup["pose_support"]
        if self.pool == "attn":
            w = logit.masked_fill(logit_pad, -1e4).softmax(1)
            pooled = (w[..., None] * h).sum(1)
        elif self.pool == "mean":  # masked mean over valid tokens
            w = m.float() / m.float().sum(1, keepdim=True)
            pooled = (w[..., None] * h).sum(1)
        else:  # spatial: masked mean inside each cell of a cells[0] x cells[1] grid, cells concatenated (keeps where things happen)
            gh, gw = self.encoder.grid; ch, cw = self.cells; th, tw = gh // ch, gw // cw
            mf = m.float().view(B, ch, th, cw, tw)
            cnt = mf.sum((2, 4))  # [B,ch,cw] valid tokens per cell
            hc = (h.view(B, ch, th, cw, tw, -1) * mf[..., None]).sum((2, 4)) / cnt.clamp_min(1)[..., None]
            pooled = hc.flatten(1)
            w = (mf / cnt.clamp_min(1)[:, :, None, :, None] / (cnt > 0).sum((1, 2)).clamp_min(1)[:, None, None, None, None]).view(B, -1)  # equivalent per-token weight
        o = self.head(pooled)
        k = torch.ones(B, device=fa.device) if k is None else k.float()
        xi = o[:, :6].float() * self.scale * k[:, None]
        return dict(xi=xi, delta=se3.exp(xi), logvar=o[:, 6:8].float(), weights=w, **info)
