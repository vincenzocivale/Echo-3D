"""Spatio-temporal ViT -> per-step rigid deltas T_{i->i-1} (mm). `window` is the ONLY local-vs-global knob:
window=None -> every token attends to the whole clip (global); window=w -> only |t-t'|<=w (local).
A token t covers frames (t,t+1), so delta t sees frames t-w..t+1+w; window=0 means no cross-token context."""
import math, torch, torch.nn as nn
from geom import pose_to_T, uv_to_mm

ID = torch.tensor([1., 0, 0, 0, 1, 0, 0, 0, 0])  # 6D identity rotation + zero translation


class Net(nn.Module):
    def __init__(s, dim=384, depth=8, heads=6, patch=16, size=(120, 160), window=None):
        super().__init__()
        s.window, s.dim = window, dim
        s.embed = nn.Conv3d(1, dim, (2, patch, patch), stride=(1, patch, patch))  # each token sees a frame pair
        s.pos = nn.Parameter(torch.zeros(1, 1, (size[0] // patch) * (size[1] // patch), dim))
        s.enc = nn.TransformerEncoder(nn.TransformerEncoderLayer(dim, heads, 4 * dim, 0., "gelu", batch_first=True, norm_first=True), depth, enable_nested_tensor=False)
        s.head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 9))
        nn.init.zeros_(s.head[1].weight); nn.init.zeros_(s.head[1].bias)  # starts at identity
        nn.init.trunc_normal_(s.pos, std=.02)

    def tpe(s, T, device):  # sinusoidal time embedding [T,dim]
        f = torch.exp(-math.log(1e4) * torch.arange(0, s.dim, 2, device=device) / s.dim)
        a = torch.arange(T, device=device)[:, None] * f
        return torch.cat([a.sin(), a.cos()], -1)

    def encode(s, x):  # x [B,L,1,h,w] in [0,1] -> scene features [B,T=L-1,P,D] (reusable by a query decoder)
        z = s.embed(x.transpose(1, 2))  # [B,D,T,gh,gw]
        B, D, T = z.shape[:3]
        P = z.shape[3] * z.shape[4]
        z = z.flatten(3).permute(0, 2, 3, 1) + s.pos + s.tpe(T, x.device)[None, :, None]  # [B,T,P,D]
        mask = None
        if s.window is not None:  # True = blocked
            t = torch.arange(T, device=x.device).repeat_interleave(P)
            mask = (t[:, None] - t[None]).abs() > s.window
        return s.enc(z.flatten(1, 2), mask=mask).view(B, T, P, D)

    def forward(s, x):  # -> deltas [B,L-1,4,4]
        z = s.encode(x).mean(2)
        with torch.autocast(z.device.type, enabled=False):  # fp32 head: bf16 can't resolve ~1e-3 rad deviations from identity
            return pose_to_T(s.head(z.float()) + ID.to(z.device))


def tsin(t, dim=32):  # sinusoidal embedding of (signed) frame indices/offsets [...] -> [...,dim]
    f = torch.exp(-math.log(1e3) * torch.arange(0, dim, 2, device=t.device) / dim)
    a = t[..., None].float() * f
    return torch.cat([a.sin(), a.cos()], -1)


def crop(x, uv, src, k=9):
    """x [B,L,1,h,w], uv [B,Q,2] in [0,1], src [B,Q] -> [B,Q,k*k] B-mode patch of frame src around (u,v) (nearest, border-clamped)."""
    B, L, _, h, w = x.shape
    cx, cy = (uv[..., 0] * w - .5).round().long(), (uv[..., 1] * h - .5).round().long()
    o = torch.arange(k, device=x.device) - k // 2
    ix = (cx[..., None, None] + o[None, None, None, :]).clamp(0, w - 1)  # [B,Q,1,k]
    iy = (cy[..., None, None] + o[None, None, :, None]).clamp(0, h - 1)  # [B,Q,k,1]
    return x[torch.arange(B, device=x.device)[:, None, None, None], src[..., None, None], 0, iy, ix].flatten(-2)


class XBlock(nn.Module):  # cross-attention + MLP only: queries never attend to each other
    def __init__(s, dim, heads):
        super().__init__()
        s.n1, s.n2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        s.att = nn.MultiheadAttention(dim, heads, batch_first=True)
        s.mlp = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim))

    def forward(s, q, kv):
        q = q + s.att(s.n1(q), kv, kv, need_weights=False)[0]
        return q + s.mlp(s.n2(q))


class PointQueryDecoder(nn.Module):
    """D4RT-style: query (u,v,t_src,t_ref) [+ 9x9 patch] cross-attends to the encoder features -> offset (units of 10 mm)."""

    def __init__(s, enc_dim=384, dim=256, depth=4, heads=8, nfreq=8, patch=9):
        super().__init__()
        s.nfreq = nfreq
        s.kv = nn.Sequential(nn.Linear(enc_dim, dim), nn.LayerNorm(dim))
        s.e_uv, s.e_t, s.e_patch = nn.Linear(4 * nfreq, dim), nn.Linear(96, dim), nn.Linear(patch * patch, dim)
        s.blocks = nn.ModuleList(XBlock(dim, heads) for _ in range(depth))
        s.out = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 3))
        nn.init.zeros_(s.out[1].weight); nn.init.zeros_(s.out[1].bias)  # starts at "same position as in the source frame"

    def forward(s, feats, uv, src, ref, patches):  # feats [B,N,D_enc]; uv [B,Q,2]; src,ref [B,Q]; patches [B,Q,81]
        a = uv[..., None] * (2. ** torch.arange(s.nfreq, device=uv.device) * math.pi)  # Fourier features [B,Q,2,nf]
        q = s.e_uv(torch.cat([a.sin(), a.cos()], -1).flatten(-2)) + s.e_patch(patches.to(a))
        q = q + s.e_t(torch.cat([tsin(src), tsin(ref), tsin(ref - src)], -1))
        kv = s.kv(feats)
        for b in s.blocks:
            q = b(q, kv)
        with torch.autocast(q.device.type, enabled=False):
            return s.out(q.float())


class D4RT(nn.Module):
    """Encoder (same Net.encode as the pose-regression baseline) + point-query decoder. Output = 3D point (mm) of the
    source pixel in the image-mm frame of `ref`, predicted as (known source-frame position) + learned offset."""

    def __init__(s, enc, S, **kw):
        super().__init__()
        s.enc, s.dec = enc, PointQueryDecoder(enc.dim, **kw)
        s.register_buffer("S", S.float())

    def encode(s, x):  # -> [B,T*P,D]
        return s.enc.encode(x).flatten(1, 2)

    def decode(s, feats, x, uv, src, ref):
        return uv_to_mm(uv, s.S) + 10 * s.dec(feats, uv, src, ref, crop(x, uv, src))

    def forward(s, x, uv, src, ref):
        return s.decode(s.encode(x), x, uv, src, ref)


def accumulate(Td):
    """Deltas T_{i->i-1} [B,L-1,4,4] -> T_{i->0} [B,L,4,4] with T_{0->0}=I (chained product)."""
    out = [torch.eye(4, device=Td.device).expand(Td.shape[0], 4, 4)]
    for i in range(Td.shape[1]):
        out.append(out[-1] @ Td[:, i])
    return torch.stack(out, 1)


if __name__ == "__main__":  # self-check: identity init, shapes, window really limits the receptive field
    torch.manual_seed(0)
    x = torch.rand(1, 12, 1, 120, 160)
    for w in (None, 1):
        m = Net(dim=64, depth=2, heads=2, window=w).eval()
        nn.init.normal_(m.head[1].weight, std=.02)
        assert torch.allclose(Net(dim=64, depth=2, heads=2, window=w)(x), torch.eye(4).expand(1, 11, 4, 4), atol=1e-6)
        a = m(x); x2 = x.clone(); x2[:, -1] += 1  # perturb only the last frame
        d = (m(x2) - a).abs().amax((0, 2, 3))
        assert (d[0] > 0) == (w is None), (w, d)  # global: first delta changes; local: it does not
    assert accumulate(a).shape == (1, 12, 4, 4)
    print("model ok", sum(p.numel() for p in Net().parameters()) / 1e6, "M params")
