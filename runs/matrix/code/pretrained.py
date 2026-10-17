"""Pretrained encoders for the feasibility study. Light re-implementations of USFM (BEiT-style ViT-B/16, shared 2D relative position
bias, layer scale) and VideoMAE-B (tubelet ViT-B, fixed sinusoidal positions), plus Net2D = per-frame spatial encoder + temporal
transformer with the same encode() interface as model.Net (so the D4RT decoder is unchanged).
Weights are never loaded silently: load_* returns an audit (loaded / adapted / checkpoint tensors not used / model tensors left random)."""
import math, torch, torch.nn as nn, torch.nn.functional as F

CK = "/raid/DATASETS/Echo3D_weights/"
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


# ---------------------------------------------------------------- audit
def load_with_audit(model, mapped, adapted, ckpt):
    """mapped: {model_key: tensor} to load strictly by shape; adapted: {key: note}; ckpt: full checkpoint dict (to list unused tensors)."""
    res = model.load_state_dict(mapped, strict=False)
    assert not res.unexpected_keys, res.unexpected_keys
    used = {k for k in ckpt if k in mapped or k in adapted.get("_src", ())}
    random_init = [k for k in model.state_dict() if k in res.missing_keys]
    return dict(loaded_params=sum(v.numel() for v in mapped.values()), loaded_tensors=len(mapped),
                adapted={k: v for k, v in adapted.items() if k != "_src"},
                ckpt_not_used={k: tuple(ckpt[k].shape) for k in ckpt if k not in used}, ckpt_not_used_params=sum(ckpt[k].numel() for k in ckpt if k not in used),
                random_init_params=sum(model.state_dict()[k].numel() for k in random_init), random_init_tensors=len(random_init),
                random_init_modules=sorted({".".join(k.split(".")[:2]) for k in random_init}),
                model_params=sum(p.numel() for p in model.parameters()))


# ---------------------------------------------------------------- USFM (BEiT-style ViT-B/16)
class Mlp(nn.Module):
    def __init__(s, d, r=4):
        super().__init__(); s.fc1, s.act, s.fc2 = nn.Linear(d, r * d), nn.GELU(), nn.Linear(r * d, d)

    def forward(s, x): return s.fc2(s.act(s.fc1(x)))


class Attn(nn.Module):  # fused qkv without k-bias (q_bias, v_bias), optional additive bias [1,H,N,N]
    def __init__(s, d, h, qk_bias=True):
        super().__init__()
        s.h = h; s.qkv = nn.Linear(d, 3 * d, bias=False); s.proj = nn.Linear(d, d)
        s.q_bias, s.v_bias = (nn.Parameter(torch.zeros(d)), nn.Parameter(torch.zeros(d))) if qk_bias else (None, None)

    def forward(s, x, bias=None):
        B, N, D = x.shape
        b = None if s.q_bias is None else torch.cat([s.q_bias, torch.zeros_like(s.v_bias), s.v_bias])
        q, k, v = F.linear(x, s.qkv.weight, b).view(B, N, 3, s.h, D // s.h).permute(2, 0, 3, 1, 4)
        return s.proj(F.scaled_dot_product_attention(q, k, v, attn_mask=bias).transpose(1, 2).reshape(B, N, D))


class VBlock(nn.Module):
    def __init__(s, d, h, init_values=.1):
        super().__init__()
        s.norm1, s.norm2 = nn.LayerNorm(d, eps=1e-6), nn.LayerNorm(d, eps=1e-6)
        s.attn, s.mlp = Attn(d, h), Mlp(d)
        s.gamma_1, s.gamma_2 = nn.Parameter(init_values * torch.ones(d)), nn.Parameter(init_values * torch.ones(d))

    def forward(s, x, bias):
        x = x + s.gamma_1 * s.attn(s.norm1(x), bias)
        return x + s.gamma_2 * s.mlp(s.norm2(x))


def relpos_index(gh, gw):  # as in the USFM repo: window tokens + cls (3 extra entries)
    c = torch.stack(torch.meshgrid(torch.arange(gh), torch.arange(gw), indexing="ij")).flatten(1)
    r = (c[:, :, None] - c[:, None, :]).permute(1, 2, 0).contiguous()
    r[:, :, 0] += gh - 1; r[:, :, 1] += gw - 1; r[:, :, 0] *= 2 * gw - 1
    nd = (2 * gh - 1) * (2 * gw - 1) + 3
    idx = torch.zeros((gh * gw + 1,) * 2, dtype=r.dtype)
    idx[1:, 1:] = r.sum(-1); idx[0, :] = nd - 3; idx[:, 0] = nd - 2; idx[0, 0] = nd - 1
    return idx


def interp_table(tab, gh, gw, src=(14, 14)):
    """[(2*sh-1)(2*sw-1)+3, H] -> same for (gh,gw): bicubic resize of the 2D offset table (BEiT practice); cls entries copied."""
    if (gh, gw) == src: return tab
    body = tab[:-3].view(2 * src[0] - 1, 2 * src[1] - 1, -1).permute(2, 0, 1)[None]
    body = F.interpolate(body.float(), size=(2 * gh - 1, 2 * gw - 1), mode="bicubic", align_corners=False)[0].permute(1, 2, 0).reshape(-1, tab.shape[1])
    return torch.cat([body.to(tab.dtype), tab[-3:]])


class RelPos(nn.Module):
    def __init__(s, gh, gw, h):
        super().__init__()
        s.relative_position_bias_table = nn.Parameter(torch.zeros((2 * gh - 1) * (2 * gw - 1) + 3, h))
        s.register_buffer("relative_position_index", relpos_index(gh, gw), persistent=False)

    def forward(s):
        n = s.relative_position_index.shape[0]
        return s.relative_position_bias_table[s.relative_position_index.view(-1)].view(n, n, -1).permute(2, 0, 1)[None]  # [1,H,N,N]


class FrameViT(nn.Module):
    """USFM backbone: gray frames [N,1,h,w] in [0,1] -> patch tokens [N,P,768] (final LN applied, cls token dropped)."""

    def __init__(s, grid=(7, 10), d=768, depth=12, h=12, patch=16):
        super().__init__()
        s.grid = grid
        s.patch_embed = nn.Module(); s.patch_embed.proj = nn.Conv2d(3, d, patch, patch)
        s.cls_token = nn.Parameter(torch.zeros(1, 1, d)); s.rel_pos_bias = RelPos(*grid, h)
        s.blocks = nn.ModuleList(VBlock(d, h) for _ in range(depth)); s.norm = nn.LayerNorm(d, eps=1e-6)
        s.register_buffer("mean", torch.tensor(MEAN).view(1, 3, 1, 1), persistent=False); s.register_buffer("std", torch.tensor(STD).view(1, 3, 1, 1), persistent=False)
        s.apply(s._init)
        nn.init.trunc_normal_(s.cls_token, std=.02)
        for i, b in enumerate(s.blocks):  # BEiT fix_init_weight
            b.attn.proj.weight.data.div_(math.sqrt(2. * (i + 1))); b.mlp.fc2.weight.data.div_(math.sqrt(2. * (i + 1)))

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            nn.init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None: nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight); nn.init.zeros_(m.bias)

    def forward(s, x):
        x = s.patch_embed.proj((x.expand(-1, 3, -1, -1) - s.mean) / s.std).flatten(2).transpose(1, 2)  # gray replicated, ImageNet norm
        assert x.shape[1] == s.grid[0] * s.grid[1], (x.shape, s.grid)
        x = torch.cat([s.cls_token.expand(len(x), -1, -1).to(x.dtype), x], 1)
        bias = s.rel_pos_bias().to(x.dtype)
        for b in s.blocks: x = b(x, bias)
        return s.norm(x)[:, 1:]


def load_usfm(model, path=CK + "usfm/USFM_latest.pth"):
    ck = {k: v for k, v in torch.load(path, map_location="cpu").items() if torch.is_tensor(v)}
    mapped, adapted = {}, {"_src": set()}
    for k, v in ck.items():
        if k in ("mask_token", "rel_pos_bias.relative_position_index"): continue  # MIM-only token / derived buffer
        if k.endswith("relative_position_bias_table"):
            v = interp_table(v, *model.grid)
            if tuple(model.grid) != (14, 14): adapted[k] = f"bicubic resize of the 27x27 offset table (+3 cls entries) for a {model.grid[0]}x{model.grid[1]} token grid"
        mapped[k] = v
    return load_with_audit(model, mapped, adapted, ck)


# ---------------------------------------------------------------- VideoMAE-B encoder (HF MCG-NJU/videomae-base)
def sincos(n, d):  # HF get_sinusoid_encoding_table: sin on even, cos on odd dims, flat token index
    p = torch.arange(n)[:, None] / torch.pow(10000, 2 * (torch.arange(d)[None] // 2) / d)
    p[:, 0::2], p[:, 1::2] = torch.sin(p[:, 0::2]), torch.cos(p[:, 1::2])
    return p[None].float()


class VBlockMAE(nn.Module):  # pre-LN, no layer scale, eps 1e-12 (HF config)
    def __init__(s, d, h):
        super().__init__()
        s.norm1, s.norm2 = nn.LayerNorm(d, eps=1e-12), nn.LayerNorm(d, eps=1e-12)
        s.attn, s.mlp = Attn(d, h), Mlp(d)

    def forward(s, x):
        x = x + s.attn(s.norm1(x))
        return x + s.mlp(s.norm2(x))


class VideoMAEEnc(nn.Module):
    """Joint space-time encoder: gray clip [B,L,1,h,w] -> tokens [B,T,P,768]; tubelet (2,16,16) with temporal stride `st` (2 = pretraining)."""

    def __init__(s, grid=(7, 10), d=768, depth=12, h=12, st=2):
        super().__init__()
        s.grid, s.st, s.dim = grid, st, d
        s.embed = nn.Conv3d(1, d, (2, 16, 16), (st, 16, 16))  # 1-channel: RGB weights collapsed exactly at load time
        s.blocks = nn.ModuleList(VBlockMAE(d, h) for _ in range(depth)); s.norm = nn.LayerNorm(d, eps=1e-12)
        s.register_buffer("mean", torch.tensor(MEAN).view(1, 1, 3, 1, 1), persistent=False)

    def encode(s, x):
        z = s.embed(x.transpose(1, 2)); B, D, T, gh, gw = z.shape
        z = z.flatten(2).transpose(1, 2)  # tokens ordered (t,h,w): the flat index the pretrained sinusoid was defined on
        z = z + sincos(z.shape[1], D).to(z)
        for b in s.blocks: z = b(z)
        return s.norm(z).view(B, T, gh * gw, D)


def load_videomae(model, path=CK + "videomae-base/model.safetensors"):
    from safetensors.torch import load_file
    ck = load_file(path); m, ad, src = {}, {}, set()
    P = "videomae.encoder.layer."
    w, b = ck["videomae.embeddings.patch_embeddings.projection.weight"], ck["videomae.embeddings.patch_embeddings.projection.bias"]
    s_ = torch.tensor(STD).view(1, 3, 1, 1, 1); mu = torch.tensor(MEAN).view(1, 3, 1, 1, 1)
    m["embed.weight"] = (w / s_).sum(1, keepdim=True); m["embed.bias"] = b - (w * mu / s_).sum((1, 2, 3, 4))
    ad["embed.weight"] = "3->1 channel: sum_c W_c/std_c (exact for a gray image replicated to RGB and ImageNet-normalised); bias absorbs the means"
    src |= {"videomae.embeddings.patch_embeddings.projection.weight", "videomae.embeddings.patch_embeddings.projection.bias"}
    for i in range(len(model.blocks)):
        p, q = f"{P}{i}.", f"blocks.{i}."
        m[q + "attn.qkv.weight"] = torch.cat([ck[p + f"attention.attention.{n}.weight"] for n in ("query", "key", "value")])
        for a_, b_ in (("attn.q_bias", "attention.attention.q_bias"), ("attn.v_bias", "attention.attention.v_bias"), ("attn.proj.weight", "attention.output.dense.weight"), ("attn.proj.bias", "attention.output.dense.bias"),
                       ("mlp.fc1.weight", "intermediate.dense.weight"), ("mlp.fc1.bias", "intermediate.dense.bias"), ("mlp.fc2.weight", "output.dense.weight"), ("mlp.fc2.bias", "output.dense.bias"),
                       ("norm1.weight", "layernorm_before.weight"), ("norm1.bias", "layernorm_before.bias"), ("norm2.weight", "layernorm_after.weight"), ("norm2.bias", "layernorm_after.bias")):
            m[q + a_] = ck[p + b_]
        src |= {k for k in ck if k.startswith(p)}
    m["norm.weight"], m["norm.bias"] = ck["videomae.layernorm.weight"], ck["videomae.layernorm.bias"]
    src |= {"videomae.layernorm.weight", "videomae.layernorm.bias"}
    return load_with_audit(model, m, {**ad, "_src": src}, ck)


# ---------------------------------------------------------------- per-frame encoder + temporal module
def _tpe(T, d, device):
    f = torch.exp(-math.log(1e4) * torch.arange(0, d, 2, device=device) / d)
    a = torch.arange(T, device=device)[:, None] * f
    return torch.cat([a.sin(), a.cos()], -1)


class Net2D(nn.Module):
    """Per-frame USFM ViT-B (spatial features kept, no pooling) -> Linear -> temporal transformer over all (frame, patch) tokens
    (window=None: global attention over the clip; window=w: |t-t'|<=w) -> features [B,L,P,dim] for the D4RT decoder."""

    def __init__(s, grid=(7, 10), dim=384, depth=4, heads=6, window=None, usfm=False):
        super().__init__()
        s.dim, s.window, s.grid = dim, window, grid
        s.backbone = FrameViT(grid); s.proj = nn.Linear(768, dim)
        s.pos = nn.Parameter(torch.zeros(1, 1, grid[0] * grid[1], dim)); nn.init.trunc_normal_(s.pos, std=.02)
        s.enc = nn.TransformerEncoder(nn.TransformerEncoderLayer(dim, heads, 4 * dim, 0., "gelu", batch_first=True, norm_first=True), depth, enable_nested_tensor=False)
        s.audit = load_usfm(s.backbone) if usfm else None  # everything else keeps its (seeded) random init in both arms

    def encode(s, x):  # x [B,L,1,h,w] in [0,1]
        B, L = x.shape[:2]; P = s.grid[0] * s.grid[1]
        z = s.proj(s.backbone(x.flatten(0, 1))).view(B, L, P, s.dim) + s.pos + _tpe(L, s.dim, x.device)[None, :, None]
        mask = None
        if s.window is not None:
            t = torch.arange(L, device=x.device).repeat_interleave(P); mask = (t[:, None] - t[None]).abs() > s.window
        return s.enc(z.flatten(1, 2), mask=mask).view(B, L, P, s.dim)


# ---------------------------------------------------------------- heads + builder for the 2x2 matrix (encoder in {usfm, videomae}) x (head in {pose, d4rt})
class PoseModel(nn.Module):
    """Direct pose regression on pooled encoder features -> T_{i->i-1} (6D rotation + translation, identity at init), same head as model.Net.
    per_frame: the encoder emits one feature set per frame (USFM-MV) -> deltas from consecutive frames; else one per frame pair (VideoMAE stride 1)."""

    def __init__(s, enc, per_frame):
        super().__init__()
        s.enc, s.per_frame = enc, per_frame
        d = enc.dim * (2 if per_frame else 1)
        s.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 9)); nn.init.zeros_(s.head[1].weight); nn.init.zeros_(s.head[1].bias)

    def forward(s, x):  # [B,L,1,h,w] -> deltas [B,L-1,4,4]
        from model import ID
        from geom import pose_to_T
        z = s.enc.encode(x).mean(2)
        if s.per_frame: z = torch.cat([z[:, :-1], z[:, 1:]], -1)
        with torch.autocast(z.device.type, enabled=False):
            return pose_to_T(s.head(z.float()) + ID.to(z.device))


def build(enc, head, S):
    """Encoder built from seed 0 (identical for both heads), head/decoder from a separate RNG stream so the data sampling stream is unaffected."""
    from model import D4RT
    torch.manual_seed(0)
    if enc == "usfm": e = Net2D(usfm=True); audit = e.audit
    else: e = VideoMAEEnc(grid=(7, 10), st=1); audit = load_videomae(e)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(1)
        m = D4RT(e, S, dim=256, heads=8) if head == "d4rt" else PoseModel(e, per_frame=enc == "usfm")
    m.audit = audit
    return m


def pretrained_params(model):
    """Parameters that come from the checkpoint (lower lr): the USFM backbone / the whole VideoMAE encoder. Everything else is new."""
    e = model.enc
    return list((e.backbone if hasattr(e, "backbone") else e).parameters())
