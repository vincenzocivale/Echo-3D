"""USFM ViT-B/16 (BEiT-style) as prepared in this repo (pretrained.py): light re-implementation + loader with weight audit. Checkpoint:
/mnt/hdd/vcivale/DATASETS/Echo3D_weights/usfm/USFM_latest.pth. Verified against the reference implementation (4e-7 relative diff)."""
import math, torch, torch.nn as nn, torch.nn.functional as F

CK = "/mnt/hdd/vcivale/DATASETS/Echo3D_weights/"
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
    ck = {k: v for k, v in torch.load(path, map_location="cpu", weights_only=True).items() if torch.is_tensor(v)}
    mapped, adapted = {}, {"_src": set()}
    for k, v in ck.items():
        if k in ("mask_token", "rel_pos_bias.relative_position_index"): continue  # MIM-only token / derived buffer
        if k.endswith("relative_position_bias_table"):
            v = interp_table(v, *model.grid)
            if tuple(model.grid) != (14, 14): adapted[k] = f"bicubic resize of the 27x27 offset table (+3 cls entries) for a {model.grid[0]}x{model.grid[1]} token grid"
        mapped[k] = v
    return load_with_audit(model, mapped, adapted, ck)


