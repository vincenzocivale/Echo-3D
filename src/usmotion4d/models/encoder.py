"""UltrasoundEncoder: B-mode frames -> multiscale spatial features. The pretrained encoder is the USFM ViT-B/16 already prepared in this
repo (not USF-MAE); it is frozen (eval, no grad). A plain ViT has a single stride, so 'multiscale' = features from several depths."""
import torch, torch.nn as nn
from .usfm_vit import FrameViT, load_usfm

CKPT = "/mnt/hdd/vcivale/DATASETS/Echo3D_weights/usfm/USFM_latest.pth"


class UltrasoundEncoder(nn.Module):
    """x [N,1,h,w] in [0,1] -> list of feature maps [N,C,gh,gw] (one per entry of `layers`, shallow to deep). `channels` / `stride` are attributes."""
    channels: int
    stride: int

    def forward(self, x): raise NotImplementedError


class USFMEncoder(UltrasoundEncoder):
    def __init__(self, size=(192, 256), layers=(8, 12), pretrained=True, frozen=True):
        super().__init__()
        assert size[0] % 16 == 0 and size[1] % 16 == 0
        self.size, self.layers, self.stride, self.channels, self.frozen = size, tuple(layers), 16, 768, frozen
        self.grid = (size[0] // 16, size[1] // 16)
        self.vit = FrameViT(self.grid)
        self.audit = load_usfm(self.vit, CKPT) if pretrained else None
        if frozen:
            self.vit.requires_grad_(False)
            self.eval()

    def train(self, mode=True):  # a frozen encoder stays in eval mode
        return super().train(mode and not self.frozen)

    def forward(self, x):
        v = self.vit
        ctx = torch.no_grad() if self.frozen else torch.enable_grad()
        with ctx:
            t = v.patch_embed.proj((x.expand(-1, 3, -1, -1) - v.mean) / v.std).flatten(2).transpose(1, 2)
            t = torch.cat([v.cls_token.expand(len(t), -1, -1).to(t.dtype), t], 1)
            bias, out = v.rel_pos_bias().to(t.dtype), ([t[:, 1:]] if 0 in self.layers else [])  # layer 0 = patch embedding (keeps speckle detail)
            for k, b in enumerate(v.blocks, 1):
                t = b(t, bias)
                if k in self.layers: out.append((v.norm(t) if k == len(v.blocks) else t)[:, 1:])
        return [o.transpose(1, 2).reshape(len(o), -1, *self.grid) for o in out]
