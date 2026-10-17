"""PairCNN: lightweight CNN trained from scratch for this task (control for the frozen-USFM arms). Early fusion: frames i and j are stacked as
2 input channels, so the network compares pixels/speckle directly. The final map is flattened (not averaged) to keep where motion happens,
which rotations need. Same output contract as PoseEstimator: xi in se(3) of Delta_ij, scaled by the per-step magnitudes times k."""
import torch, torch.nn as nn
from ..geometry import se3


class Block(nn.Module):
    def __init__(self, cin, cout, stride):
        super().__init__()
        self.f = nn.Sequential(nn.Conv2d(cin, cout, 3, stride, 1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                               nn.Conv2d(cout, cout, 3, 1, 1, bias=False), nn.BatchNorm2d(cout))
        self.skip = nn.Identity() if stride == 1 and cin == cout else nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout))

    def forward(self, x): return (self.f(x) + self.skip(x)).relu()


class PairCNN(nn.Module):
    def __init__(self, size=(192, 256), widths=(32, 64, 128, 256), dim=256, scale_rot=0.002, scale_trans=0.4, backbone="small", imagenet=False):
        super().__init__()
        self.size = size
        self.register_buffer("scale", torch.tensor([scale_rot] * 3 + [scale_trans] * 3))
        if backbone == "effb1":  # torchvision EfficientNet-B1 features (stride 32), optionally ImageNet; first conv RGB -> 2 frames: sum_c W_c / 2 per frame
            from torchvision.models import efficientnet_b1
            f = efficientnet_b1(weights="IMAGENET1K_V2" if imagenet else None).features
            w = f[0][0].weight.data.sum(1, keepdim=True) / 2
            f[0][0] = nn.Conv2d(2, w.shape[0], 3, 2, 1, bias=False); f[0][0].weight.data.copy_(w.expand(-1, 2, -1, -1))
            layers, c, s = [f], 1280, 32
        else:
            assert not imagenet, "no pretrained weights for the small backbone"
            layers, c = [nn.Conv2d(2, widths[0], 5, 2, 2, bias=False), nn.BatchNorm2d(widths[0]), nn.ReLU(inplace=True)], widths[0]  # stride 2
            for w in widths: layers += [Block(c, w, 2), Block(w, w, 1)]; c = w  # stride 2 per stage -> total 2 * 2^len(widths)
            s = 2 ** (len(widths) + 1)
        self.net = nn.Sequential(*layers, nn.Conv2d(c, 32, 1))
        n = 32 * (size[0] // s) * (size[1] // s)
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(n, dim), nn.ReLU(inplace=True), nn.Linear(dim, 8))
        nn.init.zeros_(self.head[-1].weight); nn.init.zeros_(self.head[-1].bias)  # identity at init

    def trainable_parameters(self): return [p for p in self.parameters() if p.requires_grad]

    def encode(self, x):
        """No per-frame encoder: 'features' are the frames, scaled to integers 0..255 (exact in bf16, used by pose_eval's cache)."""
        return (x * 255).round()

    def forward(self, fa, fb, k=None):
        x = torch.cat([fa, fb], 1).float() / 255
        o = self.head(self.net((x - 0.449) / 0.226)).float()  # gray ImageNet normalisation (input scale mattered in docs/DIAGNOSIS_AND_FEASIBILITY.md)
        k = torch.ones(len(x), device=x.device) if k is None else k.float()
        xi = o[:, :6] * self.scale * k[:, None]
        return dict(xi=xi, delta=se3.exp(xi), logvar=o[:, 6:8])
