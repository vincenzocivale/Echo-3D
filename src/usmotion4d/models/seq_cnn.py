"""SeqPairCNN (option B): per-pair early-fusion CNN features for each adjacent pair (t, t+1) of a window of L frames, then a small temporal
transformer over the L-1 pair features, so every step estimate sees its neighbours. Output: adjacent steps xi_t in se(3) of Delta_{t,t+1}."""
import torch, torch.nn as nn
from ..geometry import se3
from .pair_cnn import PairCNN


class SeqPairCNN(nn.Module):
    def __init__(self, size=(192, 256), L=16, dim=256, depth=2, heads=4, scale_rot=0.002, scale_trans=0.4, temporal=True, **pair_kw):
        super().__init__()
        self.size, self.L, self.temporal = size, L, temporal
        self.pair = PairCNN(size, dim=dim, scale_rot=scale_rot, scale_trans=scale_trans, **pair_kw)
        self.pair.head = self.pair.head[:-1]  # Flatten -> Linear -> ReLU: per-pair feature [dim]; the pose head is below
        self.register_buffer("scale", torch.tensor([scale_rot] * 3 + [scale_trans] * 3))
        self.pos = nn.Parameter(torch.zeros(L - 1, dim)); nn.init.normal_(self.pos, std=.02)
        self.tf = nn.TransformerEncoder(nn.TransformerEncoderLayer(dim, heads, 4 * dim, 0., "gelu", batch_first=True, norm_first=True), depth, enable_nested_tensor=False)
        self.out = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 6)); nn.init.zeros_(self.out[-1].weight); nn.init.zeros_(self.out[-1].bias)  # identity at init

    def trainable_parameters(self): return [p for p in self.parameters() if p.requires_grad]

    def forward(self, x):
        """x [B,L,1,h,w] in [0,1] -> xi [B,L-1,6] (adjacent steps), delta [B,L-1,4,4]."""
        B, L = x.shape[:2]
        p = torch.cat([x[:, :-1], x[:, 1:]], 2).flatten(0, 1)  # [B*(L-1),2,h,w]
        f = self.pair.head(self.pair.net((p - 0.449) / 0.226)).view(B, L - 1, -1)
        if self.temporal: f = self.tf(f + self.pos[:L - 1])
        xi = self.out(f).float() * self.scale
        return dict(xi=xi, delta=se3.exp(xi))
