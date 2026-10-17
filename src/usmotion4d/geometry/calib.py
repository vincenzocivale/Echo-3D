"""Calibration and plane poses. Frames are already scan-converted (curvilinear probe) in TUS-REC: no polar conversion anywhere.
Pixel convention of the official code: x in 1..W, y in 1..H (1-based centres), mm = S @ (x, y, 0, 1).
Here a calibration is a 3x3 affine `mm_from_idx` acting on 0-based pixel indices (i=column, j=row, 1); resize/crop/pad update it exactly."""
import csv, torch
from . import se3

H, W = 480, 640


def load_calib(path):
    """-> S (pixel->mm, 4x4, official 1-based convention), C = T_tool_from_plane_mm (4x4)"""
    r = list(csv.reader(open(path)))
    f = lambda a, b: torch.tensor([[float(x) for x in row] for row in r[a:b]], dtype=torch.float64)
    return f(1, 5), f(6, 10)


class PixelCalib:
    """pixel index -> in-plane mm for an image that went through crop/resize/pad."""

    def __init__(self, mm_from_idx, size):
        self.A, self.size = mm_from_idx.double(), tuple(size)  # size = (h, w)

    @classmethod
    def from_official(cls, S, size=(H, W)):
        A = torch.tensor([[S[0, 0], 0, S[0, 0]], [0, S[1, 1], S[1, 1]], [0, 0, 1]], dtype=torch.float64)  # idx i -> x = i + 1
        return cls(A, size)

    def _chain(self, M, size): return PixelCalib(self.A @ torch.tensor(M, dtype=torch.float64), size)

    def crop(self, top, left, h, w): return self._chain([[1, 0, left], [0, 1, top], [0, 0, 1]], (h, w))

    def pad(self, top, left, bottom, right):
        return self._chain([[1, 0, -left], [0, 1, -top], [0, 0, 1]], (self.size[0] + top + bottom, self.size[1] + left + right))

    def resize(self, h, w):
        """Area/bilinear resize to (h, w): new pixel centre i' sits at ((i'+.5)*f - .5) in old indices."""
        fy, fx = self.size[0] / h, self.size[1] / w
        return self._chain([[fx, 0, .5 * fx - .5], [0, fy, .5 * fy - .5], [0, 0, 1]], (h, w))

    def grid_mm(self, device=None):
        """[h, w, 2] in-plane mm of every pixel centre."""
        j, i = torch.meshgrid(torch.arange(self.size[0], dtype=torch.float64), torch.arange(self.size[1], dtype=torch.float64), indexing="ij")
        p = torch.stack([i, j, torch.ones_like(i)], -1) @ self.A.T
        return p[..., :2].to(device)

    def points_plane(self, device=None):
        """[h*w, 4] homogeneous plane points (x, y, 0, 1) in mm."""
        g = self.grid_mm().reshape(-1, 2)
        return torch.cat([g, torch.zeros_like(g[:, :1]), torch.ones_like(g[:, :1])], -1).to(device)


def plane_poses(tforms, C):
    """T_camera_from_plane = T_camera_from_tool @ T_tool_from_plane, normalised to the first frame:
    T_ref_from_plane_t = inv(T_camera_from_plane_0) @ T_camera_from_plane_t. float64 (tracker translations are ~100s of mm)."""
    Tc = tforms.double() @ C.double()
    return se3.inverse(Tc[0]) @ Tc


def delta(T_ref_from_plane, i, j):
    """Delta_ij = inv(T_ref_from_plane_i) @ T_ref_from_plane_j = T_plane_i_from_plane_j (maps plane-j mm to plane-i mm)."""
    return se3.inverse(T_ref_from_plane[i]) @ T_ref_from_plane[j]


def sequential(deltas_adj):
    """Compose adjacent Delta_{t-1,t} [n-1,4,4] into T_ref_from_plane [n,4,4] (first = identity)."""
    n = len(deltas_adj) + 1
    T = [torch.eye(4, dtype=deltas_adj.dtype, device=deltas_adj.device)]
    for t in range(n - 1): T.append(T[-1] @ deltas_adj[t])
    return torch.stack(T)
