"""TUS-REC2024 clips: downsampled frames + metric poses T_{i->0} (mm) relative to the clip's first frame."""
import glob, h5py, random, torch
import torch.nn.functional as F
from geom import load_calib, rel

ROOT = "/raid/DATASETS/TUS-REC2024/"


def split(n_val=5, seed=0, root=ROOT):
    """Subject-disjoint: (train_ids, val_ids) over the subject folders on disk."""
    ids = sorted(p[-3:] for p in glob.glob(root + "[0-4][0-9][0-9]") if int(p[-3:]) < 50)  # official train subjects only
    random.Random(seed).shuffle(ids)
    return sorted(ids[n_val:]), sorted(ids[:n_val])


class Clips(torch.utils.data.Dataset):
    """Random clip of L frames (every `stride`-th) from each scan; full=True -> whole scan from frame 0."""

    def __init__(self, ids, L=64, stride=1, size=(120, 160), full=False, root=ROOT):
        self.paths = [p for i in ids for p in sorted(glob.glob(f"{root}{i}/*.h5"))]
        self.L, self.stride, self.size, self.full = L, stride, size, full
        self.S, self.C = load_calib(root + "calib_matrix.csv")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, k):
        with h5py.File(self.paths[k], "r") as f:
            n, span = f["frames"].shape[0], (self.L - 1) * self.stride + 1
            a = 0 if self.full else random.randint(0, max(0, n - span))
            idx = slice(a, None, self.stride) if self.full else slice(a, a + span, self.stride)
            x, T = f["frames"][idx], torch.from_numpy(f["tforms"][idx])
        x = F.interpolate(torch.from_numpy(x)[:, None].float() / 255, self.size, mode="area")
        return x, rel(T, self.C, torch.arange(len(T)), torch.zeros(len(T), dtype=torch.long))  # [L,1,h,w], [L,4,4]


if __name__ == "__main__":  # self-check: disjoint split, shapes, T_0->0 = I
    tr, va = split()
    assert not set(tr) & set(va) and (tr, va) == split()
    x, T = Clips(va, L=32, stride=2)[0]
    assert x.shape == (32, 1, 120, 160) and T.shape == (32, 4, 4) and 0 <= x.min() and x.max() <= 1
    assert torch.allclose(T[0], torch.eye(4), atol=1e-4)
    xf, Tf = Clips(va, full=True, stride=4)[0]
    assert len(xf) == len(Tf)
    print("data ok", len(tr), len(va), x.shape, "full clip", tuple(xf.shape))
