"""TUS-REC2024 HDF5 loader. Train layout: <root>/<sub>/<scan>.h5 {frames,tforms}, <root>/landmark_<sub>.h5[<scan>];
public validation layout: frames/<sub>/<scan>.h5, transfs/<sub>/<scan>.h5, landmark/landmark_<sub>.h5. Scan name = <side>_<orientation>_<traj>_<direction>."""
import glob, os, random, h5py, numpy as np, torch
import torch.nn.functional as F
from ..geometry import calib, se3

ROOT = "/mnt/hdd/vcivale/DATASETS/TUS-REC2024/"
VAL_ROOT = "/mnt/hdd/vcivale/DATASETS/TUS-REC2024_val/"
_MASK = os.path.join(os.path.dirname(__file__), "..", "..", "..", "assets", "valid_mask_480x640.npy")


def scan_paths(subjects, root=ROOT):
    out = []
    for s in subjects:
        if os.path.isdir(f"{root}frames/{s}"):  # public validation layout
            out += [(s, p, p.replace("/frames/", "/transfs/"), f"{root}landmark/landmark_{s}.h5") for p in sorted(glob.glob(f"{root}frames/{s}/*.h5"))]
        else:
            out += [(s, p, p, f"{root}landmark_{s}.h5") for p in sorted(glob.glob(f"{root}{s}/*.h5"))]
    return out


def meta(sub, path):
    n = os.path.basename(path)[:-3]; side, ori, traj, d = n.split("_")
    return dict(subject=sub, scan=n, side=side, orientation=ori, trajectory=traj, direction=d, protocol="TUS-REC2024")


class Scan:
    """One sequence. Keys: frames [n,H,W] uint8, tforms [n,4,4] (tool->camera), pixel_to_plane_mm (S 4x4), plane_to_tool (C 4x4),
    valid_mask [H,W] bool, landmarks [K,3], meta. `poses` = T_ref_from_plane (first frame = identity) is derived, never given to a model."""

    def __init__(self, entry, root=ROOT):
        sub, fpath, tpath, lpath = entry
        self.meta, self.fpath, self.tpath, self.lpath = meta(sub, fpath), fpath, tpath, lpath
        self.S, self.C = calib.load_calib(root + "calib_matrix.csv")
        self.valid_mask = torch.from_numpy(np.load(_MASK))
        with h5py.File(tpath) as f: self.tforms = torch.from_numpy(f["tforms"][()])
        with h5py.File(fpath) as f: self.n = f["frames"].shape[0]
        assert len(self.tforms) == self.n

    @property
    def pixel_to_plane_mm(self): return self.S

    @property
    def plane_to_tool(self): return self.C

    @property
    def poses(self): return calib.plane_poses(self.tforms, self.C)

    @property
    def landmarks(self):
        with h5py.File(self.lpath) as f: return torch.from_numpy(f[self.meta["scan"]][()])

    def frames(self, idx=None):
        with h5py.File(self.fpath) as f: return torch.from_numpy(f["frames"][idx if idx is not None else slice(None)])


def preprocess(x, size):
    """uint8 [n,H,W] -> float [n,1,h,w] in [0,1], area resize; the matching calibration is PixelCalib.from_official(S).resize(*size)."""
    return F.interpolate(x[:, None].float() / 255, size, mode="area")


class PairDataset(torch.utils.data.Dataset):
    """Random pairs (i, j=i+k) from random scans, k in `offsets`, random direction sign. Item = frames a,b and Delta_ab = T_plane_a_from_plane_b
    (the target; pose of frame b expressed in frame a). Scans are cached per worker only by path (frames read lazily from HDF5)."""

    def __init__(self, entries, offsets=(1, 2, 4, 8), size=(192, 256), root=ROOT, length=100000, seed=0, n_unique=None, reverse_prob=0.0, frac=(0.0, 1.0)):
        self.entries, self.offsets, self.size, self.root, self.length, self.seed, self.n_unique, self.reverse_prob, self.frac = entries, tuple(offsets), size, root, length, seed, n_unique, reverse_prob, frac
        self.S, self.C = calib.load_calib(root + "calib_matrix.csv")
        self._T = {}

    def __len__(self): return self.length

    def _poses(self, e):
        if e[1] not in self._T:
            with h5py.File(e[2]) as f: self._T[e[1]] = calib.plane_poses(torch.from_numpy(f["tforms"][()]), self.C)
        return self._T[e[1]]

    def __getitem__(self, k):
        r = random.Random(self.seed * 1_000_003 + (k % self.n_unique if self.n_unique else k))
        e = self.entries[r.randrange(len(self.entries))]; T = self._poses(e); n = len(T)
        off = r.choice(self.offsets); lo, hi = int(self.frac[0] * n), int(self.frac[1] * n)  # frame range usable (temporal split for the audit)
        i = r.randrange(lo, hi - off); j = i + off
        if r.random() < self.reverse_prob: i, j = j, i  # default forward only: inference always asks (t, t+k); the sign of the elevational motion is not observable from two frames alone
        with h5py.File(e[1]) as f: x = torch.from_numpy(np.stack([f["frames"][i], f["frames"][j]]))
        return dict(x=preprocess(x, self.size), delta=calib.delta(T, i, j).float(), k=abs(j - i), i=i, j=j)


class SeqDataset(PairDataset):
    """Random windows of L consecutive frames. Item: x [L,1,h,w], T [L,4,4] window poses relative to its first frame (T_win0_from_plane_t)."""

    def __init__(self, entries, L=16, **kw):
        super().__init__(entries, **kw); self.L = L

    def __getitem__(self, k):
        r = random.Random(self.seed * 1_000_003 + (k % self.n_unique if self.n_unique else k))
        e = self.entries[r.randrange(len(self.entries))]; T = self._poses(e)
        s = r.randrange(0, len(T) - self.L + 1)
        with h5py.File(e[1]) as f: x = torch.from_numpy(f["frames"][s:s + self.L])
        return dict(x=preprocess(x, self.size), T=(se3.inverse(T[s]) @ T[s:s + self.L]).float())
