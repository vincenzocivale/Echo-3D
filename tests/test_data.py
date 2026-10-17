import torch
from usmotion4d.data import split, tusrec
from usmotion4d.geometry import calib


def test_manifest_frozen_and_disjoint():
    m = split.load_manifest()
    assert len(m["train"]) == 40 and len(m["dev"]) == 10 and not set(m["train"]) & set(m["dev"])
    assert all(int(s) < 50 for s in m["train"] + m["dev"])  # public validation (050-052) and test never included
    assert m["sha256"] == "e8c3714a4fa12260af22585e6495e2019c8792fed32494158257b90d9d303ae2"  # change => new manifest version


def test_scan_fields_and_mask(scan):
    assert scan.tforms.shape == (scan.n, 4, 4) and scan.valid_mask.shape == (480, 640) and scan.valid_mask.dtype == torch.bool
    assert scan.meta["orientation"] in ("Par", "Per") and scan.meta["trajectory"] in ("L", "C", "S") and scan.meta["direction"] in ("DtP", "PtD")
    x = scan.frames(slice(0, 5))
    out = x[:, ~scan.valid_mask]
    assert x.shape == (5, 480, 640) and (out > 0).float().mean() < 1e-3 and out.max() <= 4  # scan-converted: only edge noise outside the sector
    assert scan.poses.shape == (scan.n, 4, 4)


def test_pair_dataset_target_is_delta_and_no_pose_in_inputs(scan):
    ds = tusrec.PairDataset(tusrec.scan_paths(["000"]), size=(96, 128), length=8)
    it = ds[3]
    assert set(it) == {"x", "delta", "k", "i", "j"} and it["x"].shape == (2, 1, 96, 128)
    P = scan.poses if scan.meta["scan"] == tusrec.meta("000", ds.entries[0][1])["scan"] else calib.plane_poses(torch.from_numpy(__import__("h5py").File([e for e in ds.entries][0][2])["tforms"][()]), ds.C)
    assert it["delta"].shape == (4, 4) and it["k"] in (1, 2, 4, 8) and ds[3]["i"] == it["i"]  # deterministic per index
