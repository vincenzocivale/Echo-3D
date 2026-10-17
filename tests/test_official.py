"""Test 6: numerical comparison with the official TUS-REC2024 baseline code (third_party/, only matmul paths; pytorch3d is stubbed)."""
import os, sys, types, pytest, torch
from usmotion4d.geometry import se3, calib
from usmotion4d.evaluation.metrics import tusrec_metrics
from usmotion4d.data import tusrec

BASE = os.path.join(os.path.dirname(__file__), "..", "third_party", "tus-rec-challenge_baseline")
CALIB = tusrec.ROOT + "calib_matrix.csv"


@pytest.fixture(scope="module")
def off():
    if not (os.path.isdir(BASE) and os.path.exists(CALIB)): pytest.skip("official code / data missing")
    sys.modules.setdefault("pytorch3d", types.ModuleType("pytorch3d")); sys.modules.setdefault("pytorch3d.transforms", types.ModuleType("pytorch3d.transforms"))
    sys.path.insert(0, BASE)
    from utils.transform import LabelTransform
    from utils.plot_functions import read_calib_matrices, reference_image_points, data_pairs_adjacent, data_pairs_local
    sys.path.remove(BASE)
    return LabelTransform, read_calib_matrices, reference_image_points, data_pairs_adjacent, data_pairs_local


def test_calibration_matches_official_reader(off):
    S_o, Cmm_o, _ = off[1](CALIB)
    S, C = calib.load_calib(CALIB)
    assert torch.allclose(S.float(), S_o) and torch.allclose(C.float(), Cmm_o)


def test_delta_and_pixel_ddf_match_official_labeltransform(scan, off):
    LT, rcm, rip, adj, loc = off
    S_o, C_o, T_o = rcm(CALIB)
    pts = rip((480, 640), (480, 640))
    tf = scan.tforms[:60]
    mk = lambda kind, pairs: LT(kind, pairs=pairs, image_points=pts, tform_image_to_tool=T_o, tform_image_mm_to_tool=C_o, tform_image_pixel_to_mm=S_o)
    tfi = torch.linalg.inv(tf)
    # 1) the official label T_image_mm(1)->image_mm(0) for pairs (0,n) equals Delta_{0n} up to float32 rounding of the official code
    pairs = adj(60)
    lab = mk("transform", pairs)(tf[None], tfi[None])[0]
    P = calib.plane_poses(tf, scan.C)
    mine = torch.stack([calib.delta(P, 0, n) for n in range(60)])
    assert (lab.double() - mine).abs().max() < 5e-3  # float32 official pipeline vs float64 here (tracker translations ~100s of mm)
    # 2) official global pixel DDF (GT) vs ours
    glob = mk("point", adj(60)[1:])(tf[None], tfi[None])[0]  # [59,3,N] coordinates in frame-0 mm
    P4 = S_o @ pts
    ours = (mine[1:] @ scan.S.expand(1, 4, 4)[0].double() @ pts.double())[:, :3]
    assert (glob.double() - ours).abs().max() < 5e-3
    # 3) local (adjacent) pairs
    lp = mk("point", loc(59))(tf[None], tfi[None])[0]
    ml = torch.stack([calib.delta(P, t, t + 1) for t in range(59)])
    assert (lp.double() - (ml @ scan.S.double() @ pts.double())[:, :3]).abs().max() < 5e-3


def test_metrics_zero_for_gt_and_scale_with_error(scan):
    P = scan.poses[:80]
    lm = scan.landmarks; lm = lm[lm[:, 0] < 80]
    m0 = tusrec_metrics(P, P, scan.S, lm)
    assert max(m0.values()) < 1e-9
    # constant extra translation of 1 mm in-plane per frame: LPE is exactly 1 mm (displacement of every pixel differs by 1 mm)
    step = torch.eye(4, dtype=torch.float64); step[0, 3] = 1.
    adj = torch.stack([calib.delta(P, t, t + 1) for t in range(79)]) @ step
    m = tusrec_metrics(calib.sequential(adj), P, scan.S, lm)
    assert abs(m["LPE"] - 1.0) < 0.05 and m["GPE"] > m["LPE"], m
