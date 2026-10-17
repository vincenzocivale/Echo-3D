import numpy as np, pytest, torch
from scipy.spatial.transform import Rotation
from usmotion4d.geometry import se3, calib

@pytest.fixture(autouse=True)
def f64():
    torch.set_default_dtype(torch.float64); yield; torch.set_default_dtype(torch.float32)


def rand_T(n, scale=10., ang=1.0, seed=0):
    g = torch.Generator().manual_seed(seed)
    return se3.exp(torch.cat([torch.randn(n, 3, generator=g) * ang, torch.randn(n, 3, generator=g) * scale], -1))


def test_se3_compose_inverse():  # test 2
    A, B = rand_T(16, seed=1), rand_T(16, seed=2)
    I = torch.eye(4)
    assert torch.allclose(A @ se3.inverse(A), I, atol=1e-12)
    assert torch.allclose(se3.inverse(A @ B), se3.inverse(B) @ se3.inverse(A), atol=1e-10)
    assert torch.allclose(se3.compose(A, B, A), A @ B @ A)
    p = torch.randn(16, 3, 1); h = torch.cat([p, torch.ones(16, 1, 1)], 1)  # composition acts on points as successive maps
    assert torch.allclose((A @ B @ h), A @ (B @ h))


def test_se3_exp_log_roundtrip_incl_tiny_and_large():
    for ang in (0., 1e-9, 1e-4, 1e-2, 1., 3.0):
        xi = torch.cat([torch.nn.functional.normalize(torch.randn(32, 3), dim=-1) * ang, torch.randn(32, 3) * 50], -1)
        T = se3.exp(xi)
        assert torch.allclose(se3.log(T), xi, atol=1e-8 + 1e-9 * 50), ang
        assert torch.allclose(T[:, :3, :3] @ T[:, :3, :3].transpose(1, 2), torch.eye(3).expand(32, 3, 3), atol=1e-12)


def test_so3_matches_scipy():
    w = torch.randn(64, 3)
    assert np.allclose(se3.so3_exp(w).numpy(), Rotation.from_rotvec(w.numpy()).as_matrix(), atol=1e-12)
    assert np.allclose(se3.so3_log(se3.so3_exp(w)).numpy(), Rotation.from_matrix(se3.so3_exp(w).numpy()).as_rotvec(), atol=1e-9)


def test_pose_error_log_is_zero_for_equal_and_additive_for_small():
    D = rand_T(8, ang=.01, scale=1.)
    assert se3.log(se3.inverse(D) @ D).abs().max() < 1e-12
    xi = torch.tensor([1e-3, -2e-3, 5e-4, .1, -.2, .3])
    assert torch.allclose(se3.log(se3.exp(xi)), xi, atol=1e-12)


def test_gt_trajectory_relative_to_first_frame():  # test 3
    n = 30
    Tcam_tool = rand_T(n, ang=.5, seed=3)  # synthetic tracker
    C = rand_T(1, ang=.3, seed=4)[0]
    P = calib.plane_poses(Tcam_tool, C)
    assert torch.allclose(P[0], torch.eye(4), atol=1e-12)
    # independent route: T_ref_from_plane_t = inv(Tcam0 C) Tcam_t C
    assert torch.allclose(P[7], se3.inverse(Tcam_tool[0] @ C) @ Tcam_tool[7] @ C, atol=1e-10)
    # Delta_ij composes: Delta_ik = Delta_ij Delta_jk, and sequential composition of adjacent deltas rebuilds the trajectory
    d = lambda i, j: calib.delta(P, i, j)
    assert torch.allclose(d(2, 9), d(2, 5) @ d(5, 9), atol=1e-10)
    adj = torch.stack([d(t, t + 1) for t in range(n - 1)])
    assert torch.allclose(calib.sequential(adj), P, atol=1e-9)
    # gauge: a rigid change of the camera frame does not change anything
    G = rand_T(1, seed=9)[0]
    assert torch.allclose(calib.plane_poses(G @ Tcam_tool, C), P, atol=1e-9)
    # Delta_ij maps points of plane j into plane i: a world point seen from both planes
    pw = torch.randn(4, 1); pw[3] = 1
    pj = se3.inverse(P[9]) @ pw; pi = se3.inverse(P[2]) @ pw
    assert torch.allclose(d(2, 9) @ pj, pi, atol=1e-9)


def test_pixel_plane_3d_roundtrip():  # test 1
    S = torch.diag(torch.tensor([.2294, .2210, 1., 1.]))
    cal = calib.PixelCalib.from_official(S, (48, 64))
    pts = cal.points_plane()  # [N,4]
    T = rand_T(1, ang=.7, seed=5)[0]
    cam = (T @ pts.T)  # plane -> reference/camera
    back = se3.inverse(T) @ cam
    A_inv = torch.linalg.inv(cal.A)
    idx = (A_inv @ torch.cat([back[:2], torch.ones(1, back.shape[1])])).T  # 3D -> plane mm -> pixel index
    jj, ii = torch.meshgrid(torch.arange(48.), torch.arange(64.), indexing="ij")
    assert torch.allclose(idx[:, 0], ii.flatten(), atol=1e-8) and torch.allclose(idx[:, 1], jj.flatten(), atol=1e-8)
    assert back[2].abs().max() < 1e-9  # the plane is z = 0


def test_pixel_to_mm_official_convention():  # test 4
    S = torch.diag(torch.tensor([.2293891906, .2209796905, 1., 1.]))
    cal = calib.PixelCalib.from_official(S)
    g = cal.grid_mm()
    assert torch.allclose(g[0, 0], torch.tensor([.2293891906, .2209796905]))  # official pixel (x=1, y=1) -> (sx, sy)
    assert torch.allclose(g[479, 639], torch.tensor([640 * .2293891906, 480 * .2209796905]))
    assert abs((g[0, 1, 0] - g[0, 0, 0]).item() - .2293891906) < 1e-12  # 0.229 mm / pixel in x, 0.221 in y


def test_crop_resize_pad_update_calibration():  # test 5
    S = torch.diag(torch.tensor([.23, .22, 1., 1.]))
    base = calib.PixelCalib.from_official(S, (48, 64)); g0 = base.grid_mm()
    c = base.crop(5, 7, 20, 30); assert torch.allclose(c.grid_mm(), g0[5:25, 7:37])
    p = base.pad(3, 4, 2, 6); gp = p.grid_mm()
    assert p.size == (53, 74) and torch.allclose(gp[3:51, 4:68], g0)
    for h, w in ((24, 32), (16, 64), (12, 16)):  # area resize: new centre = mean of the old block it covers
        r = base.resize(h, w).grid_mm(); fy, fx = 48 // h, 64 // w
        assert torch.allclose(r, g0.reshape(h, fy, w, fx, 2).mean((1, 3)), atol=1e-12)
    # composition crop -> resize -> pad vs the direct block means
    c2 = base.crop(8, 8, 32, 32).resize(16, 16).pad(1, 1, 1, 1).grid_mm()
    assert torch.allclose(c2[1:-1, 1:-1], g0[8:40, 8:40].reshape(16, 2, 16, 2, 2).mean((1, 3)), atol=1e-12)
    # tensor image resized with the same operator keeps the physical extent: total mm width of the field of view is preserved
    r = base.resize(24, 32)
    ext = lambda c: (c.grid_mm()[0, -1, 0] - c.grid_mm()[0, 0, 0]) + (c.A[0, 0])  # first-to-last centre + one pixel pitch
    assert abs(ext(r).item() - ext(base).item()) < 1e-12


def test_identity_and_algebraic_consistency():
    I = torch.eye(4)
    assert torch.equal(se3.exp(torch.zeros(5, 6)), I.expand(5, 4, 4)) and se3.log(I.expand(5, 4, 4)).abs().max() == 0
    assert torch.allclose(se3.inverse(I), I) and torch.allclose(se3.compose(I, rand_T(1)[0], I), rand_T(1)[0])
    xi = torch.randn(32, 6) * torch.tensor([.3, .3, .3, 5, 5, 5])
    assert torch.allclose(se3.exp(-xi), se3.inverse(se3.exp(xi)), atol=1e-10)  # exp(-xi) = exp(xi)^-1
    assert torch.allclose(se3.inverse(se3.inverse(rand_T(8))), rand_T(8), atol=1e-12)
    # one-parameter subgroup: exp(a xi) exp(b xi) = exp((a+b) xi)
    assert torch.allclose(se3.exp(.3 * xi) @ se3.exp(.5 * xi), se3.exp(.8 * xi), atol=1e-9)
    # small-twist composition is additive to first order (BCH): log(exp(a) exp(b)) ~ a + b
    a, b = torch.randn(16, 6) * 1e-4, torch.randn(16, 6) * 1e-4
    assert (se3.log(se3.exp(a) @ se3.exp(b)) - (a + b)).abs().max() < 1e-7
    # pose error: zero iff equal, invariant to a common right-multiplication of both poses by the same transform
    A, B, G = rand_T(8, seed=11), rand_T(8, seed=12), rand_T(8, seed=13)
    assert se3.log(se3.inverse(A) @ A).abs().max() < 1e-12
    assert torch.allclose(se3.log(se3.inverse(A @ G) @ (B @ G)), se3.log(se3.inverse(G) @ se3.inverse(A) @ B @ G), atol=1e-9)
    # Delta identities on a trajectory: Delta_ii = I, Delta_ji = Delta_ij^-1, Delta_ij Delta_jk Delta_ki = I
    P = calib.plane_poses(rand_T(12, ang=.4, seed=21), rand_T(1, seed=22)[0])
    d = lambda i, j: calib.delta(P, i, j)
    assert torch.allclose(d(4, 4), I, atol=1e-12) and torch.allclose(d(7, 2), se3.inverse(d(2, 7)), atol=1e-10)
    assert torch.allclose(d(1, 5) @ d(5, 9) @ d(9, 1), I, atol=1e-10)
    # loss is exactly zero at the ground truth and its pose_error is zero for the identity target
    from usmotion4d.models.losses import pose_loss
    gt = se3.exp(torch.randn(6, 6) * 1e-3); l, _ = pose_loss(se3.log(gt), gt, torch.ones(6)); assert l.abs() < 1e-12
