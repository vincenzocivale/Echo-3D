"""Does 480x640 -> 120x160 destroy the motion information in the speckle? Model-free test on val scans:
phase correlation between frames i and i+k at several resolutions vs the tracked in-plane displacement, and NCC vs elevational distance.
python speckle_res.py"""
import json, h5py, math, torch
import torch.nn.functional as F
import data
from geom import rel, load_calib

tr, va = data.split(5); S, C = load_calib(data.ROOT + "calib_matrix.csv"); dev = "cuda" if torch.cuda.is_available() else "cpu"
RES = {"480x640": 1, "240x320": 2, "120x160": 4}
PX_MM = (float(S[0, 0]), float(S[1, 1]))


def shift(a, b):  # phase correlation (Hann-windowed) with parabolic sub-pixel peak: returns (dx, dy) in pixels, b ~ a shifted by (dx, dy)
    h, w = a.shape[-2:]
    win = torch.hann_window(h, device=a.device)[:, None] * torch.hann_window(w, device=a.device)[None]
    A, B = torch.fft.rfft2((a - a.mean((-2, -1), keepdim=True)) * win), torch.fft.rfft2((b - b.mean((-2, -1), keepdim=True)) * win)
    R = A.conj() * B; r = torch.fft.irfft2(R / (R.abs() + 1e-9), s=(h, w))
    n = r.shape[0]; idx = r.flatten(1).argmax(1); py, px = idx // w, idx % w
    out = []
    for i in range(n):
        y, x = int(py[i]), int(px[i]); d = []
        for c, size, ax in ((x, w, 1), (y, h, 0)):
            m, z, p = (r[i, y, (x - 1) % w], r[i, y, x], r[i, y, (x + 1) % w]) if ax == 1 else (r[i, (y - 1) % h, x], r[i, y, x], r[i, (y + 1) % h, x])
            off = 0.5 * (m - p) / (m - 2 * z + p + 1e-12); s = c + float(off); d.append(s - size if s > size / 2 else s)
        out.append(d)
    return torch.tensor(out)


ds = data.Clips(va, full=True, size=(8, 8)); rows = {k: [] for k in RES}
for si in range(0, len(ds), 5):  # 24 scans, all types
    with h5py.File(ds.paths[si]) as f:
        fr, T = f["frames"], torch.from_numpy(f["tforms"][()])
        for k in (1, 4):
            idx = torch.arange(10, len(T) - k - 1, 20)  # ~20 pairs per scan
            Rk = rel(T, C, idx + k, idx)  # image_mm(i+k) -> image_mm(i)
            c = torch.tensor([[320 * PX_MM[0], 240 * PX_MM[1], 0, 1.]]).T
            disp = (Rk @ c)[:, :3, 0] - c[:3, 0]  # displacement of the image centre (mm): in-plane (x,y) and elevational (z)
            A = torch.from_numpy(fr[idx.numpy().tolist()]).float().to(dev) / 255; B = torch.from_numpy(fr[(idx + k).numpy().tolist()]).float().to(dev) / 255
            for name, ds_ in RES.items():
                a, b = (F.avg_pool2d(A[:, None], ds_)[:, 0], F.avg_pool2d(B[:, None], ds_)[:, 0]) if ds_ > 1 else (A, B)
                sh = shift(a, b)
                ncc = ((a - a.mean((1, 2), keepdim=True)) * (b - b.mean((1, 2), keepdim=True))).mean((1, 2)) / (a.std((1, 2)) * b.std((1, 2)) + 1e-9)
                rows[name].append(torch.cat([torch.full((len(idx), 1), k), disp[:, :2] / torch.tensor(PX_MM) / ds_, disp[:, 2:3].abs(), sh.cpu(), ncc.cpu()[:, None]], 1))
out = {}
for name, v in rows.items():
    R = torch.cat(v); out[name] = {}
    for k in (1, 4):
        r = R[R[:, 0] == k]; gt, est = r[:, 1:3], r[:, 4:6]
        sgn = torch.sign(torch.corrcoef(torch.stack([gt[:, 0], est[:, 0]]))[0, 1])
        mae = (est * sgn - gt).abs().mean(0)
        z = r[:, 3]; ncc = r[:, 6]
        out[name][f"k={k}"] = dict(n=len(r), gt_inplane_px_mean=round(gt.abs().mean().item(), 3), phasecorr_corr_x=round(abs(torch.corrcoef(torch.stack([gt[:, 0], est[:, 0]]))[0, 1].item()), 3),
                                   phasecorr_corr_y=round(abs(torch.corrcoef(torch.stack([gt[:, 1], est[:, 1]]))[0, 1].item()), 3), phasecorr_mae_px=[round(m.item(), 3) for m in mae],
                                   ncc_mean=round(ncc.mean().item(), 3), ncc_vs_elev_corr=round(torch.corrcoef(torch.stack([z, ncc]))[0, 1].item(), 3), elev_mm_mean=round(z.mean().item(), 3))
print(json.dumps(out, indent=1)); json.dump(out, open("runs/speckle_res.json", "w"), indent=1)
