"""Cost of candidate encoders inside the full D4RT step (encoder + decoder, Q=512 queries, L1 loss, bf16 autocast, fwd+bwd+AdamW).
Batch size is halved on OOM. Run on an otherwise idle GPU: python feasibility.py [name_filter]"""
import json, sys, time, torch
from model import Net, D4RT
import pretrained as pt
from geom import load_calib

S, _ = load_calib("/raid/DATASETS/TUS-REC2024/calib_matrix.csv"); dev = "cuda"
Q = 512


def measure(name, make, size, L, bs):
    while bs >= 1:
        try:
            torch.manual_seed(0); enc = make(); m = D4RT(enc, S, dim=256, heads=8).to(dev); opt = torch.optim.AdamW(m.parameters(), 1e-4)
            x = torch.rand(bs, L, 1, *size, device=dev); uv = torch.rand(bs, Q, 2, device=dev)
            src, ref = torch.randint(0, L, (bs, Q), device=dev), torch.randint(0, L, (bs, Q), device=dev); gt = torch.randn(bs, Q, 3, device=dev)
            torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
            for i in range(10):
                if i == 3: torch.cuda.synchronize(); t0 = time.time()
                with torch.autocast("cuda", torch.bfloat16):
                    loss = (m(x, uv, src, ref) - gt).abs().mean()
                opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            torch.cuda.synchronize(); dt = (time.time() - t0) / 7
            with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
                kv = m.encode(x[:1]).shape[1]
            n_enc, n_dec = sum(p.numel() for p in m.enc.parameters()), sum(p.numel() for p in m.dec.parameters())
            return dict(name=name, size=size, L=L, kv_tokens_per_clip=kv, batch=bs, params_enc_m=round(n_enc / 1e6, 1), params_dec_m=round(n_dec / 1e6, 1),
                        vram_gb=round(torch.cuda.max_memory_allocated() / 1e9, 1), sec_per_step=round(dt, 3), clips_per_s=round(bs / dt, 1),
                        hours_per_3000_steps_at_bs8=round(3000 * dt * 8 / bs / 3600, 2))
        except torch.cuda.OutOfMemoryError:
            del enc; torch.cuda.empty_cache(); bs //= 2
    return dict(name=name, size=size, L=L, note="OOM even at bs=1")


C = [("ours Net (scratch) 120x160 L64", lambda: Net(window=None), (120, 160), 64, 8),
     ("USFM-MV (Net2D) 120x160 L64 [7x10 grid]", lambda: pt.Net2D(grid=(7, 10)), (120, 160), 64, 8),
     ("USFM-MV (Net2D) 176x240 L64 [11x15]", lambda: pt.Net2D(grid=(11, 15)), (176, 240), 64, 8),
     ("USFM-MV (Net2D) 240x320 L64 [15x20]", lambda: pt.Net2D(grid=(15, 20)), (240, 320), 64, 8),
     ("VideoMAE-B native 224x224 L16 stride2 [8x14x14]", lambda: pt.VideoMAEEnc(grid=(14, 14), st=2), (224, 224), 16, 8),
     ("VideoMAE-B 120x160 L64 stride2 [32x7x10]", lambda: pt.VideoMAEEnc(grid=(7, 10), st=2), (120, 160), 64, 8),
     ("VideoMAE-B 120x160 L64 stride1 [63x7x10]", lambda: pt.VideoMAEEnc(grid=(7, 10), st=1), (120, 160), 64, 8),
     ("VideoMAE-B 176x240 L64 stride2 [32x11x15]", lambda: pt.VideoMAEEnc(grid=(11, 15), st=2), (176, 240), 64, 8)]
flt = sys.argv[1] if len(sys.argv) > 1 else ""
res = []
for c in C:
    if flt in c[0]:
        r = measure(*c); res.append(r); print(json.dumps(r), flush=True)
json.dump(res, open("runs/feasibility.json", "w"), indent=1)
