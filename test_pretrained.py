"""Equivalence of our USFM / VideoMAE re-implementations with the reference code (USFM repo, HF transformers) at the native input size,
plus the weight audits. Needs the scratch pylib: PYTHONPATH=/raid/DATASETS/Echo3D_weights/pylib python test_pretrained.py"""
import importlib.util, json, sys, types, torch
import pretrained as pt

torch.manual_seed(0)
REPO = pt.CK + "repos/USFM/usdsgen/modules/backbone/vision_transformer.py"
# ---- USFM reference (stub the repo-internal import that only loads checkpoints)
for n in ("usdsgen", "usdsgen.utils", "usdsgen.utils.modelutils"): sys.modules[n] = types.ModuleType(n)
sys.modules["usdsgen.utils.modelutils"].load_pretrained = None
spec = importlib.util.spec_from_file_location("ref_vit", REPO); ref = importlib.util.module_from_spec(spec); spec.loader.exec_module(ref)
from functools import partial
R = ref.VisionTransformer(norm_layer=partial(torch.nn.LayerNorm, eps=1e-6), img_size=224, patch_size=16, in_chans=3, num_classes=0, embed_dim=768, depth=12, num_heads=12,
                          mlp_ratio=4, qkv_bias=True, init_values=.1, use_abs_pos_emb=False, use_rel_pos_bias=False, use_shared_rel_pos_bias=True, use_mean_pooling=False)
ck = torch.load(pt.CK + "usfm/USFM_latest.pth", map_location="cpu")
msg = R.load_state_dict(ck, strict=False); print("USFM reference load: missing", msg.missing_keys, "unexpected", msg.unexpected_keys)
assert not msg.missing_keys, "reference model did not receive every weight"
M = pt.FrameViT(grid=(14, 14)).eval(); a_native = pt.load_usfm(M); R.eval()
g = torch.rand(2, 1, 224, 224)
with torch.no_grad():
    x = R.patch_embed(((g.expand(-1, 3, -1, -1) - M.mean) / M.std))
    x = torch.cat([R.cls_token.expand(2, -1, -1), x], 1); bias = R.rel_pos_bias()
    for b in R.blocks: x = b(x, rel_pos_bias=bias)
    ref_tok = R.norm(x)[:, 1:]; mine = M(g)
d = (ref_tok - mine).abs().max().item() / ref_tok.abs().max().item(); print("USFM native 14x14: max rel diff vs reference", d); assert d < 1e-4
# ---- VideoMAE reference (HF)
from transformers import VideoMAEModel
H = VideoMAEModel.from_pretrained(pt.CK + "videomae-base").eval()
V = pt.VideoMAEEnc(grid=(14, 14), st=2).eval(); a_vm = pt.load_videomae(V)
gv = torch.rand(1, 16, 1, 224, 224)
with torch.no_grad():
    pix = ((gv.expand(-1, -1, 3, -1, -1) - V.mean) / torch.tensor(pt.STD).view(1, 1, 3, 1, 1))
    ref_tok = H(pixel_values=pix).last_hidden_state; mine = V.encode(gv).flatten(1, 2)
d = (ref_tok - mine).abs().max().item() / ref_tok.abs().max().item(); print("VideoMAE native 8x14x14: max rel diff vs HF", d, tuple(ref_tok.shape)); assert d < 1e-4
# ---- audits for the configurations we would actually use
out = {"usfm_native_14x14": a_native, "videomae_native": a_vm,
       "usfm_7x10": pt.load_usfm(pt.FrameViT(grid=(7, 10))), "videomae_7x10_stride2": pt.load_videomae(pt.VideoMAEEnc(grid=(7, 10), st=2)),
       "videomae_7x10_stride1": pt.load_videomae(pt.VideoMAEEnc(grid=(7, 10), st=1))}
n2 = pt.Net2D(usfm=True); out["Net2D_usfm_total"] = dict(audit=n2.audit, total_params=sum(p.numel() for p in n2.parameters()),
                                                       by_module={k: sum(p.numel() for p in m.parameters()) for k, m in n2.named_children()})
json.dump(out, open("runs/pretrained_audit.json", "w"), indent=1, default=str)
for k, v in out.items():
    v = v.get("audit", v); print(f"{k:24s} loaded {v['loaded_params']/1e6:6.2f}M ({v['loaded_tensors']} tensors) | ckpt not used {v['ckpt_not_used_params']/1e6:5.2f}M {list(v['ckpt_not_used'])[:3]}... | random {v['random_init_params']/1e6:5.2f}M {v['random_init_modules']} | adapted {list(v['adapted'])}")
print("test_pretrained ok")
