#!/bin/bash
cd /home/vcivale/Echo-3D
PY=~/micromamba/envs/methyl-predictor/bin/python
until grep -q '^done' outputs/cache_feats.log; do sleep 10; done
$PY scripts/train_longctx.py --L 16  --name L16                                  > outputs/longctx_L16.log 2>&1
$PY scripts/train_longctx.py --L 64  --name L64                                  > outputs/longctx_L64.log 2>&1
$PY scripts/train_longctx.py --L 256 --name L256                                 > outputs/longctx_L256.log 2>&1
$PY scripts/train_longctx.py --L 64  --name L64_k32  --offsets 1 2 4 8 16 32     > outputs/longctx_L64_k32.log 2>&1
$PY scripts/train_longctx.py --L 256 --name L256_k64 --offsets 1 2 4 8 16 32 64  > outputs/longctx_L256_k64.log 2>&1
echo ALLDONE > outputs/longctx_done
