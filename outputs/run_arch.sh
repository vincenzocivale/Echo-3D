#!/bin/bash
cd /home/vcivale/Echo-3D; PY=~/micromamba/envs/methyl-predictor/bin/python
for L in 16 64 256; do for A in tf_sin gru; do
$PY scripts/train_longctx.py --L $L --arch $A --name ${A}_L$L > outputs/longctx_${A}_L$L.log 2>&1; done; done
echo ARCHDONE > outputs/arch_done
