#!/bin/bash
cd /home/vcivale/Echo-3D; PY=~/micromamba/envs/methyl-predictor/bin/python
for S in 42 97; do
  $PY scripts/train_longctx.py --L 16 --seed $S --name tf_L16_s$S > outputs/longctx_tf_L16_s$S.log 2>&1
  for A in tf tf_sin gru; do for L in 64 256; do
    $PY scripts/train_longctx.py --L $L --arch $A --seed $S --name ${A}_L${L}_s$S > outputs/longctx_${A}_L${L}_s$S.log 2>&1
  done; done
done
echo SEEDSDONE > outputs/seeds_done
