cd ~/Echo-3D; E=~/miniconda3/envs/echo3d/bin/python; W=/raid/DATASETS/Echo3D_weights/matrix; M=runs/matrix/code/runs
one(){ # gpu enc head
  until [ -f $M/$2_$3.json ]; do sleep 30; done
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=$1 $E diagnose.py --arm $3 --enc $2 --ckpt $W/$2_$3.pt --name $2_$3 > runs/diag_$2_$3.log 2>&1
}
one 4 videomae d4rt & one 5 videomae pose & one 1 usfm d4rt & one 2 usfm pose & wait; echo DONE > runs/diag_matrix_done
