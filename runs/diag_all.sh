cd ~/Echo-3D; E=~/miniconda3/envs/echo3d/bin/python; export CUDA_VISIBLE_DEVICES=2
$E diagnose.py --arm pair --ckpt runs/pairwise.pt --name pairwise_random > runs/diag_pairwise_random.log 2>&1
for w in -1 4 8 1; do n=$([ $w = -1 ] && echo global || echo local$w); $E diagnose.py --arm pose --ckpt runs/pilot/code/runs/$n.pt --window $w --name pose_$n > runs/diag_pose_$n.log 2>&1; done
echo DONE > runs/diag_A_done
