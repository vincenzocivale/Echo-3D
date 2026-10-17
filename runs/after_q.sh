cd ~/Echo-3D; E=~/miniconda3/envs/echo3d/bin/python
until [ -f runs/pilot_q/code/runs/ALLDONE ]; do sleep 20; done
export CUDA_VISIBLE_DEVICES=4
$E feasibility.py > runs/feasibility.log 2>&1
export CUDA_VISIBLE_DEVICES=5
$E diagnose.py --arm d4rt --ckpt runs/pilot_q/code/runs/d4rt_global.pt --window -1 --name d4rt_global > runs/diag_d4rt_global.log 2>&1
$E diagnose.py --arm d4rt --ckpt runs/pilot_q/code/runs/d4rt_local4.pt --window 4 --name d4rt_local4 > runs/diag_d4rt_local4.log 2>&1
echo DONE > runs/after_q_done
