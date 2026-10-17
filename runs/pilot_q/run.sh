cd /home/vcivale/Echo-3D/runs/pilot_q/code
CUDA_VISIBLE_DEVICES=4 /home/vcivale/miniconda3/envs/echo3d/bin/python train_q.py --name d4rt_global --window -1 --steps 3000 --eval_every 500 > runs/d4rt_global.log 2>&1 &
CUDA_VISIBLE_DEVICES=5 /home/vcivale/miniconda3/envs/echo3d/bin/python train_q.py --name d4rt_local4 --window 4 --steps 3000 --eval_every 500 > runs/d4rt_local4.log 2>&1 &
wait; echo ALLDONE > runs/ALLDONE
