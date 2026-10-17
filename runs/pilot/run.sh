cd /home/vcivale/Echo-3D/runs/pilot/code
run(){ CUDA_VISIBLE_DEVICES=$1 /home/vcivale/miniconda3/envs/echo3d/bin/python train.py --name $2 --window $3 --steps 3000 --eval_every 500 > runs/$2.log 2>&1; }
run 4 global -1 & run 5 local4 4 & wait
run 4 local8 8 & run 5 local1 1 & wait
echo ALLDONE > runs/ALLDONE
