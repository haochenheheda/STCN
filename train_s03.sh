#static images
#CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=4 python -m torch.distributed.launch --master_port 9842 --nproc_per_node=2 train.py --id retrain_b32_s0 --stage 0 --lr 3e-5 --batch_size 32 --iterations 150000 --steps 75000
#CUDA_VISIBLE_DEVICES=2,3 OMP_NUM_THREADS=4 python -m torch.distributed.launch --master_port 9843 --nproc_per_node=2 train.py --id retrain_b8_g2_s0 --stage 0
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=4 python -m torch.distributed.launch --master_port 9842 --nproc_per_node=1 train.py --id retrain_b8_g1_s0 --stage 0 --batch_size 16 --num_workers 16


#main train on 150K
#CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=4 python -m torch.distributed.launch --master_port 9842 --nproc_per_node=2 train.py --id retrain_s03 --load_network [path_to_trained_s0.pth] --stage 3
