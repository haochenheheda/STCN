#static images
#CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=4 python -m torch.distributed.launch --master_port 9843 --nproc_per_node=2 train.py --id retrain_b16_g2_s0 --stage 0
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 python -m torch.distributed.launch --master_port 9842 --nproc_per_node=1 train.py --id retrain_b16_g1_s0 --stage 0 --batch_size 16 --num_workers 16
#CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 python -m torch.distributed.launch --master_port 9844 --nproc_per_node=1 train.py --id retrain_b32_g1_s0 --stage 0 --batch_size 32 --num_workers 16 --lr 2e-5 --iterations 200000 --steps 100000


#main train on 150K
#CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=4 python -m torch.distributed.launch --master_port 9842 --nproc_per_node=2 train.py --id retrain_s03 --load_network [path_to_trained_s0.pth] --stage 3
