#!/bin/sh
#SBATCH -N 1
#SBATCH -p gpu
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=18
#SBATCH --time=80:00:00
##SBATCH --exclusive


module load anaconda3
conda init bash
source activate
conda activate STCN


#static images
#CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 python -m torch.distributed.launch --master_port 9848 --nproc_per_node=1 general_train.py --id retrain_b16_g1_s0_resnet18 --stage 0 --batch_size 16 --num_workers 16 --value_encoder_type resnet18

#main train on 150K
#CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 python -m torch.distributed.launch --master_port 9877 --nproc_per_node=1 general_train.py --id retrain_b16_g1_s03_resnet18 --load_network saves/retrain_b16_g1_s0_resnet18/retrain_b16_g1_s0_resnet18_300000.pth --stage 3 --batch_size 16 --num_workers 16 --value_encoder_type resnet18
#CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 python -m torch.distributed.launch --master_port 9878 --nproc_per_node=1 general_train.py --id retrain_b16_g1_s03_resnet50 --load_network saves/retrain_b16_g1_s0_resnet50/retrain_b16_g1_s0_resnet50_300000.pth --stage 3 --batch_size 16 --num_workers 16 --value_encoder_type resnet50
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 python -m torch.distributed.launch --master_port 9879 --nproc_per_node=1 general_train.py --id retrain_b16_g1_s03_resnest101 --load_network saves/retrain_b16_g1_s0_resnest101/retrain_b16_g1_s0_resnest101_300000.pth --stage 3 --batch_size 16 --num_workers 16 --value_encoder_type resnest101
