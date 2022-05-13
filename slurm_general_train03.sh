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
#CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 python -m torch.distributed.launch --master_port 9848 --nproc_per_node=1 general_train.py --stage 0 --batch_size 16 --num_workers 16 --value_encoder_type resnet18 --key_encoder_type resnet50 --id retrain_b16_g1_s0_k_resnet50
#CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 python -m torch.distributed.launch --master_port 9849 --nproc_per_node=1 general_train.py --stage 0 --batch_size 16 --num_workers 16 --value_encoder_type resnet18 --key_encoder_type resnet50 --id retrain_b16_g1_s0_k_resnet50_aspp --aspp
#CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 python -m torch.distributed.launch --master_port 9850 --nproc_per_node=1 general_train.py --stage 0 --batch_size 16 --num_workers 16 --value_encoder_type resnet18 --key_encoder_type wide_resnet50 --id retrain_b16_g1_s0_k_wide_resnet50
#CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 python -m torch.distributed.launch --master_port 9851 --nproc_per_node=1 general_train.py --stage 0 --batch_size 16 --num_workers 16 --value_encoder_type resnet18 --key_encoder_type wide_resnet50 --id retrain_b16_g1_s0_k_wide_resnet50_aspp --aspp
#CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 python -m torch.distributed.launch --master_port 9952 --nproc_per_node=1 general_train.py --stage 0 --batch_size 16 --num_workers 16 --value_encoder_type resnet18 --key_encoder_type regnet --id retrain_b16_g1_s0_k_regnet --no_amp
#CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 python -m torch.distributed.launch --master_port 9953 --nproc_per_node=1 general_train.py --stage 0 --batch_size 16 --num_workers 16 --value_encoder_type resnet18 --key_encoder_type regnet --id retrain_b16_g1_s0_k_regnet_aspp --aspp --no_amp
#CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 python -m torch.distributed.launch --master_port 9954 --nproc_per_node=1 general_train.py --stage 0 --batch_size 16 --num_workers 16 --value_encoder_type resnet18 --key_encoder_type convext --id retrain_b16_g1_s0_k_convext --no_amp
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 python -m torch.distributed.launch --master_port 9955 --nproc_per_node=1 general_train.py --stage 0 --batch_size 16 --num_workers 16 --value_encoder_type resnet18 --key_encoder_type convext --id retrain_b16_g1_s0_k_convext_aspp --aspp --no_amp

#CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 python -m torch.distributed.launch --master_port 9852 --nproc_per_node=1 general_train.py --stage 0 --batch_size 16 --num_workers 16 --value_encoder_type resnet18 --key_encoder_type resnet200d --id retrain_b16_g1_s0_k_resnet200d
#CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 python -m torch.distributed.launch --master_port 9853 --nproc_per_node=1 general_train.py --stage 0 --batch_size 16 --num_workers 16 --value_encoder_type resnet18 --key_encoder_type resnet200d --id retrain_b16_g1_s0_k_resnet200d_aspp --aspp
#CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 python -m torch.distributed.launch --master_port 9854 --nproc_per_node=1 general_train.py --stage 0 --batch_size 16 --num_workers 16 --value_encoder_type resnet18 --key_encoder_type seresnet152d --id retrain_b16_g1_s0_k_seresnet152d
#CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 python -m torch.distributed.launch --master_port 9855 --nproc_per_node=1 general_train.py --stage 0 --batch_size 16 --num_workers 16 --value_encoder_type resnet18 --key_encoder_type seresnet152d --id retrain_b16_g1_s0_k_seresnet152d_aspp --aspp
#CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 python -m torch.distributed.launch --master_port 9856 --nproc_per_node=1 general_train.py --stage 0 --batch_size 16 --num_workers 16 --value_encoder_type resnet18 --key_encoder_type resnest101 --id retrain_b16_g1_s0_k_resnest101
#CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 python -m torch.distributed.launch --master_port 9857 --nproc_per_node=1 general_train.py --stage 0 --batch_size 16 --num_workers 16 --value_encoder_type resnet18 --key_encoder_type resnest101 --id retrain_b16_g1_s0_k_resnest101_aspp --aspp

#main train on 150K
#CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 python -m torch.distributed.launch --master_port 9848 --nproc_per_node=1 general_train.py --stage 3 --batch_size 16 --num_workers 8 --value_encoder_type resnet18 --key_encoder_type resnet50 --id retrain_b16_g1_s03_k_resnet50_additional --load_network saves/retrain_b16_g1_s0_k_resnet50/retrain_b16_g1_s0_k_resnet50_300000.pth --crop_size 384 --additional_data
#CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 python -m torch.distributed.launch --master_port 9849 --nproc_per_node=1 general_train.py --stage 3 --batch_size 16 --num_workers 8 --value_encoder_type resnet18 --key_encoder_type resnet50 --id retrain_b16_g1_s03_k_resnet50_aspp_additional --aspp --load_network saves/retrain_b16_g1_s0_k_resnet50_aspp/retrain_b16_g1_s0_k_resnet50_aspp_300000.pth --crop_size 384 --additional_data
#CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 python -m torch.distributed.launch --master_port 9850 --nproc_per_node=1 general_train.py --stage 3 --batch_size 16 --num_workers 8 --value_encoder_type resnet18 --key_encoder_type wide_resnet50 --id retrain_b16_g1_s03_k_wide_resnet50_additional --load_network saves/retrain_b16_g1_s0_k_wide_resnet50/retrain_b16_g1_s0_k_wide_resnet50_300000.pth --crop_size 384 --additional_data
#CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 python -m torch.distributed.launch --master_port 9851 --nproc_per_node=1 general_train.py --stage 3 --batch_size 16 --num_workers 8 --value_encoder_type resnet18 --key_encoder_type wide_resnet50 --id retrain_b16_g1_s03_k_wide_resnet50_aspp_additional --aspp --load_network saves/retrain_b16_g1_s0_k_wide_resnet50_aspp/retrain_b16_g1_s0_k_wide_resnet50_aspp_300000.pth --crop_size 384 --additional_data
