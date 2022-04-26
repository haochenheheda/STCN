#!/bin/sh
#SBATCH -N 1
#SBATCH -p gpu
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=18
#SBATCH --time=2:00:00
##SBATCH --exclusive


module load anaconda3
conda init bash
source activate
conda activate STCN

python eval_youtube.py --output output_bs32_g1 --model saves/retrain_b32_g1_s03/retrain_b32_g1_s03_100000.pth

