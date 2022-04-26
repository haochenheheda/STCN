#!/bin/sh
#SBATCH -N 1
#SBATCH -p gpu
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=18
#SBATCH --time=60:00:00
##SBATCH --exclusive


module load anaconda3
conda init bash
source activate
conda activate STCN

python eval_youtube_ms.py --output output_ms_top_kmn

