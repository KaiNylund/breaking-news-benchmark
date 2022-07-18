#!/bin/sh
#SBATCH --job-name=knylund-gpt2-finetuning1
#SBATCH --partition=gpu-titan
#SBATCH --account=ark
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:2
#SBATCH --time=120:00:00
#SBATCH --chdir=/mmfs1/gscratch/scrubbed/knylund/mind-the-gap-replication
#SBATCH --mail-type=ALL
#SBATCH --mail-user=knylund@uw.edu

module load cuda/11.6.0
python gpt2_finetuning.py