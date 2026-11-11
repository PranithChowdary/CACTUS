#!/bin/bash
#SBATCH --job-name=cactus_train
#SBATCH --account=cminds_anandi
#SBATCH --partition=cn3_anandi
#SBATCH --qos=anandi
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8   ## max allowd is 8 for cn3_anandi partition
#SBATCH --mem=62G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Load shell configurations and activate conda
source ~/miniconda3/bin/activate

# Navigate to your working directory
cd /users/staff/pranith/CACTUS/

# Ensure the required directories exist before execution
mkdir -p logs results checkpoints

# Setup local conda environment inside CACTUS if it doesn't exist
if [ ! -d "cactus_env" ]; then
    echo "Creating new local conda environment 'cactus_env'..."
    conda create --prefix ./cactus_env python=3.12 -y
    conda activate ./cactus_env
    echo "Installing requirements..."
    pip install -r requirements.txt
else
    echo "Activating existing local conda environment 'cactus_env'..."
    conda activate ./cactus_env
fi

srun python experiments/train_ppo.py
srun python experiments/evaluate_all.py
