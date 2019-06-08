#!/bin/bash
#SBATCH --partition=deep
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=115G

# only use the following on partition with GPUs
#SBATCH --gres=gpu:1

#SBATCH --job-name=train_cartpole_3
#SBATCH --output=./sbatch_logs/train_cartpole_3-%j.out

# only use the following if you want email notification

# list out some useful information
echo "SLURM_JOBID="$SLURM_JOBID
echo "SLURM_JOB_NODELIST"=$SLURM_JOB_NODELIST
echo "SLURM_NNODES"=$SLURM_NNODES
echo "SLURMTMPDIR="$SLURMTMPDIR
echo "working directory = "$SLURM_SUBMIT_DIR

# sample job
cd /deep/u/maximev/RL_project
source venv/bin/activate
cd ./baselines
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/deep/u/maximev/cuda/lib64
python -m baselines.deepq.experiments.train_cartpole_3

# done
echo "Done"
