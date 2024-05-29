#!/bin/bash
#SBATCH -J train
#SBATCH -p gpu
#SBATCH --gpus-per-task=2
#SBATCH --cpus-per-task=16
#SBATCH --constraint=a100
#SBATCH --ntasks=1
#SBATCH --mem=1000G
#SBATCH --time=4-00:00:00
#SBATCH -o %j.out
#SBATCH -e %j.err

export PYTHONPATH=/mnt/home/nkavokine/pydlr/libdlr:$PYTHONPATH

export OMP_DYNAMIC=FALSE
export OMP_PROC_BIND=TRUE
export OMP_NESTED=FALSE
export OMP_WAIT_POLICY=ACTIVE
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export MKL_DYNAMIC=FALSE

python3 2_train_ACANN.py 
