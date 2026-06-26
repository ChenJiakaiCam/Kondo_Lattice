#!/bin/bash

# 1. Load the cluster's foundational networking and hardware drivers
. /etc/profile.d/modules.sh
module purge
module load rhel8/default-amp
module load gcc/11

# 2. Activate your Python environment (Make sure Conda is NOT active first)
source /rds/user/jc2405/hpc-work/JaQMC/jaqmc/.venv/bin/activate

# 3. Safely link ONLY the specific Nvidia math and NCCL libraries
VENV_LIB="/rds/user/jc2405/hpc-work/JaQMC/jaqmc/.venv/lib/python3.12/site-packages"
export LD_LIBRARY_PATH="$VENV_LIB/nvidia/nccl/lib:$VENV_LIB/nvidia/cudnn/lib:$VENV_LIB/nvidia/cublas/lib:$VENV_LIB/nvidia/cusparse/lib:$VENV_LIB/nvidia/cusolver/lib:$VENV_LIB/nvidia/cufft/lib:$LD_LIBRARY_PATH"

# 4. Force JAX to use the GPU
export JAX_PLATFORMS=cuda

echo "JaQMC Environment Activated, Hardware Drivers Loaded, and GPU Paths Linked Safely!"
