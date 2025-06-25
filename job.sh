#!/bin/bash
#SBATCH --job-name=vits2
#SBATCH -o output.%j.out
#SBATCH -e myJob.%j.err
#SBATCH --partition=gpu
#SBATCH --time=2-00:00
#SBATCH -N 4
#SBATCH --mail-user=alejandro_quintero@sil.org
#SBATCH --mail-type=ALL
module load singularity
singularity exec --env-file .env --nv --bind $PWD/downloaded_datasets:/app/downloaded_datasets --bind /opt/software/NVHPC/25.3-CUDA-12.8.0/Linux_x86_64/25.3/cuda/12.8/compat/libcuda.so.570.124.06:/usr/local/cuda-12.8/targets/x86_64-linux/lib/stubs/libcuda.so vits2_develop.sif bash -c "cd /app && make all"