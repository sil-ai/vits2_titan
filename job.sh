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
singularity exec --env-file .env --nv --bind $PWD/downloaded_datasets:/app/downloaded_datasets --bind /usr/lib64/libcuda.so.575.57.08:/usr/local/cuda-12.4/compat/libcuda.so.570.86.10 vits2_develop.sif bash -c "cd /app && make all"