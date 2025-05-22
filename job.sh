#!/bin/bash

#SBATCH --job-name=vits2
#SBATCH -o output
#SBATCH -e myJob.%j.err
#SBATCH --partition=gpu
#SBATCH --time=2-00:00
#SBATCH -N 4
#SBATCH --mail-user=alejandro_quintero@sil.org
#SBATCH --mail-type=ALL

module load singularity

if [ -d downloaded_datasets ]; then
    rm -rf downloaded_datasets
fi

mkdir downloaded_datasets

if [ ! -f vits2.sif ]; then
    singularity cache clean -f
    singularity pull vits2.sif docker://alejandroquinterosil/vits2:latest
fi

singularity exec --nv --bind $PWD/downloaded_datasets:/app/downloaded_datasets --bind /usr/lib64/libcuda.so.550.127.05:/usr/local/cuda-12.4/compat/libcuda.so.550.54.15 vits2.sif bash -c "cd /app && make all"
