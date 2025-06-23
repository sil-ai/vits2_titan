# VITS2 Titan - High Performance Computing Implementation

This project is an HPC-optimized implementation of VITS2 (Variational Inference with adversarial learning for end-to-end Text-to-Speech), based on the work from [daniilrobnikov/vits2](https://github.com/daniilrobnikov/vits2).

## About VITS2

VITS2 is an advanced single-stage text-to-speech model that efficiently synthesizes high-quality, natural speech through improved adversarial learning and architecture design. It represents a significant improvement over the original VITS model, offering:

- **Enhanced Naturalness**: Improved speech quality with reduced artifacts
- **Computational Efficiency**: Optimized training and inference processes
- **Reduced Phoneme Dependency**: Supports fully end-to-end single-stage approach
- **Multi-speaker Capabilities**: Support for multiple speakers (coming soon)

The original research paper: [VITS2: Improving Quality and Efficiency of Single-Stage Text-to-Speech with Adversarial Learning and Architecture Design](https://arxiv.org/abs/2307.16430)

## Current Status

**✅ Available**: Single-speaker model (`ljs_base`) - Fully functional for LJSpeech dataset training and inference

**🚧 Coming Soon**: Multi-speaker model support with VCTK dataset compatibility

## HPC Execution

### Prerequisites

- Access to the Titan HPC cluster via [ORCA OnDemand](https://ood.orca.offn.onenet.net/)
- Valid cluster credentials
- AWS S3 credentials for model checkpoint storage

### Execution Steps

Follow these steps to execute the VITS2 training on the Titan HPC cluster:

#### 1. Access the HPC Cluster.

Using UI

1. Navigate to [https://ood.orca.offn.onenet.net/](https://ood.orca.offn.onenet.net/)
2. Log in with your provided credentials
3. In the cluster section, click on **titan_shell_access** to access the terminal

for example to aquintero@titan.orca.oru.edu the **username** is: _aquintero_

or

Using SSH conecction with the command and then added your password.

```bash
ssh aquintero@titan.orca.oru.edu
```

#### 2. Setup Working Directory and Download Docker Image

Create the required dataset directory:
```bash
mkdir downloaded_datasets
```

Next, load the Singularity module and pull the desired Docker image from Docker Hub. You can choose between the `main` (stable) and `develop` (latest features) branches.

**Option A: Pull the `main` image (recommended):**
```bash
module load singularity
singularity cache clean -f
singularity pull vits2_main.sif docker://alejandroquinterosil/vits2:main
```

**Option B: Pull the `develop` image (for the latest changes):**
```bash
module load singularity
singularity cache clean -f
singularity pull vits2_develop.sif docker://alejandroquinterosil/vits2:develop
```

#### 3. Create Job Script

Create a `job.sh` file with the following content. This script configures SLURM job parameters including:
- Job name and resource allocation (4 nodes, GPU partition)
- Output and error file redirection (`output.%j.out` and `myJob.%j.err`)
- 2-day time limit for training completion
- Email notifications for job status updates
- Singularity container execution with GPU support and volume bindings

```bash
#!/bin/bash

#SBATCH --job-name=vits2
#SBATCH -o output.%j.out
#SBATCH -e myJob.%j.err
#SBATCH --partition=gpu
#SBATCH --time=2-00:00
#SBATCH -N 4
#SBATCH --mail-user=your_email@domain.com
#SBATCH --mail-type=ALL

module load singularity

# IMPORTANT: Make sure to use the correct .sif file name from the pull step (e.g., vits2_main.sif)
singularity exec --env-file .env --nv --bind $PWD/downloaded_datasets:/app/downloaded_datasets --bind /usr/lib64/libcuda.so.550.127.05:/usr/local/cuda-12.4/compat/libcuda.so.550.54.15 vits2_main.sif bash -c "cd /app && make all"
```

#### 4. Configure Credentials

Create a `.env` file in the same directory with your credentials. Use `touch .env` and `nano .env` to create and modify it.

**AWS S3 Credentials (for model checkpoint storage):**
Model checkpoints are automatically uploaded to clearml during training, which requires these credentials.
```bash
AWS_ACCESS_KEY_ID=your_access_key_here
AWS_SECRET_ACCESS_KEY=your_secret_key_here
AWS_STORAGE_BUCKET_NAME=vits2-titan
```
**Important**: Replace the placeholder values with your actual AWS credentials.

**ClearML Credentials (for experiment tracking):**
The project is integrated with ClearML to log metrics, hyperparameters, and model checkpoints.
```bash
CLEARML_API_ACCESS_KEY=your_api_acces_key
CLEARML_API_SECRET_KEY=your_api_secret_key
CLEARML_API_HOST=https://api.sil.hosted.allegro.ai
CLEARML_WEB_HOST=https://app.sil.hosted.allegro.ai
CLEARML_FILES_HOST=https://files.sil.hosted.allegro.ai
```
**Important**: You can find your ClearML credentials in the "Profile" section of the ClearML web UI. The experiment tracking will fail if these are not provided.

#### 5. Submit the Job

Execute the training job using SLURM:
```bash
sbatch job.sh
```

This will return a job ID (e.g., `Submitted batch job 12345`) which you can use to monitor the job status.

#### 6. Monitor Job Execution

The job execution generates two output files in your working directory:
- **`myJob.{job_id}.err`**: Contains error messages and debugging information
- **`output.{job_id}.out`**: Contains standard output and training progress logs

To check if your job is still running, use:
```bash
squeue -u username
```
Replace `username` with your actual cluster username.

### Training Pipeline

The container automatically executes the following workflow:

1. **Dataset Download**: Downloads LJSpeech-1.1 dataset
2. **Preprocessing**: Generates mel-spectrograms from audio files
3. **File Lists Generation**: Creates training/validation splits and vocabulary
4. **Training**: Initiates model training with distributed GPU support

### Configuration

The training uses the default LJSpeech configuration located at `downloaded_datasets/config.yaml`. Key parameters:

- **Sample Rate**: 22.05 kHz
- **Mel Channels**: 80
- **Batch Size**: 32
- **Training Epochs**: 20,000
- **GPU Support**: Multi-GPU distributed training

#### Modifying Training Parameters
To adjust the training parameters, you must create a file identical to `datasets/ljs_base/config.yaml` inside the `downloaded_datasets` directory. Once you have created this file and configured it with your desired settings, you can proceed to execute the job. The system will automatically detect your custom configuration and apply it during the training process.

### Expected Outputs

- **Model Checkpoints**: Automatically saved to your configured S3 bucket during training
- **Local Logs**: Job execution logs in `myJob.{job_id}.err` and `output.{job_id}.out`
- **Training Logs**: Available in TensorBoard format within the container
- **Dataset**: Downloaded LJSpeech dataset stored in `downloaded_datasets/` directory

### System Requirements

- **GPU Memory**: Minimum 8GB VRAM (16GB+ recommended for multi-GPU)
- **Storage**: ~15GB for dataset and model checkpoints
- **CPU**: Multi-core recommended for data preprocessing

### Monitoring Training

Monitor training progress through:
- **Job Status**: Use `squeue -u username` to check if the job is running
- **Output Logs**: Check `output.{job_id}.out` for training progress and metrics
- **Error Logs**: Review `myJob.{job_id}.err` for any error messages
- **S3 Checkpoints**: Model checkpoints are automatically uploaded to your S3 bucket
- **Job Completion**: You'll receive email notifications when the job starts, completes, or fails

### Support and Issues

For technical issues or questions:
- Check the original [VITS2 repository](https://github.com/daniilrobnikov/vits2) for model-specific documentation
- Review container logs for debugging information
- Ensure proper GPU drivers and CUDA compatibility

---

**Note**: This implementation is optimized for HPC environments and includes containerized dependencies for reproducible training across different computing clusters.