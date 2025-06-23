FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel
# FROM nvidia/cuda:12.4.0-devel-ubuntu20.04


# set working directory
WORKDIR /app

# remove existing repository files to avoid conflicts
RUN rm -f /etc/apt/sources.list.d/cuda.list /etc/apt/sources.list.d/nvidia-ml.list

# install necessary tools to handle GPG keys
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    gnupg \
    && rm -rf /var/lib/apt/lists/*

# download and install the correct GPG key for the CUDA repository
RUN curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub | gpg --dearmor -o /usr/share/keyrings/nvidia-archive-keyring.gpg

# download and install the GPG key for the machine learning repository
RUN curl -fsSL https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu2004/x86_64/7fa2af80.pub | gpg --dearmor -o /usr/share/keyrings/nvidia-ml-archive-keyring.gpg

# configure the repositories
RUN echo "deb [signed-by=/usr/share/keyrings/nvidia-archive-keyring.gpg] https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/ /" > /etc/apt/sources.list.d/cuda.list
RUN echo "deb [signed-by=/usr/share/keyrings/nvidia-ml-archive-keyring.gpg] https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu2004/x86_64/ /" > /etc/apt/sources.list.d/nvidia-ml.list
# install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    python3-dev \
    python3-pip \
    git \
    make \
    espeak-ng \
    libsndfile1 \
    ffmpeg \
    wget \
    bzip2 \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y \
    cuda-nvrtc-12-4 \
    cuda-nvcc-12-4 \
    # cuda-libraries-12-4 \
    # cuda-libraries-dev-12-4 \
    # libcublas-12-4 \
    # libcublas-dev-12-4 \
    && rm -rf /var/lib/apt/lists/*

# Copy only the requirements file first to leverage Docker cache
COPY requirements.txt .

# install python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# create directory for datasets
RUN mkdir -p /app/downloaded_datasets

# config environment variables
ENV PYTHONPATH="/app"
ENV DATASET_PATH="/app/downloaded_datasets/LJSpeech-1.1"

ENV CUDA_HOME="/usr/local/cuda-12.4"
ENV PATH="${CUDA_HOME}/bin:${PATH}"
ENV LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"
# This variable tells Numba where to find the CUDA driver
ENV NUMBA_CUDA_DRIVER="/usr/local/cuda-12.4/compat/libcuda.so.550.54.15"
# ensure correct permissions
RUN chmod -R 777 /app

# default command
CMD ["make", "all"]