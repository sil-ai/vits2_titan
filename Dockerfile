FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-devel


# Establecer el directorio de trabajo
WORKDIR /app

# Eliminar archivos de repositorios preexistentes para evitar conflictos
RUN rm -f /etc/apt/sources.list.d/cuda.list /etc/apt/sources.list.d/nvidia-ml.list

# Instalar herramientas necesarias para manejar claves GPG
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    gnupg \
    && rm -rf /var/lib/apt/lists/*

# Descargar e instalar la clave GPG correcta para el repositorio de CUDA
RUN curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub | gpg --dearmor -o /usr/share/keyrings/nvidia-archive-keyring.gpg

# Descargar e instalar la clave GPG correcta para el repositorio de machine learning
RUN curl -fsSL https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub | gpg --dearmor -o /usr/share/keyrings/nvidia-ml-archive-keyring.gpg

# Configurar el repositorio de CUDA con la clave correcta
RUN echo "deb [signed-by=/usr/share/keyrings/nvidia-archive-keyring.gpg] https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/ /" > /etc/apt/sources.list.d/cuda.list

# Configurar el repositorio de machine learning con la clave correcta
RUN echo "deb [signed-by=/usr/share/keyrings/nvidia-ml-archive-keyring.gpg] https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/ /" > /etc/apt/sources.list.d/nvidia-ml.list

# Instalar dependencias del sistema
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

# RUN apt-get update && apt-get install -y \
#     cuda-nvrtc-11-7 \
#     cuda-nvcc-11-7 \
#     cuda-nvvp-11-7 \
#     libcublas-dev-11-7 \
#     libnvvm-samples-11-7 \
#     libnvvm3 \
#     && rm -rf /var/lib/apt/lists/*


# Clonar el repositorio
RUN git clone https://github.com/sil-ai/vits2_titan.git /app

# Instalar dependencias de Python
RUN pip install --no-cache-dir -r /app/requirements.txt

# Crear directorio para datasets
RUN mkdir -p /app/downloaded_datasets

# Configurar variables de entorno
ENV PYTHONPATH="/app"
ENV DATASET_PATH="/app/downloaded_datasets/LJSpeech-1.1"

# Asegurar permisos correctos
RUN chmod -R 777 /app

# Comando por defecto
CMD ["make", "all"]