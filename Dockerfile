FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime
# FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu20.04 nueva version
# FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-devel



WORKDIR /app

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    python3-dev \
    python3-pip \
    git \
    espeak-ng \
    libsndfile1 \
    ffmpeg \
    wget \
    bzip2 \
    && rm -rf /var/lib/apt/lists/*

# Instalar requisitos de Python primero (para aprovechar la caché)
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Copiar todo el código fuente
COPY . /app/

# Asegurarse de que el Makefile existe y tiene permisos adecuados
RUN ls -la /app/Makefile && chmod +x /app/Makefile

# Variables de entorno
ENV PYTHONPATH="/app"
ENV DATASET_PATH="/app/downloaded_datasets/LJSpeech-1.1"
RUN mkdir -p /app/datasets_local
EXPOSE 6006

RUN chmod -R 777 /app

CMD ["all"]
