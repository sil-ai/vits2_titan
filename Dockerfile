FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime

# Establecer el directorio de trabajo específico para el usuario aquintero
WORKDIR /app

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

RUN git clone https://github.com/sil-ai/vits2_titan.git /app

RUN pip install --no-cache-dir -r requirements.txt

RUN mkdir -p /app/downloaded_datasets
ENV PYTHONPATH="/app"
ENV DATASET_PATH="/app/downloaded_datasets/LJSpeech-1.1"

RUN chmod -R 777 /app

CMD ["make", "all"]
