FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime

# Establecer el directorio de trabajo específico para el usuario aquintero
WORKDIR /home/aquintero/vits2_train

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

RUN git clone https://github.com/sil-ai/vits2_titan.git .

RUN pip install --no-cache-dir -r requirements.txt

ENV PYTHONPATH="/home/aquintero/vits2_train"
ENV DATASET_PATH="/home/aquintero/vits2_train/downloaded_datasets/LJSpeech-1.1"

RUN chmod -R 777 /home/aquintero/vits2_train

CMD ["make", "all"]
