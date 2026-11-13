FROM python:3.11-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    EASYOCR_MODULE_PATH=/root/.EasyOCR \
    HF_HOME=/root/.cache/huggingface

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

RUN python -c "\
import easyocr; \
easyocr.Reader(['ru', 'en'], gpu=False, download_enabled=True); \
from sentence_transformers import SentenceTransformer; \
SentenceTransformer('BAAI/bge-m3'); \
"

COPY . .

CMD ["python", "main.py"]