# 1. Base image
FROM python:3.10-slim

# 2. Set working directory
WORKDIR /app

# 3. Install system deps
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    default-libmysqlclient-dev && \ 
    rm -rf /var/lib/apt/lists/*

# 4. Copy and install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy project code
COPY . .

# 6. Set environment variables defaults
ENV HF_HOME=/root/.cache/huggingface \
    DB_HOST=host.docker.internal

# 7. Entrypoint: run the full pipeline
ENTRYPOINT ["bash", "-lc", \
  "python ingest_clean.py && \
   python prepare_data.py && \
   python finetune_bert.py && \
   python export_bert_metrics.py && \
   python app.py"]
