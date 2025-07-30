#!/usr/bin/env bash
# Source the .env file to load variables, but no committed
if [ -f .env ]; then
  export $(grep -v '^#' .env | xargs)
fi

docker build -t intent-pipeline:latest .
docker run --rm \
  -e HF_TOKEN ="$HF_TOKEN" \
  -e DB_HOST = "host.docker.internal" \
  -v "$(pwd)/artifacts:/app/artifacts" \
  -p 8501:8501 \
  intent-pipeline:latest