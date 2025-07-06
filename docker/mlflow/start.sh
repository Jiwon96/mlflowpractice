#!/bin/bash
echo "환경변수 확인:"
echo "AWS_ACCESS_KEY_ID: ${AWS_ACCESS_KEY_ID}"
echo "MLFLOW_S3_ENDPOINT_URL: ${MLFLOW_S3_ENDPOINT_URL}"

export AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID}
export AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY}

mlflow server \
    --host 0.0.0.0 \
    --port 5000 \
    --backend-store-uri postgresql://mlflow:${POSTGRES_MLFLOW_PASSWORD}@postgres-mlflow:5432/mlflow \
    --default-artifact-root s3://mlflow/