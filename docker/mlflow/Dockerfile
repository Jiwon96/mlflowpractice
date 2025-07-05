FROM python:3.9

WORKDIR /app

RUN pip install mlflow psycopg2-binary

EXPOSE 5000

CMD mlflow server \
    --backend-store-uri postgresql://mlflow:${POSTGRES_MLFLOW_PASSWORD}@postgres-mlflow:5432/mlflow \
    --default-artifact-root ./mlruns \
    --host 0.0.0.0 \
    --port 5000