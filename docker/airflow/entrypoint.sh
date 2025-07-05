#!/bin/bash
set -e

# DB 마이그레이션
airflow db migrate

# 관리자 계정 생성 (이미 존재하면 무시)
airflow users create \
    --username ${AIRFLOW_ADMIN_USERNAME} \
    --password ${AIRFLOW_ADMIN_PASSWORD} \
    --firstname Admin \
    --lastname User \
    --role Admin \
    --email ${AIRFLOW_ADMIN_EMAIL} || true

# 웹서버 시작
exec airflow webserver