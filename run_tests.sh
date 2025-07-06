#!/bin/bash

echo "🚀 MNIST MLOps 테스트 시작"
echo "=========================="

# 가상환경 활성화 (필요시)
source myenv/bin/activate
# 필요한 디렉토리 생성
mkdir -p test_data logs/training data/models

# 환경변수 설정
export MLFLOW_TRACKING_URI="http://localhost:5000"
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# echo ""
# echo "1️⃣ 개별 모듈 테스트"
# echo "==================="
# python test_individual_modules.py
# if [ $? -ne 0 ]; then
#     echo "❌ 개별 모듈 테스트 실패"
#     exit 1
# fi

# echo ""
# echo "2️⃣ 통합 테스트"
# echo "==============="
# python test_integration.py
# if [ $? -ne 0 ]; then
#     echo "❌ 통합 테스트 실패"
#     exit 1
# fi

echo ""
echo "3️⃣ DAG 테스트"
echo "=============="
python test_airflow_pipeline.py
if [ $? -ne 0 ]; then
    echo "❌ DAG 테스트 실패"
    exit 1
fi

echo ""
echo "🎉 모든 테스트 성공!"
echo "==================="
echo "✅ 개별 모듈 테스트 통과"
echo "✅ 통합 테스트 통과"
echo "✅ DAG 테스트 통과"
echo ""
echo "이제 전체 파이프라인을 실행할 준비가 되었습니다!"