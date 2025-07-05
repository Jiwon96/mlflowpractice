#!/usr/bin/env python3
"""
Airflow DAG 로컬 테스트
Docker 없이 DAG 로직 검증
"""

import os
import sys
from datetime import datetime

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_dag_tasks():
    """DAG 태스크들을 순차적으로 테스트"""
    print("Airflow DAG 로컬 테스트")
    print("=" * 50)
    
    # DAG 함수들 import
    from dags.mnist_pipeline import (
        download_and_validate_data,
        preprocess_data,
        train_model,
        evaluate_model,
        deploy_model
    )
    
    # Mock context 생성
    class MockTaskInstance:
        def __init__(self):
            self.xcom_data = {}
        
        def xcom_pull(self, task_ids):
            return self.xcom_data.get(task_ids, {})
        
        def xcom_push(self, key, value):
            self.xcom_data[key] = value
    
    mock_ti = MockTaskInstance()
    context = {'task_instance': mock_ti}
    
    try:
        # 1. 데이터 다운로드 및 검증
        print("1️⃣ 데이터 다운로드 및 검증 태스크...")
        result1 = download_and_validate_data(**context)
        mock_ti.xcom_data['download_and_validate_data'] = result1
        print("✅ 완료")
        
        # 2. 데이터 전처리
        print("2️⃣ 데이터 전처리 태스크...")
        result2 = preprocess_data(**context)
        mock_ti.xcom_data['preprocess_data'] = result2
        print("✅ 완료")
        
        # 3. 모델 훈련 (빠른 버전)
        print("3️⃣ 모델 훈련 태스크...")
        # 설정 파일 수정하여 빠른 훈련
        os.environ['QUICK_TEST'] = 'true'
        result3 = train_model(**context)
        mock_ti.xcom_data['train_model'] = result3
        print("✅ 완료")
        
        print("\n🎉 DAG 태스크 테스트 성공!")
        return True
        
    except Exception as e:
        print(f"❌ DAG 테스트 실패: {e}")
        return False

if __name__ == "__main__":
    success = test_dag_tasks()
    sys.exit(0 if success else 1)