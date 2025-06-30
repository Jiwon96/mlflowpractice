# serve_model.py (수정된 버전)
import subprocess
import sys
import mlflow
import requests
import os
from common import CONFIG

def check_tracking_server():
    """MLflow tracking server 연결 상태 확인"""
    tracking_uri = CONFIG['mlflow']['tracking_uri']
    try:
        response = requests.get(f"{tracking_uri}/health", timeout=5)
        if response.status_code == 200:
            print(f"✅ MLflow tracking server is running at {tracking_uri}")
            return True
        else:
            print(f"❌ MLflow tracking server responded with status {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print(f"❌ Cannot connect to MLflow tracking server at {tracking_uri}")
        print("Please start the tracking server first:")
        print("mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --host 0.0.0.0 --port 5000")
        return False
    except Exception as e:
        print(f"❌ Error checking tracking server: {e}")
        return False

def serve_model():
    """훈련된 모델을 REST API로 서빙"""
    # 먼저 tracking server 상태 확인
    if not check_tracking_server():
        return
        
    model_name = CONFIG['project']['model_name']
    model_stage = CONFIG['mlflow']['model_stage']
    tracking_uri = CONFIG['mlflow']['tracking_uri']
    
    # MLflow 추적 URI 설정
    mlflow.set_tracking_uri(tracking_uri)
    
    # 모델 URI 설정
    model_uri = f"models:/{model_name}/{model_stage}"
    
    print(f"Serving model: {model_uri}")
    print("Starting MLflow model server...")
    
    # 환경변수 설정 (subprocess에 전달)
    env = os.environ.copy()
    env['MLFLOW_TRACKING_URI'] = tracking_uri
    
    # 명령줄로 서빙 실행
    cmd = [
        "mlflow", "models", "serve",
        "-m", model_uri,
        "--host", "0.0.0.0",
        "--port", "8081",
        "--no-conda"
    ]
    
    print(f"Command: {' '.join(cmd)}")
    print(f"Environment: MLFLOW_TRACKING_URI={tracking_uri}")
    print("Server will be available at: http://localhost:8081")
    print("Press Ctrl+C to stop the server")
    
    try:
        # 환경변수와 함께 서버 실행
        subprocess.run(cmd, env=env)
    except KeyboardInterrupt:
        print("\nServer stopped.")
    except Exception as e:
        print(f"Error starting server: {e}")

if __name__ == "__main__":
    serve_model()