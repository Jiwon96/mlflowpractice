# serve_model.py (수정된 버전)
import subprocess
import sys
import mlflow
from common import CONFIG

def serve_model():
    """훈련된 모델을 REST API로 서빙"""
    model_name = CONFIG['project']['model_name']
    model_stage = CONFIG['mlflow']['model_stage']
    
    # MLflow 추적 URI 설정
    mlflow.set_tracking_uri(CONFIG['mlflow']['tracking_uri'])
    
    # 모델 URI 설정
    model_uri = f"models:/{model_name}/{model_stage}"
    
    print(f"Serving model: {model_uri}")
    print("Starting MLflow model server...")
    
    # 명령줄로 서빙 실행
    cmd = [
        "mlflow", "models", "serve",
        "-m", model_uri,
        "--host", "0.0.0.0",
        "--port", "8080",
        "--no-conda"
    ]
    
    print(f"Command: {' '.join(cmd)}")
    print("Server will be available at: http://localhost:8080")
    print("Press Ctrl+C to stop the server")
    
    try:
        # 서버 실행
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\nServer stopped.")
    except Exception as e:
        print(f"Error starting server: {e}")

if __name__ == "__main__":
    serve_model()