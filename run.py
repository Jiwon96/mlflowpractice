# run.py (새로 생성)
import subprocess
import time
import sys
from common import CONFIG

def start_mlflow_server():
    """config에서 읽은 포트로 MLflow 서버 시작"""
    port = CONFIG['mlflow']['server_port']
    tracking_uri = CONFIG['mlflow']['tracking_uri']
    
    print(f"Starting MLflow server on port {port}")
    print(f"Tracking URI: {tracking_uri}")
    
    # 백그라운드에서 서버 시작
    process = subprocess.Popen([
        'mlflow', 'server', 
        '--host', '0.0.0.0', 
        '--port', str(port)
    ])
    
    # 서버 시작 대기
    time.sleep(5)
    return process

def main():
    # 1. MLflow 서버 시작
    server_process = start_mlflow_server()
    
    try:
        # 2. 훈련 실행
        from train import main as train_main
        train_main()
    finally:
        # 3. 서버 종료
        server_process.terminate()

if __name__ == "__main__":
    main()