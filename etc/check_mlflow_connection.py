# check_mlflow_connection.py
import mlflow
import requests
from common import CONFIG

def check_mlflow_connection():
    """MLflow 서버 연결 상태 확인"""
    
    tracking_uri = CONFIG['mlflow']['tracking_uri']
    print(f"Config Tracking URI: {tracking_uri}")
    
    # 현재 MLflow 설정 확인
    current_uri = mlflow.get_tracking_uri()
    print(f"Current MLflow URI: {current_uri}")
    
    # URI 설정
    mlflow.set_tracking_uri(tracking_uri)
    updated_uri = mlflow.get_tracking_uri()
    print(f"Updated MLflow URI: {updated_uri}")
    
    # 서버 연결 테스트
    try:
        # HTTP 요청으로 서버 상태 확인
        response = requests.get(f"{tracking_uri}/health")
        if response.status_code == 200:
            print("✅ MLflow server is running and accessible")
        else:
            print(f"❌ MLflow server responded with status: {response.status_code}")
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to MLflow server")
        print("Make sure MLflow server is running:")
        print(f"   mlflow server --host 0.0.0.0 --port {CONFIG['mlflow']['server_port']}")
        return False
    except Exception as e:
        print(f"❌ Error connecting to MLflow: {e}")
        return False
    
    # 실험 목록 확인
    try:
        experiments = mlflow.search_experiments()
        print(f"✅ Found {len(experiments)} experiments")
        for exp in experiments:
            print(f"   - {exp.name} (ID: {exp.experiment_id})")
    except Exception as e:
        print(f"❌ Error accessing experiments: {e}")
        return False
    
    return True

if __name__ == "__main__":
    if check_mlflow_connection():
        print("\n🎉 MLflow connection is working properly!")
    else:
        print("\n💥 MLflow connection failed. Please fix the issues above.")