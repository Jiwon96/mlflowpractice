# model_registry.py
import mlflow
from mlflow.tracking import MlflowClient
from common import CONFIG

def register_best_model():
    """최고 성능 모델을 Model Registry에 등록"""
    client = MlflowClient()
    experiment_name = CONFIG['project']['experiment_name']
    model_name = CONFIG['project']['model_name']
    
    # 실험에서 최고 성능 모델 찾기
    experiment = mlflow.get_experiment_by_name(experiment_name)
    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
    best_run = runs.loc[runs['metrics.test_rmse'].idxmin()]
    
    print(f"Registering best model from run: {best_run['run_id']}")
    
    # 모델 등록
    model_uri = f"runs:/{best_run['run_id']}/model"
    
    try:
        # 새 버전으로 등록
        model_version = mlflow.register_model(
            model_uri=model_uri,
            name=model_name
        )
        
        print(f"Model registered as version {model_version.version}")
        
        # Staging으로 이동
        client.transition_model_version_stage(
            name=model_name,
            version=model_version.version,
            stage="Staging"
        )
        
        print(f"Model version {model_version.version} moved to Staging")
        
        return model_version
        
    except Exception as e:
        print(f"Error registering model: {e}")
        return None

def promote_to_production(version=None):
    """모델을 Production 스테이지로 승격"""
    client = MlflowClient()
    model_name = CONFIG['project']['model_name']
    
    if version is None:
        # 최신 Staging 모델 찾기
        staging_models = client.get_latest_versions(model_name, stages=["Staging"])
        if not staging_models:
            print("No models in Staging stage")
            return
        version = staging_models[0].version
    
    # Production으로 이동
    client.transition_model_version_stage(
        name=model_name,
        version=version,
        stage="Production"
    )
    
    print(f"Model version {version} promoted to Production")

def list_model_versions():
    """등록된 모델 버전들 조회"""
    client = MlflowClient()
    model_name = CONFIG['project']['model_name']
    
    try:
        versions = client.search_model_versions(f"name='{model_name}'")
        
        print("=== Model Versions ===")
        for version in versions:
            print(f"Version: {version.version}")
            print(f"Stage: {version.current_stage}")
            print(f"Status: {version.status}")
            print(f"Run ID: {version.run_id}")
            print("-" * 30)
            
    except Exception as e:
        print(f"Error listing models: {e}")

if __name__ == "__main__":
    print("1. Listing current model versions...")
    list_model_versions()
    
    print("\n2. Registering best model...")
    model_version = register_best_model()
    
    if model_version:
        print("\n3. Promoting to production...")
        promote_to_production(model_version.version)
        
        print("\n4. Updated model versions:")
        list_model_versions()