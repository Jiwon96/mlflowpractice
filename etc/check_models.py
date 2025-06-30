import mlflow
from common import CONFIG
from datetime import datetime

def check_models():
    """MLflow에 등록된 모델들 확인"""
    
    mlflow.set_tracking_uri(CONFIG['mlflow']['tracking_uri'])
    client = mlflow.tracking.MlflowClient()
    
    print("="*60)
    print("CHECKING MLFLOW MODELS")
    print("="*60)
    
    try:
        models = client.search_registered_models()
        
        if not models:
            print("❌ No registered models found!")
            return False
        
        expected_models = [
            CONFIG['project']['model_name'],  # e2e-ml-pipeline
            f"{CONFIG['project']['model_name']}_XGBoost"  # e2e-ml-pipeline_XGBoost
        ]
        
        found_models = []
        
        for model in models:
            model_name = model.name
            versions = client.get_latest_versions(model_name)
            latest_version = max(versions, key=lambda x: int(x.version)) if versions else None
            
            print(f"📦 Model: {model_name}")
            if latest_version:
                creation_time = datetime.fromtimestamp(int(latest_version.creation_timestamp)/1000)
                print(f"   Version: {latest_version.version}")
                print(f"   Created: {creation_time.strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"   Status: {'✅ Ready for comparison' if model_name in expected_models else '🔍 Other model'}")
            else:
                print(f"   Status: ❌ No versions found")
            print()
            
            if model_name in expected_models:
                found_models.append(model_name)
        
        print("-"*60)
        print("COMPARISON READINESS:")
        print("-"*60)
        
        for expected in expected_models:
            if expected in found_models:
                print(f"✅ {expected} - Ready")
            else:
                print(f"❌ {expected} - Missing")
                if 'XGBoost' in expected:
                    print(f"   → Run 'python train2.py' to create this model")
                else:
                    print(f"   → Run 'python train.py' to create this model")
        
        if len(found_models) >= 2:
            print(f"\n🚀 Ready for comparison! Run 'python compare_models.py'")
            return True
        else:
            print(f"\n⚠️  Need {2 - len(found_models)} more model(s) for comparison")
            return False
            
    except Exception as e:
        print(f"❌ Error checking models: {str(e)}")
        return False

def check_experiments():
    """실험 목록도 함께 확인"""
    
    print("\n" + "="*60)
    print("CHECKING MLFLOW EXPERIMENTS")
    print("="*60)
    
    try:
        client = mlflow.tracking.MlflowClient()
        experiments = client.search_experiments()
        
        expected_experiments = [
            CONFIG['project']['experiment_name'],  # e2e-ml-pipeline
            f"{CONFIG['project']['experiment_name']}_XGBoost",  # e2e-ml-pipeline_XGBoost
        ]
        
        found_experiments = []
        
        for exp in experiments:
            if exp.name != "Default":  # Default 실험 제외
                print(f"🧪 Experiment: {exp.name}")
                print(f"   ID: {exp.experiment_id}")
                
                # 실험 내 실행 수 확인
                runs = client.search_runs([exp.experiment_id])
                print(f"   Runs: {len(runs)}")
                print(f"   Status: {'✅ Active' if exp.name in expected_experiments else '🔍 Other'}")
                print()
                
                if exp.name in expected_experiments:
                    found_experiments.append(exp.name)
        
        print("-"*60)
        for expected in expected_experiments:
            if expected in found_experiments:
                print(f"✅ {expected} - Exists")
            else:
                print(f"❌ {expected} - Missing")
        
    except Exception as e:
        print(f"❌ Error checking experiments: {str(e)}")

if __name__ == "__main__":
    models_ready = check_models()
    check_experiments()
    
    print("\n" + "="*60)
    print("NEXT STEPS:")
    print("="*60)
    
    if models_ready:
        print("🎉 All models are ready for comparison!")
        print("   Run: python compare_models.py")
    else:
        print("📝 To create missing models:")
        print("   1. For RandomForest: python train.py")
        print("   2. For XGBoost: python train2.py")
        print("   3. For comparison: python compare_models.py")