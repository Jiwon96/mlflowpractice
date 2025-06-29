# evaluate.py
import mlflow
import mlflow.tracking
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import pandas as pd
from common import CONFIG, build_data
mlflow.set_tracking_uri(CONFIG['mlflow']['tracking_uri'])
print(f"MLflow Tracking URI: {mlflow.get_tracking_uri()}")
def evaluate_models():
    """여러 실험의 모델들을 비교 평가"""
    experiment_name = CONFIG['project']['experiment_name']
    
    # 실험의 모든 run 가져오기
    experiment = mlflow.get_experiment_by_name(experiment_name)
    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
    
    print("=== Model Comparison ===")
    print(runs[['run_id', 'metrics.test_rmse', 'metrics.test_r2', 'status']].head())
    
    # 최고 성능 모델 찾기
    best_run = runs.loc[runs['metrics.test_rmse'].idxmin()]
    print(f"\nBest Model:")
    print(f"Run ID: {best_run['run_id']}")
    print(f"Test RMSE: {best_run['metrics.test_rmse']:.4f}")
    print(f"Test R²: {best_run['metrics.test_r2']:.4f}")
    
    return best_run

def evaluate_current_production_model():
    """현재 프로덕션 모델 상세 평가"""
    model_name = CONFIG['project']['model_name']
    model_uri = f"models:/{model_name}/Production"
    
    try:
        # 모델 로드
        model = mlflow.pyfunc.load_model(model_uri)
        
        # 테스트 데이터로 평가
        X_train, X_test, y_train, y_test = build_data()
        predictions = model.predict(X_test)
        
        # 상세 메트릭 계산
        rmse = mean_squared_error(y_test, predictions, squared=False)
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        
        print("=== Production Model Evaluation ===")
        print(f"RMSE: {rmse:.4f}")
        print(f"MAE: {mae:.4f}")
        print(f"R²: {r2:.4f}")
        
        # 예측 결과 분석
        results_df = pd.DataFrame({
            'actual': y_test.values.ravel(),
            'predicted': predictions,
            'residual': y_test.values.ravel() - predictions
        })
        
        print("\nPrediction Analysis:")
        print(results_df.describe())
        
        return results_df
        
    except Exception as e:
        print(f"No production model found: {e}")
        return None

if __name__ == "__main__":
    evaluate_models()
    print("\n" + "="*50)
    evaluate_current_production_model()