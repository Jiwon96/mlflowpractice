import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import pandas as pd
import numpy as np
import logging
from pathlib import Path

# common.py에서 가져오기
from common import CONFIG, build_data, validate_data

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train_model(config=None):
    """모델 훈련 및 MLflow 로깅"""
    
    if config is None:
        config = CONFIG
    
    # MLflow 설정
    mlflow.set_tracking_uri(config['mlflow']['tracking_uri'])
    mlflow.set_experiment(config['project']['experiment_name'])
    
    logger.info("Starting model training...")
    
    with mlflow.start_run():
        # 1. 데이터 로딩
        logger.info("Loading and splitting data...")
        X_train, X_test, y_train, y_test = build_data(config)
        
        # 데이터 검증
        validate_data(X_train, X_test, y_train, y_test)
        
        # 2. 하이퍼파라미터 설정
        params = config.get('model', {}).get('hyperparameters', {
            'n_estimators': 100,
            'max_depth': 6,
            'random_state': 42
        })
        
        # MLflow에 파라미터 로깅
        mlflow.log_params(params)
        
        # 데이터 정보 로깅
        mlflow.log_param("train_samples", len(X_train))
        mlflow.log_param("test_samples", len(X_test))
        mlflow.log_param("features", X_train.shape[1])
        
        # 3. 모델 생성 및 훈련
        logger.info("Training RandomForest model...")
        model = RandomForestRegressor(**params)
        model.fit(X_train, y_train.values.ravel())
        
        # 4. 예측
        train_predictions = model.predict(X_train)
        test_predictions = model.predict(X_test)
        
        # 5. 평가 메트릭 계산
        train_rmse = np.sqrt(mean_squared_error(y_train, train_predictions))
        test_rmse = np.sqrt(mean_squared_error(y_test, test_predictions))
        
        train_mae = mean_absolute_error(y_train, train_predictions)
        test_mae = mean_absolute_error(y_test, test_predictions)
        
        train_r2 = r2_score(y_train, train_predictions)
        test_r2 = r2_score(y_test, test_predictions)
        
        # 6. 메트릭 로깅
        mlflow.log_metric("train_rmse", train_rmse)
        mlflow.log_metric("test_rmse", test_rmse)
        mlflow.log_metric("train_mae", train_mae)
        mlflow.log_metric("test_mae", test_mae)
        mlflow.log_metric("train_r2", train_r2)
        mlflow.log_metric("test_r2", test_r2)
        
        # 오버피팅 체크
        overfitting_score = train_rmse - test_rmse
        mlflow.log_metric("overfitting_score", overfitting_score)
        
        # 7. 모델 저장
        model_name = config['project']['model_name']
        
        # sklearn 모델로 로깅
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name=model_name,
            input_example=X_train.head(3),  # 입력 예시
            signature=mlflow.models.infer_signature(X_train, train_predictions)
        )
        
        # 8. 결과 출력
        logger.info("Training completed!")
        logger.info(f"Train RMSE: {train_rmse:.4f}")
        logger.info(f"Test RMSE: {test_rmse:.4f}")
        logger.info(f"Test R²: {test_r2:.4f}")
        
        # 9. 추가 아티팩트 저장 (선택사항)
        save_additional_artifacts(X_train, y_train, test_predictions, y_test)
        
        # 현재 실행 정보 반환
        run_id = mlflow.active_run().info.run_id
        logger.info(f"MLflow Run ID: {run_id}")
        
        return {
            'run_id': run_id,
            'model_name': model_name,
            'test_rmse': test_rmse,
            'test_r2': test_r2
        }

def save_additional_artifacts(X_train, y_train, test_predictions, y_test):
    """추가 아티팩트 저장"""
    
    # 피처 중요도 저장
    if hasattr(mlflow.active_run(), 'info'):
        import matplotlib.pyplot as plt
        
        # 간단한 예측 결과 저장
        results_df = pd.DataFrame({
            'actual': y_test.values.ravel(),
            'predicted': test_predictions,
            'residual': y_test.values.ravel() - test_predictions
        })
        
        # CSV로 저장
        results_path = "prediction_results.csv"
        results_df.to_csv(results_path, index=False)
        mlflow.log_artifact(results_path)
        
        # 간단한 통계 정보
        stats = {
            'mean_actual': float(y_test.mean()),
            'mean_predicted': float(test_predictions.mean()),
            'std_actual': float(y_test.std()),
            'std_predicted': float(test_predictions.std())
        }
        
        # JSON으로 저장
        import json
        stats_path = "model_stats.json"
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        mlflow.log_artifact(stats_path)
        
        # 임시 파일 삭제
        Path(results_path).unlink(missing_ok=True)
        Path(stats_path).unlink(missing_ok=True)

def main():
    """메인 실행 함수"""
    try:
        result = train_model()
        
        print("\n" + "="*50)
        print("Training Summary:")
        print("="*50)
        print(f"Model Name: {result['model_name']}")
        print(f"Run ID: {result['run_id']}")
        print(f"Test RMSE: {result['test_rmse']:.4f}")
        print(f"Test R²: {result['test_r2']:.4f}")
        print("\nCheck MLflow UI for detailed results:")
        print(f"http://localhost:{CONFIG['mlflow']['server_port']}")
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()