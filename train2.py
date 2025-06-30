import mlflow
import mlflow.xgboost
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from datetime import datetime
import json

# common.py에서 가져오기
from common import CONFIG, build_data, validate_data

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train_xgboost_model(config=None):
    """XGBoost 모델 훈련 및 MLflow 로깅"""
    
    if config is None:
        config = CONFIG
    
    # MLflow 설정
    mlflow.set_tracking_uri(config['mlflow']['tracking_uri'])
    
    # XGBoost 전용 실험 생성
    experiment_name = f"{config['project']['experiment_name']}_XGBoost"
    mlflow.set_experiment(experiment_name)
    
    logger.info(f"Starting XGBoost experiment: {experiment_name}")
    
    with mlflow.start_run(run_name=f"XGBoost_{datetime.now().strftime('%Y%m%d_%H%M')}"):
        # 1. 데이터 로딩
        logger.info("Loading and splitting data...")
        X_train, X_test, y_train, y_test = build_data(config)
        
        # 데이터 검증
        validate_data(X_train, X_test, y_train, y_test)
        
        # 2. XGBoost 하이퍼파라미터 설정
        xgb_params = {
            'n_estimators': 200,
            'max_depth': 8,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'random_state': 42,
            'n_jobs': -1,
            'eval_metric':'rmse',
            'early_stopping_rounds':20
        }
        
        # MLflow에 파라미터 로깅
        mlflow.log_params(xgb_params)
        
        # 데이터 정보 로깅
        mlflow.log_param("train_samples", len(X_train))
        mlflow.log_param("test_samples", len(X_test))
        mlflow.log_param("features", X_train.shape[1])
        mlflow.log_param("model_type", "XGBoost")
        mlflow.log_param("algorithm", "Gradient Boosting")
        
        # 3. XGBoost 모델 생성 및 훈련
        logger.info("Training XGBoost model...")
        model = xgb.XGBRegressor(**xgb_params)
        
        # 조기 중단을 위한 평가 세트 설정
        eval_set = [(X_train, y_train.values.ravel()), (X_test, y_test.values.ravel())]
        
        model.fit(
            X_train, 
            y_train.values.ravel(),
            eval_set=eval_set,
            
            verbose=False
        )
        
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
        
        # 6. XGBoost 특화 메트릭
        # Feature importance
        feature_importance = model.feature_importances_
        
        # Best iteration (early stopping)
        best_iteration = model.best_iteration if hasattr(model, 'best_iteration') else xgb_params['n_estimators']
        
        # 7. 메트릭 로깅
        mlflow.log_metric("train_rmse", train_rmse)
        mlflow.log_metric("test_rmse", test_rmse)
        mlflow.log_metric("train_mae", train_mae)
        mlflow.log_metric("test_mae", test_mae)
        mlflow.log_metric("train_r2", train_r2)
        mlflow.log_metric("test_r2", test_r2)
        
        # XGBoost 특화 메트릭
        mlflow.log_metric("best_iteration", best_iteration)
        mlflow.log_metric("feature_importance_mean", np.mean(feature_importance))
        mlflow.log_metric("feature_importance_std", np.std(feature_importance))
        
        # 오버피팅 체크
        overfitting_score = train_rmse - test_rmse
        mlflow.log_metric("overfitting_score", overfitting_score)
        
        # 8. 모델 저장
        model_name = f"{config['project']['model_name']}_XGBoost"
        
        # XGBoost 모델로 로깅
        mlflow.xgboost.log_model(
            xgb_model=model,
            artifact_path="model",
            registered_model_name=model_name,
            input_example=X_train.head(3),
            signature=mlflow.models.infer_signature(X_train, train_predictions)
        )
        
        # 9. XGBoost 특화 아티팩트 저장
        save_xgboost_artifacts(model, X_train, y_train, test_predictions, y_test, feature_importance)
        
        # 10. 결과 출력
        logger.info("XGBoost training completed!")
        logger.info(f"Train RMSE: {train_rmse:.4f}")
        logger.info(f"Test RMSE: {test_rmse:.4f}")
        logger.info(f"Test R²: {test_r2:.4f}")
        logger.info(f"Best Iteration: {best_iteration}")
        
        # 현재 실행 정보 반환
        run_id = mlflow.active_run().info.run_id
        logger.info(f"MLflow Run ID: {run_id}")
        
        return {
            'run_id': run_id,
            'model_name': model_name,
            'test_rmse': test_rmse,
            'test_r2': test_r2,
            'test_mae': test_mae,
            'best_iteration': best_iteration,
            'overfitting_score': overfitting_score
        }

def save_xgboost_artifacts(model, X_train, y_train, test_predictions, y_test, feature_importance):
    """XGBoost 특화 아티팩트 저장"""
    
    # 1. 예측 결과 저장
    results_df = pd.DataFrame({
        'actual': y_test.values.ravel(),
        'predicted': test_predictions,
        'residual': y_test.values.ravel() - test_predictions,
        'abs_residual': np.abs(y_test.values.ravel() - test_predictions)
    })
    
    results_path = "xgboost_predictions.csv"
    results_df.to_csv(results_path, index=False)
    mlflow.log_artifact(results_path)
    
    # 2. Feature Importance 저장
    if hasattr(X_train, 'columns'):
        feature_names = X_train.columns.tolist()
    else:
        feature_names = [f'feature_{i}' for i in range(X_train.shape[1])]
    
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': feature_importance
    }).sort_values('importance', ascending=False)
    
    importance_path = "feature_importance.csv"
    importance_df.to_csv(importance_path, index=False)
    mlflow.log_artifact(importance_path)
    
    # 3. XGBoost 상세 통계
    stats = {
        'model_type': 'XGBoost',
        'training_stats': {
            'best_iteration': int(model.best_iteration) if hasattr(model, 'best_iteration') else None,
            'n_features': len(feature_names),
            'total_boost_rounds': model.n_estimators
        },
        'prediction_stats': {
            'mean_actual': float(y_test.mean()),
            'mean_predicted': float(test_predictions.mean()),
            'std_actual': float(y_test.std()),
            'std_predicted': float(test_predictions.std()),
            'max_residual': float(np.max(np.abs(y_test.values.ravel() - test_predictions))),
            'residual_std': float(np.std(y_test.values.ravel() - test_predictions))
        },
        'feature_importance_stats': {
            'top_3_features': importance_df.head(3).to_dict('records'),
            'importance_concentration': float(importance_df.head(5)['importance'].sum() / importance_df['importance'].sum())
        }
    }
    
    stats_path = "xgboost_detailed_stats.json"
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    mlflow.log_artifact(stats_path)
    
    # 4. 학습 곡선 정보 (가능한 경우)
    if hasattr(model, 'evals_result_'):
        evals_result = model.evals_result_
        training_curve = {
            'train_rmse': evals_result.get('validation_0', {}).get('rmse', []),
            'test_rmse': evals_result.get('validation_1', {}).get('rmse', [])
        }
        
        curve_path = "training_curve.json"
        with open(curve_path, 'w') as f:
            json.dump(training_curve, f, indent=2)
        mlflow.log_artifact(curve_path)
    
    # 임시 파일 삭제
    for path in [results_path, importance_path, stats_path]:
        Path(path).unlink(missing_ok=True)
    
    if 'curve_path' in locals():
        Path(curve_path).unlink(missing_ok=True)

def main():
    """메인 실행 함수"""
    try:
        result = train_xgboost_model()
        
        print("\n" + "="*60)
        print("XGBoost Training Summary:")
        print("="*60)
        print(f"Model Name: {result['model_name']}")
        print(f"Run ID: {result['run_id']}")
        print(f"Test RMSE: {result['test_rmse']:.4f}")
        print(f"Test R²: {result['test_r2']:.4f}")
        print(f"Test MAE: {result['test_mae']:.4f}")
        print(f"Best Iteration: {result['best_iteration']}")
        print(f"Overfitting Score: {result['overfitting_score']:.4f}")
        print("\nCheck MLflow UI for detailed results:")
        print(f"http://localhost:{CONFIG['mlflow']['server_port']}")
        
    except Exception as e:
        logger.error(f"XGBoost training failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()