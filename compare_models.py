import mlflow
import mlflow.sklearn
import mlflow.xgboost
import mlflow.pyfunc
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import pandas as pd
import numpy as np
import logging
from datetime import datetime
import json
import os
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import seaborn as sns

# common.py에서 가져오기
from common import CONFIG, build_data, validate_data

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelComparator:
    """MLflow에 등록된 모델들을 비교하는 클래스"""
    
    def __init__(self, config=None):
        self.config = config if config else CONFIG
        mlflow.set_tracking_uri(self.config['mlflow']['tracking_uri'])
        self.client = mlflow.tracking.MlflowClient()
        
        # 테스트 데이터 로드
        _, self.X_test, _, self.y_test = build_data(self.config)
        logger.info(f"Loaded test data: {len(self.X_test)} samples")
    
    def load_model_by_name(self, model_name: str, version: Optional[str] = None, stage: Optional[str] = None) -> Tuple[object, str]:
        """모델명으로 모델 로드"""
        try:
            if version:
                model_uri = f"models:/{model_name}/{version}"
                model_version = version
            elif stage:
                model_versions = self.client.get_latest_versions(model_name, stages=[stage])
                if not model_versions:
                    raise ValueError(f"No model found in stage '{stage}' for {model_name}")
                model_uri = f"models:/{model_name}/{model_versions[0].version}"
                model_version = model_versions[0].version
            else:
                # 최신 버전 가져오기
                model_versions = self.client.get_latest_versions(model_name)
                if not model_versions:
                    raise ValueError(f"No versions found for model {model_name}")
                latest_version = max(model_versions, key=lambda x: int(x.version))
                model_uri = f"models:/{model_name}/{latest_version.version}"
                model_version = latest_version.version
            
            # 모델 정보 가져오기
            model_info = self.client.get_model_version(model_name, model_version)
            
            # 모델 타입 감지 (여러 방법으로 시도)
            model_loaded = False
            model = None
            
            # 1. 모델명에서 타입 추론
            if 'xgboost' in model_name.lower() or 'xgb' in model_name.lower():
                try:
                    model = mlflow.xgboost.load_model(model_uri)
                    model_loaded = True
                    logger.info(f"Loaded as XGBoost model: {model_name} v{model_version}")
                except Exception as e:
                    logger.debug(f"Failed to load as XGBoost: {str(e)}")
            
            # 2. sklearn으로 시도
            if not model_loaded:
                try:
                    model = mlflow.sklearn.load_model(model_uri)
                    model_loaded = True
                    logger.info(f"Loaded as sklearn model: {model_name} v{model_version}")
                except Exception as e:
                    logger.debug(f"Failed to load as sklearn: {str(e)}")
            
            # 3. 일반적인 MLflow 모델로 시도
            if not model_loaded:
                try:
                    model = mlflow.pyfunc.load_model(model_uri)
                    model_loaded = True
                    logger.info(f"Loaded as pyfunc model: {model_name} v{model_version}")
                except Exception as e:
                    logger.debug(f"Failed to load as pyfunc: {str(e)}")
            
            # 4. XGBoost로 다시 시도 (모든 XGBoost 모델용)
            if not model_loaded:
                try:
                    model = mlflow.xgboost.load_model(model_uri)
                    model_loaded = True
                    logger.info(f"Loaded as XGBoost model (fallback): {model_name} v{model_version}")
                except Exception as e:
                    logger.debug(f"Failed to load as XGBoost (fallback): {str(e)}")
            
            if not model_loaded:
                raise ValueError(f"Could not load model {model_name} with any available flavor")
            
            return model, model_version
            
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {str(e)}")
            raise
    
    def evaluate_model(self, model, model_name: str, model_version: str) -> Dict:
        """단일 모델 평가"""
        try:
            # 예측
            predictions = model.predict(self.X_test)
            
            # 메트릭 계산
            rmse = np.sqrt(mean_squared_error(self.y_test, predictions))
            mae = mean_absolute_error(self.y_test, predictions)
            r2 = r2_score(self.y_test, predictions)
            
            # 추가 통계
            residuals = self.y_test.values.ravel() - predictions
            
            metrics = {
                'model_name': model_name,
                'model_version': model_version,
                'rmse': rmse,
                'mae': mae,
                'r2': r2,
                'mean_absolute_residual': np.mean(np.abs(residuals)),
                'max_absolute_residual': np.max(np.abs(residuals)),
                'residual_std': np.std(residuals),
                'predictions': predictions,
                'residuals': residuals
            }
            
            logger.info(f"{model_name} v{model_version} - RMSE: {rmse:.4f}, R²: {r2:.4f}")
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to evaluate model {model_name}: {str(e)}")
            raise
    
    def compare_models(self, model_configs: List[Dict]) -> Dict:
        """여러 모델 비교
        
        Args:
            model_configs: [
                {'name': 'model_name', 'version': '1', 'stage': None},
                {'name': 'model_name_XGBoost', 'version': None, 'stage': None}
            ]
        """
        results = {}
        
        logger.info(f"Starting comparison of {len(model_configs)} models...")
        
        for config in model_configs:
            model_name = config['name']
            version = config.get('version')
            stage = config.get('stage')
            
            try:
                # 모델 로드
                model, model_version = self.load_model_by_name(model_name, version, stage)
                
                # 평가
                metrics = self.evaluate_model(model, model_name, model_version)
                results[f"{model_name}_v{model_version}"] = metrics
                
            except Exception as e:
                logger.warning(f"Skipping model {model_name}: {str(e)}")
                continue
        
        if not results:
            raise ValueError("No models could be loaded for comparison")
        
        return results
    
    def generate_comparison_report(self, results: Dict) -> Dict:
        """비교 보고서 생성"""
        
        # 성능 순위 (RMSE 기준)
        sorted_models = sorted(results.items(), key=lambda x: x[1]['rmse'])
        
        # 최고/최악 성능
        best_model = sorted_models[0]
        worst_model = sorted_models[-1]
        
        # 상대적 성능 계산
        baseline_rmse = worst_model[1]['rmse']  # 가장 안 좋은 모델을 베이스라인으로
        
        comparison_data = []
        for model_key, metrics in sorted_models:
            improvement = ((baseline_rmse - metrics['rmse']) / baseline_rmse) * 100
            
            comparison_data.append({
                'model': model_key,
                'rmse': metrics['rmse'],
                'mae': metrics['mae'],
                'r2': metrics['r2'],
                'rmse_improvement_pct': improvement,
                'max_residual': metrics['max_absolute_residual'],
                'residual_std': metrics['residual_std']
            })
        
        report = {
            'comparison_date': datetime.now().isoformat(),
            'test_samples': len(self.X_test),
            'best_model': {
                'name': best_model[0],
                'metrics': {k: v for k, v in best_model[1].items() if k not in ['predictions', 'residuals']}
            },
            'worst_model': {
                'name': worst_model[0],
                'metrics': {k: v for k, v in worst_model[1].items() if k not in ['predictions', 'residuals']}
            },
            'performance_gap': {
                'rmse_diff': worst_model[1]['rmse'] - best_model[1]['rmse'],
                'r2_diff': best_model[1]['r2'] - worst_model[1]['r2'],
                'improvement_pct': ((worst_model[1]['rmse'] - best_model[1]['rmse']) / worst_model[1]['rmse']) * 100
            },
            'detailed_comparison': comparison_data,
            'raw_results': {k: {key: val for key, val in v.items() if key not in ['predictions', 'residuals']} 
                           for k, v in results.items()}
        }
        
        return report
    
    def save_comparison_artifacts(self, results: Dict, report: Dict):
        """비교 결과를 MLflow 아티팩트로 저장"""
        
        # 새로운 실험에서 비교 결과 저장
        comparison_experiment = f"{self.config['project']['experiment_name']}_Model_Comparison"
        mlflow.set_experiment(comparison_experiment)
        
        with mlflow.start_run(run_name=f"Model_Comparison_{datetime.now().strftime('%Y%m%d_%H%M')}"):
            
            # 전체 비교 정보 로깅
            mlflow.log_param("models_compared", len(results))
            mlflow.log_param("test_samples", len(self.X_test))
            mlflow.log_param("comparison_date", report['comparison_date'])
            
            # 최고 성능 모델 정보
            best_model_info = report['best_model']
            mlflow.log_param("best_model", best_model_info['name'])
            mlflow.log_metric("best_rmse", best_model_info['metrics']['rmse'])
            mlflow.log_metric("best_r2", best_model_info['metrics']['r2'])
            
            # 성능 차이
            mlflow.log_metric("performance_gap_rmse", report['performance_gap']['rmse_diff'])
            mlflow.log_metric("performance_gap_r2", report['performance_gap']['r2_diff'])
            mlflow.log_metric("improvement_percentage", report['performance_gap']['improvement_pct'])
            
            # 1. 상세 비교 테이블 저장
            comparison_df = pd.DataFrame(report['detailed_comparison'])
            comparison_df.to_csv("model_comparison_table.csv", index=False)
            mlflow.log_artifact("model_comparison_table.csv")
            
            # 2. 예측 결과 비교 저장
            predictions_df = pd.DataFrame({'actual': self.y_test.values.ravel()})
            
            for model_key, metrics in results.items():
                predictions_df[f'{model_key}_pred'] = metrics['predictions']
                predictions_df[f'{model_key}_residual'] = metrics['residuals']
            
            predictions_df.to_csv("predictions_comparison.csv", index=False)
            mlflow.log_artifact("predictions_comparison.csv")
            
            # 3. 상세 보고서 저장
            with open("comparison_report.json", 'w') as f:
                json.dump(report, f, indent=2, default=str)
            mlflow.log_artifact("comparison_report.json")
            
            # 4. 시각화 생성 및 저장 (선택사항)
            self.create_comparison_plots(results)
            
            # 임시 파일 정리
            import os
            for file in ["model_comparison_table.csv", "predictions_comparison.csv", "comparison_report.json"]:
                if os.path.exists(file):
                    os.remove(file)
    
    def create_comparison_plots(self, results: Dict):
        """비교 시각화 생성"""
        try:
            plt.style.use('default')
            
            # 1. 성능 메트릭 비교 바 차트
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            models = list(results.keys())
            rmse_values = [results[m]['rmse'] for m in models]
            mae_values = [results[m]['mae'] for m in models]
            r2_values = [results[m]['r2'] for m in models]
            
            # RMSE 비교
            axes[0].bar(models, rmse_values, color='lightcoral')
            axes[0].set_title('RMSE Comparison')
            axes[0].set_ylabel('RMSE')
            axes[0].tick_params(axis='x', rotation=45)
            
            # MAE 비교
            axes[1].bar(models, mae_values, color='lightblue')
            axes[1].set_title('MAE Comparison')
            axes[1].set_ylabel('MAE')
            axes[1].tick_params(axis='x', rotation=45)
            
            # R² 비교
            axes[2].bar(models, r2_values, color='lightgreen')
            axes[2].set_title('R² Comparison')
            axes[2].set_ylabel('R²')
            axes[2].tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            plt.savefig('metrics_comparison.png', dpi=300, bbox_inches='tight')
            mlflow.log_artifact('metrics_comparison.png')
            plt.close()
            
            # 2. 잔차 분포 비교
            fig, axes = plt.subplots(1, len(results), figsize=(5*len(results), 4))
            if len(results) == 1:
                axes = [axes]
            
            for i, (model_key, metrics) in enumerate(results.items()):
                axes[i].hist(metrics['residuals'], bins=30, alpha=0.7, color=f'C{i}')
                axes[i].set_title(f'{model_key}\nResiduals Distribution')
                axes[i].set_xlabel('Residuals')
                axes[i].set_ylabel('Frequency')
                axes[i].axvline(x=0, color='red', linestyle='--', alpha=0.7)
            
            plt.tight_layout()
            plt.savefig('residuals_comparison.png', dpi=300, bbox_inches='tight')
            mlflow.log_artifact('residuals_comparison.png')
            plt.close()
            
            # 파일 정리
            for file in ['metrics_comparison.png', 'residuals_comparison.png']:
                if os.path.exists(file):
                    os.remove(file)
                    
        except Exception as e:
            logger.warning(f"Could not create plots: {str(e)}")
    
    def print_comparison_summary(self, report: Dict):
        """비교 결과 요약 출력"""
        
        print("\n" + "="*80)
        print("MODEL COMPARISON SUMMARY")
        print("="*80)
        
        print(f"Comparison Date: {report['comparison_date']}")
        print(f"Test Samples: {report['test_samples']}")
        print(f"Models Compared: {len(report['detailed_comparison'])}")
        
        print("\n" + "-"*80)
        print("PERFORMANCE RANKING (Best to Worst by RMSE):")
        print("-"*80)
        
        for i, model_data in enumerate(report['detailed_comparison'], 1):
            print(f"{i}. {model_data['model']}")
            print(f"   RMSE: {model_data['rmse']:.4f}")
            print(f"   MAE:  {model_data['mae']:.4f}")
            print(f"   R²:   {model_data['r2']:.4f}")
            print(f"   Improvement: {model_data['rmse_improvement_pct']:+.2f}%")
            print()
        
        print("-"*80)
        print("BEST vs WORST MODEL:")
        print("-"*80)
        
        best = report['best_model']
        worst = report['worst_model']
        gap = report['performance_gap']
        
        print(f"🏆 BEST:  {best['name']}")
        print(f"   RMSE: {best['metrics']['rmse']:.4f}")
        print(f"   R²:   {best['metrics']['r2']:.4f}")
        
        print(f"\n📉 WORST: {worst['name']}")
        print(f"   RMSE: {worst['metrics']['rmse']:.4f}")
        print(f"   R²:   {worst['metrics']['r2']:.4f}")
        
        print(f"\n💡 PERFORMANCE GAP:")
        print(f"   RMSE Difference: {gap['rmse_diff']:.4f}")
        print(f"   R² Difference: {gap['r2_diff']:.4f}")
        print(f"   Overall Improvement: {gap['improvement_pct']:.2f}%")

def list_available_models(config=None):
    """MLflow에 등록된 모델 목록 확인"""
    if config is None:
        config = CONFIG
    
    mlflow.set_tracking_uri(config['mlflow']['tracking_uri'])
    client = mlflow.tracking.MlflowClient()
    
    print("\n" + "="*60)
    print("AVAILABLE MODELS IN MLFLOW")
    print("="*60)
    
    try:
        models = client.search_registered_models()
        
        if not models:
            print("No registered models found.")
            return []
        
        model_info = []
        for model in models:
            versions = client.get_latest_versions(model.name)
            latest_version = max(versions, key=lambda x: int(x.version)) if versions else None
            
            model_data = {
                'name': model.name,
                'latest_version': latest_version.version if latest_version else 'N/A',
                'creation_time': latest_version.creation_timestamp if latest_version else 'N/A'
            }
            model_info.append(model_data)
            
            print(f"📦 {model.name}")
            print(f"   Latest Version: {model_data['latest_version']}")
            if latest_version:
                print(f"   Created: {datetime.fromtimestamp(int(latest_version.creation_timestamp)/1000).strftime('%Y-%m-%d %H:%M')}")
            print()
        
        return model_info
        
    except Exception as e:
        print(f"Error listing models: {str(e)}")
        return []

def main():
    """메인 실행 함수"""
    try:
        # 사용 가능한 모델 목록 먼저 확인
        print("Checking available models...")
        available_models = list_available_models()
        
        if len(available_models) < 2:
            print("⚠️  Warning: Found less than 2 models. Make sure you have both RandomForest and XGBoost models trained.")
            print("   Run train.py and train2.py first to create models for comparison.")
            
            if available_models:
                print(f"\n🔍 Found {len(available_models)} model(s). Proceeding with available model(s)...")
            else:
                print("\n❌ No models found. Please train some models first.")
                return
        
        # 모델 비교기 초기화
        comparator = ModelComparator()
        
        # 비교할 모델 설정 - 실제 존재하는 모델만
        model_configs = []
        
        # 기본 모델 확인
        base_model_name = CONFIG['project']['model_name']
        xgb_model_name = f"{CONFIG['project']['model_name']}_XGBoost"
        
        for model_info in available_models:
            model_name = model_info['name']
            
            if model_name == base_model_name:
                model_configs.append({
                    'name': model_name,
                    'version': None,
                    'stage': None
                })
                print(f"✅ Found base model: {model_name}")
            
            elif model_name == xgb_model_name:
                model_configs.append({
                    'name': model_name,
                    'version': None,
                    'stage': None
                })
                print(f"✅ Found XGBoost model: {model_name}")
        
        if not model_configs:
            print("❌ No matching models found for comparison.")
            print(f"   Expected: {base_model_name} and/or {xgb_model_name}")
            return
        
        print(f"\n🚀 Starting comparison of {len(model_configs)} model(s)...")
        
        # 모델 비교 실행
        results = comparator.compare_models(model_configs)
        
        # 비교 보고서 생성
        report = comparator.generate_comparison_report(results)
        
        # 결과 출력
        comparator.print_comparison_summary(report)
        
        # MLflow에 결과 저장
        comparator.save_comparison_artifacts(results, report)
        
        # 결과를 로컬 파일로도 저장
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        local_report_file = f"model_comparison_report_{timestamp}.json"
        
        with open(local_report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"\n📊 Detailed report saved to: {local_report_file}")
        print(f"🔗 Check MLflow UI for interactive comparison:")
        print(f"   http://localhost:{CONFIG['mlflow']['server_port']}")
        
        # 추천 사항 출력
        if len(results) > 1:
            print_recommendations(report)
        else:
            print(f"\n💡 TIP: Train more models (run train2.py) to enable comparative analysis!")
        
    except Exception as e:
        logger.error(f"Model comparison failed: {str(e)}")
        raise

def print_recommendations(report: Dict):
    """모델 선택 추천 사항 출력"""
    
    print("\n" + "="*80)
    print("RECOMMENDATIONS")
    print("="*80)
    
    best_model = report['best_model']
    performance_gap = report['performance_gap']
    
    # 성능 기반 추천
    if performance_gap['improvement_pct'] > 5:
        print(f"✅ STRONG RECOMMENDATION: Use {best_model['name']}")
        print(f"   Significant improvement of {performance_gap['improvement_pct']:.1f}% over alternatives")
    elif performance_gap['improvement_pct'] > 1:
        print(f"✅ RECOMMENDATION: Consider {best_model['name']}")
        print(f"   Moderate improvement of {performance_gap['improvement_pct']:.1f}%")
    else:
        print(f"⚠️  MARGINAL DIFFERENCE: Performance gap is only {performance_gap['improvement_pct']:.1f}%")
        print("   Consider other factors like training time, interpretability, etc.")
    
    # R² 기반 추천
    best_r2 = best_model['metrics']['r2']
    if best_r2 > 0.9:
        print(f"🎯 EXCELLENT model performance (R² = {best_r2:.3f})")
    elif best_r2 > 0.8:
        print(f"✅ GOOD model performance (R² = {best_r2:.3f})")
    elif best_r2 > 0.7:
        print(f"⚠️  ACCEPTABLE model performance (R² = {best_r2:.3f})")
    else:
        print(f"❌ POOR model performance (R² = {best_r2:.3f}) - Consider feature engineering")
    
    # 모델별 특성 고려사항
    print(f"\n📋 MODEL CHARACTERISTICS:")
    
    for model_data in report['detailed_comparison']:
        model_name = model_data['model']
        
        if 'RandomForest' in model_name or 'random_forest' in model_name.lower():
            print(f"   🌲 {model_name}:")
            print(f"      - Good interpretability with feature importance")
            print(f"      - Robust to outliers")
            print(f"      - Less prone to overfitting")
            
        elif 'XGBoost' in model_name or 'xgboost' in model_name.lower():
            print(f"   🚀 {model_name}:")
            print(f"      - Often superior predictive performance")
            print(f"      - Built-in regularization")
            print(f"      - Requires more hyperparameter tuning")
            
        elif 'Ridge' in model_name or 'ridge' in model_name.lower():
            print(f"   📐 {model_name}:")
            print(f"      - Fastest training and prediction")
            print(f"      - Good baseline model")
            print(f"      - Assumes linear relationships")
    
    print(f"\n🔄 NEXT STEPS:")
    print(f"   1. If satisfied with {best_model['name']}, promote to Production stage")
    print(f"   2. Consider ensemble methods combining top performers")
    print(f"   3. Investigate hyperparameter tuning for further improvements")
    print(f"   4. Monitor model performance on new data")

def compare_specific_models(model_names: List[str], versions: List[str] = None):
    """특정 모델들만 비교하는 유틸리티 함수"""
    
    comparator = ModelComparator()
    
    model_configs = []
    for i, name in enumerate(model_names):
        version = versions[i] if versions and i < len(versions) else None
        model_configs.append({
            'name': name,
            'version': version,
            'stage': None
        })
    
    results = comparator.compare_models(model_configs)
    report = comparator.generate_comparison_report(results)
    comparator.print_comparison_summary(report)
    
    return results, report

if __name__ == "__main__":
    main()