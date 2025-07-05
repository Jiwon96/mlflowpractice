from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.operators.bash_operator import BashOperator
from airflow.utils.dates import days_ago
from datetime import datetime, timedelta
import sys
import os

# 프로젝트 경로 추가
sys.path.append('/opt/airflow/dags/mnist-mlops')

from src.data.download import MNISTDataLoader
from src.data.preprocess import DataPreprocessor
from src.data.validation import DataValidator
from src.model.cnn_model import MNISTCNNModel
from src.model.train import ModelTrainer
from src.model.evaluate import ModelEvaluator
from src.model.predict import OnDeviceModelManager
from config.config import load_config

# 기본 DAG 설정
default_args = {
    'owner': 'mlops-team',
    'depends_on_past': False,
    'start_date': days_ago(1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# DAG 정의
dag = DAG(
    'mnist_mlops_pipeline',
    default_args=default_args,
    description='MNIST MLOps 전체 파이프라인',
    schedule_interval='0 2 * * *',  # 매일 새벽 2시
    catchup=False,
    tags=['mlops', 'mnist', 'computer-vision'],
)

def download_and_validate_data(**context):
    """데이터 다운로드 및 검증"""
    print("데이터 다운로드 시작...")
    
    # 데이터 로더 초기화
    data_loader = MNISTDataLoader()
    
    # 데이터 다운로드
    ds_train, ds_test, ds_info = data_loader.download_data()
    
    # 데이터 검증
    validator = DataValidator()
    
    # 훈련 데이터 검증
    train_validation = validator.validate_data_quality(ds_train, "train")
    test_validation = validator.validate_data_quality(ds_test, "test")
    
    # 검증 결과 확인
    if not train_validation["is_normalized"]:
        raise ValueError("훈련 데이터가 정규화되지 않았습니다.")
    
    if not test_validation["is_normalized"]:
        raise ValueError("테스트 데이터가 정규화되지 않았습니다.")
    
    print("데이터 다운로드 및 검증 완료")
    
    # XCom으로 다음 태스크에 정보 전달
    return {
        "train_size": train_validation["total_samples"],
        "test_size": test_validation["total_samples"],
        "data_quality_passed": True
    }

def preprocess_data(**context):
    """데이터 전처리"""
    print("데이터 전처리 시작...")
    
    # 이전 태스크 결과 확인
    data_info = context['task_instance'].xcom_pull(task_ids='download_and_validate_data')
    
    if not data_info["data_quality_passed"]:
        raise ValueError("데이터 품질 검증 실패")
    
    # 데이터 로더 및 전처리기 초기화
    data_loader = MNISTDataLoader()
    preprocessor = DataPreprocessor(batch_size=32)
    
    # 데이터 로드
    ds_train, ds_test, ds_info = data_loader.download_data()
    
    # 데이터 파이프라인 생성
    train_ds, val_ds, test_ds = preprocessor.create_data_pipeline(ds_train, ds_test, ds_info)
    
    # 데이터 누출 검사
    validator = DataValidator()
    leakage_check = validator.check_data_leakage(train_ds, val_ds)
    
    if leakage_check["is_leakage_detected"]:
        raise ValueError("데이터 누출이 감지되었습니다!")
    
    print("데이터 전처리 완료")
    
    return {
        "preprocessing_completed": True,
        "train_batches": len(list(train_ds)),
        "val_batches": len(list(val_ds)),
        "test_batches": len(list(test_ds))
    }

def train_model(**context):
    """모델 훈련"""
    print("모델 훈련 시작...")
    
    # 이전 태스크 결과 확인
    preprocess_info = context['task_instance'].xcom_pull(task_ids='preprocess_data')
    
    if not preprocess_info["preprocessing_completed"]:
        raise ValueError("데이터 전처리 실패")
    
    # 설정 로드
    config = load_config()
    
    # 데이터 준비
    data_loader = MNISTDataLoader()
    preprocessor = DataPreprocessor()
    ds_train, ds_test, ds_info = data_loader.download_data()
    train_ds, val_ds, test_ds = preprocessor.create_data_pipeline(ds_train, ds_test, ds_info)
    
    # 모델 생성
    model = MNISTCNNModel()
    
    # 더미 입력으로 모델 빌드
    dummy_input = tf.random.normal([1, 28, 28, 1])
    _ = model(dummy_input)
    
    # 훈련기 초기화
    trainer = ModelTrainer(model, train_ds, val_ds, config['model'])
    
    # 훈련 실행
    history, best_accuracy = trainer.train(epochs=config['model'].get('epochs', 10))
    
    print(f"모델 훈련 완료. 최고 정확도: {best_accuracy:.4f}")
    
    return {
        "training_completed": True,
        "best_accuracy": best_accuracy,
        "total_epochs": len(history.history['loss'])
    }

def evaluate_model(**context):
    """모델 평가"""
    print("모델 평가 시작...")
    
    # 이전 태스크 결과 확인
    train_info = context['task_instance'].xcom_pull(task_ids='train_model')
    
    if not train_info["training_completed"]:
        raise ValueError("모델 훈련 실패")
    
    # 최신 모델 로드
    model_manager = OnDeviceModelManager()
    best_model_info = model_manager.get_best_model()
    
    if not best_model_info:
        raise ValueError("평가할 모델이 없습니다.")
    
    # 데이터 준비
    data_loader = MNISTDataLoader()
    preprocessor = DataPreprocessor()
    ds_train, ds_test, ds_info = data_loader.download_data()
    _, _, test_ds = preprocessor.create_data_pipeline(ds_train, ds_test, ds_info)
    
    # 모델 로드
    model = tf.keras.models.load_model(best_model_info['keras_model_path'])
    
    # 평가 실행
    evaluator = ModelEvaluator(model, test_ds)
    evaluation_results = evaluator.evaluate_model()
    
    # TFLite 모델 평가 (있는 경우)
    if 'tflite_model_path' in best_model_info:
        tflite_accuracy = evaluator.evaluate_tflite_model(best_model_info['tflite_model_path'])
        evaluation_results['tflite_accuracy'] = tflite_accuracy
    
    print(f"모델 평가 완료. 테스트 정확도: {evaluation_results['test_accuracy']:.4f}")
    
    return {
        "evaluation_completed": True,
        "test_accuracy": evaluation_results['test_accuracy'],
        "test_loss": evaluation_results['test_loss']
    }

def deploy_model(**context):
    """모델 배포 준비"""
    print("모델 배포 준비 시작...")
    
    # 이전 태스크 결과 확인
    eval_info = context['task_instance'].xcom_pull(task_ids='evaluate_model')
    
    if not eval_info["evaluation_completed"]:
        raise ValueError("모델 평가 실패")
    
    # 정확도 임계값 확인
    min_accuracy = 0.95
    if eval_info["test_accuracy"] < min_accuracy:
        raise ValueError(f"모델 정확도({eval_info['test_accuracy']:.4f})가 임계값({min_accuracy}) 미달")
    
    # 모델 관리자 초기화
    model_manager = OnDeviceModelManager()
    
    # 모델 메타데이터 등록
    model_info = {
        "accuracy": eval_info["test_accuracy"],
        "loss": eval_info["test_loss"],
        "deployment_ready": True,
        "model_version": datetime.now().strftime("%Y%m%d_%H%M%S")
    }
    
    model_id = model_manager.register_model(model_info)
    
    # 오래된 모델 정리
    model_manager.cleanup_old_models(keep_count=5)
    
    print(f"모델 배포 준비 완료. 모델 ID: {model_id}")
    
    return {
        "deployment_ready": True,
        "model_id": model_id,
        "model_accuracy": eval_info["test_accuracy"]
    }

# 태스크 정의
download_task = PythonOperator(
    task_id='download_and_validate_data',
    python_callable=download_and_validate_data,
    dag=dag,
)

preprocess_task = PythonOperator(
    task_id='preprocess_data',
    python_callable=preprocess_data,
    dag=dag,
)

train_task = PythonOperator(
    task_id='train_model',
    python_callable=train_model,
    dag=dag,
)

evaluate_task = PythonOperator(
    task_id='evaluate_model',
    python_callable=evaluate_model,
    dag=dag,
)

deploy_task = PythonOperator(
    task_id='deploy_model',
    python_callable=deploy_model,
    dag=dag,
)

# 모델 성능 모니터링 태스크
monitor_task = BashOperator(
    task_id='monitor_model_performance',
    bash_command='echo "모델 성능 모니터링 시작..." && python /opt/airflow/dags/mnist-mlops/src/utils/monitor.py',
    dag=dag,
)

# 태스크 의존성 설정
download_task >> preprocess_task >> train_task >> evaluate_task >> deploy_task >> monitor_task