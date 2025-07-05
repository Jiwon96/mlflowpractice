#!/usr/bin/env python3
"""
통합 테스트 스크립트
전체 파이프라인을 작은 규모로 테스트
"""

import os
import sys
import mlflow
from datetime import datetime

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def run_mini_pipeline():
    """미니 파이프라인 실행 (빠른 통합 테스트)"""
    print("MNIST MLOps 미니 파이프라인 테스트")
    print("=" * 50)
    
    # MLflow 실험 설정
    experiment_name = f"mnist_integration_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    mlflow.set_experiment(experiment_name)
    
    try:
        # 1. 데이터 파이프라인
        print("1️⃣ 데이터 파이프라인 실행...")
        from src.data.download import MNISTDataLoader
        from src.data.preprocess import DataPreprocessor
        from src.data.validation import DataValidator
        
        data_loader = MNISTDataLoader(data_dir="./test_data")
        ds_train, ds_test, ds_info = data_loader.download_data()
        
        preprocessor = DataPreprocessor(batch_size=32, validation_split=0.1)
        train_ds, val_ds, test_ds = preprocessor.create_data_pipeline(ds_train, ds_test, ds_info)
        
        validator = DataValidator()
        train_validation = validator.validate_data_quality(train_ds, "train")
        
        print("✅ 데이터 파이프라인 완료")
        
        # 2. 모델 파이프라인
        print("2️⃣ 모델 파이프라인 실행...")
        from src.model.cnn_model import MNISTCNNModel
        from src.model.train import ModelTrainer
        
        model = MNISTCNNModel()
        
        # 빠른 테스트 설정
        config = {
            'learning_rate': 0.001,
            'epochs': 3  # 통합 테스트용
        }
        
        trainer = ModelTrainer(model, train_ds, val_ds, config)
        history, best_accuracy = trainer.train(epochs=3)
        
        print("✅ 모델 훈련 완료")
        
        # 3. 평가 파이프라인
        print("3️⃣ 평가 파이프라인 실행...")
        from src.model.evaluate import ModelEvaluator
        
        evaluator = ModelEvaluator(model, test_ds)
        evaluation_results = evaluator.evaluate_model()
        
        print("✅ 모델 평가 완료")
        
        # 4. 예측 파이프라인
        print("4️⃣ 예측 파이프라인 실행...")
        from src.model.predict import ModelPredictor
        
        predictor = ModelPredictor()
        predictor.model = model
        
        # 테스트 이미지로 예측
        for images, labels in test_ds.take(1):
            result = predictor.predict_single(images[0].numpy())
            print(f"✅ 예측 결과: 클래스 {result['predicted_class']}, 신뢰도: {result['confidence']:.4f}")
            break
        
        print("\n🎉 통합 테스트 성공!")
        print(f"최종 정확도: {evaluation_results['test_accuracy']:.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ 통합 테스트 실패: {e}")
        return False

if __name__ == "__main__":
    success = run_mini_pipeline()
    sys.exit(0 if success else 1)