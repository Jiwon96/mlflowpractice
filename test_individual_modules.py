#!/usr/bin/env python3
"""
개별 모듈 테스트 스크립트
각 모듈을 독립적으로 테스트하여 문제점 조기 발견
"""

import os
import sys
import tensorflow as tf
import mlflow

# 프로젝트 경로 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_data_download():
    """데이터 다운로드 모듈 테스트"""
    print("=" * 50)
    print("데이터 다운로드 모듈 테스트")
    print("=" * 50)
    
    try:
        from src.data.download import MNISTDataLoader
        
        # 테스트용 작은 데이터 디렉토리
        data_loader = MNISTDataLoader(data_dir="./test_data")
        ds_train, ds_test, ds_info = data_loader.download_data()
        
        print(f"✅ 훈련 데이터 크기: {ds_info.splits['train'].num_examples}")
        print(f"✅ 테스트 데이터 크기: {ds_info.splits['test'].num_examples}")
        print("✅ 데이터 다운로드 모듈 테스트 성공!")
        
        return ds_train, ds_test, ds_info
        
    except Exception as e:
        print(f"❌ 데이터 다운로드 모듈 테스트 실패: {e}")
        return None, None, None

def test_data_preprocessing(ds_train, ds_test, ds_info):
    """데이터 전처리 모듈 테스트"""
    print("\n" + "=" * 50)
    print("데이터 전처리 모듈 테스트")
    print("=" * 50)
    
    try:
        from src.data.preprocess import DataPreprocessor
        
        # 작은 배치 크기로 빠른 테스트
        preprocessor = DataPreprocessor(batch_size=16, validation_split=0.1)
        train_ds, val_ds, test_ds = preprocessor.create_data_pipeline(ds_train, ds_test, ds_info)
        
        # 첫 번째 배치 확인
        for images, labels in train_ds.take(1):
            print(f"✅ 배치 이미지 모양: {images.shape}")
            print(f"✅ 배치 레이블 모양: {labels.shape}")
            print(f"✅ 픽셀 값 범위: [{tf.reduce_min(images):.3f}, {tf.reduce_max(images):.3f}]")
            break
        
        print("✅ 데이터 전처리 모듈 테스트 성공!")
        return train_ds, val_ds, test_ds
        
    except Exception as e:
        print(f"❌ 데이터 전처리 모듈 테스트 실패: {e}")
        return None, None, None

def test_model_creation():
    """모델 생성 모듈 테스트"""
    print("\n" + "=" * 50)
    print("모델 생성 모듈 테스트")
    print("=" * 50)
    
    try:
        from src.model.cnn_model import MNISTCNNModel
        
        model = MNISTCNNModel()
        
        # 더미 입력으로 모델 테스트
        dummy_input = tf.random.normal([2, 28, 28, 1])  # 배치 크기 2
        output = model(dummy_input)
        
        print(f"✅ 모델 입력 모양: {dummy_input.shape}")
        print(f"✅ 모델 출력 모양: {output.shape}")
        
        # 모델 정보 출력
        model_summary = model.get_model_summary()
        print(f"✅ 총 파라미터 수: {model_summary['total_parameters']:,}")
        print(f"✅ 모델 크기: {model_summary['model_size_mb']:.2f} MB")
        
        print("✅ 모델 생성 모듈 테스트 성공!")
        return model
        
    except Exception as e:
        print(f"❌ 모델 생성 모듈 테스트 실패: {e}")
        return None

def test_quick_training(model, train_ds, val_ds):
    """빠른 훈련 테스트 (1-2 에포크)"""
    print("\n" + "=" * 50)
    print("빠른 훈련 모듈 테스트")
    print("=" * 50)
    
    try:
        from src.model.train import ModelTrainer
        
        # 빠른 테스트를 위한 설정
        config = {
            'learning_rate': 0.001,
            'epochs': 2  # 빠른 테스트용
        }
        
        trainer = ModelTrainer(model, train_ds, val_ds, config)
        
        # 모델 컴파일만 테스트
        trainer.compile_model()
        print("✅ 모델 컴파일 성공!")
        
        # 1 에포크만 실행해서 훈련 로직 검증
        print("1 에포크 훈련 테스트 시작...")
        history, best_accuracy = trainer.train(epochs=1)
        
        print(f"✅ 1 에포크 훈련 완료! 정확도: {best_accuracy:.4f}")
        print("✅ 훈련 모듈 테스트 성공!")
        
        return model, history
        
    except Exception as e:
        print(f"❌ 훈련 모듈 테스트 실패: {e}")
        return None, None

def run_module_tests():
    """모든 모듈 테스트 실행"""
    print("MNIST MLOps 개별 모듈 테스트 시작")
    print("=" * 70)
    
    # MLflow 실험 설정
    mlflow.set_experiment("mnist_module_tests")
    
    # 1. 데이터 다운로드 테스트
    ds_train, ds_test, ds_info = test_data_download()
    if ds_train is None:
        return False
    
    # 2. 데이터 전처리 테스트
    train_ds, val_ds, test_ds = test_data_preprocessing(ds_train, ds_test, ds_info)
    if train_ds is None:
        return False
    
    # 3. 모델 생성 테스트
    model = test_model_creation()
    if model is None:
        return False
    
    # 4. 빠른 훈련 테스트
    trained_model, history = test_quick_training(model, train_ds, val_ds)
    if trained_model is None:
        return False
    
    print("\n" + "=" * 70)
    print("🎉 모든 개별 모듈 테스트 성공!")
    print("=" * 70)
    
    return True

if __name__ == "__main__":
    success = run_module_tests()
    sys.exit(0 if success else 1)