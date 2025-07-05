import tensorflow as tf
import numpy as np
import mlflow
import mlflow.tensorflow
from datetime import datetime
import json
import os

class ModelPredictor:
    def __init__(self, model_path=None, model_name="mnist_cnn_model", model_version="latest"):
        self.model_path = model_path
        self.model_name = model_name
        self.model_version = model_version
        self.model = None
        self.tflite_interpreter = None
        
    def load_keras_model(self):
        """Keras 모델 로드"""
        if self.model_path:
            self.model = tf.keras.models.load_model(self.model_path)
        else:
            # MLflow에서 모델 로드
            model_uri = f"models:/{self.model_name}/{self.model_version}"
            self.model = mlflow.tensorflow.load_model(model_uri)
        
        print(f"Keras 모델 로드 완료: {self.model_name}")
        return self.model
    
    def load_tflite_model(self, tflite_path):
        """TensorFlow Lite 모델 로드"""
        self.tflite_interpreter = tf.lite.Interpreter(model_path=tflite_path)
        self.tflite_interpreter.allocate_tensors()
        
        self.input_details = self.tflite_interpreter.get_input_details()
        self.output_details = self.tflite_interpreter.get_output_details()
        
        print(f"TFLite 모델 로드 완료: {tflite_path}")
        return self.tflite_interpreter
    
    def predict_single(self, image, use_tflite=False):
        """단일 이미지 예측"""
        
        # 이미지 전처리
        if len(image.shape) == 2:  # (28, 28) -> (1, 28, 28, 1)
            image = np.expand_dims(image, axis=[0, -1])
        elif len(image.shape) == 3:  # (28, 28, 1) -> (1, 28, 28, 1)
            image = np.expand_dims(image, axis=0)
        
        # 정규화 (0-255 -> 0-1)
        if image.max() > 1:
            image = image.astype(np.float32) / 255.0
        
        if use_tflite and self.tflite_interpreter:
            return self._predict_tflite(image)
        else:
            return self._predict_keras(image)
    
    def _predict_keras(self, image):
        """Keras 모델로 예측"""
        if self.model is None:
            self.load_keras_model()
        
        start_time = datetime.now()
        prediction = self.model.predict(image, verbose=0)
        inference_time = (datetime.now() - start_time).total_seconds()
        
        predicted_class = np.argmax(prediction[0])
        confidence = float(np.max(prediction[0]))
        
        return {
            "predicted_class": int(predicted_class),
            "confidence": confidence,
            "probabilities": prediction[0].tolist(),
            "inference_time_ms": inference_time * 1000,
            "model_type": "keras"
        }
    
    def _predict_tflite(self, image):
        """TensorFlow Lite 모델로 예측"""
        if self.tflite_interpreter is None:
            raise ValueError("TFLite 모델이 로드되지 않았습니다.")
        
        start_time = datetime.now()
        
        # 입력 설정
        self.tflite_interpreter.set_tensor(
            self.input_details[0]['index'], 
            image.astype(np.float32)
        )
        
        # 추론 실행
        self.tflite_interpreter.invoke()
        
        # 결과 가져오기
        prediction = self.tflite_interpreter.get_tensor(self.output_details[0]['index'])
        inference_time = (datetime.now() - start_time).total_seconds()
        
        predicted_class = np.argmax(prediction[0])
        confidence = float(np.max(prediction[0]))
        
        return {
            "predicted_class": int(predicted_class),
            "confidence": confidence,
            "probabilities": prediction[0].tolist(),
            "inference_time_ms": inference_time * 1000,
            "model_type": "tflite"
        }
    
    def predict_batch(self, images, use_tflite=False, batch_size=32):
        """배치 예측"""
        results = []
        
        if use_tflite:
            # TFLite는 배치 처리 불가, 개별 처리
            for image in images:
                result = self.predict_single(image, use_tflite=True)
                results.append(result)
        else:
            # Keras 모델 배치 처리
            if self.model is None:
                self.load_keras_model()
            
            # 이미지 전처리
            processed_images = self._preprocess_batch(images)
            
            start_time = datetime.now()
            predictions = self.model.predict(processed_images, batch_size=batch_size, verbose=0)
            total_inference_time = (datetime.now() - start_time).total_seconds()
            
            for i, prediction in enumerate(predictions):
                predicted_class = np.argmax(prediction)
                confidence = float(np.max(prediction))
                
                results.append({
                    "predicted_class": int(predicted_class),
                    "confidence": confidence,
                    "probabilities": prediction.tolist(),
                    "inference_time_ms": (total_inference_time / len(predictions)) * 1000,
                    "model_type": "keras"
                })
        
        return results
    
    def _preprocess_batch(self, images):
        """배치 이미지 전처리"""
        processed = []
        
        for image in images:
            if len(image.shape) == 2:  # (28, 28) -> (28, 28, 1)
                image = np.expand_dims(image, axis=-1)
            
            # 정규화
            if image.max() > 1:
                image = image.astype(np.float32) / 255.0
            
            processed.append(image)
        
        return np.array(processed)
    
    def benchmark_performance(self, test_images, num_iterations=100):
        """모델 성능 벤치마크"""
        
        keras_times = []
        tflite_times = []
        
        # 샘플 이미지 선택
        sample_images = test_images[:min(num_iterations, len(test_images))]
        
        print(f"성능 벤치마크 시작: {len(sample_images)}개 이미지")
        
        # Keras 모델 벤치마크
        if self.model is None:
            self.load_keras_model()
        
        for image in sample_images:
            result = self.predict_single(image, use_tflite=False)
            keras_times.append(result['inference_time_ms'])
        
        # TFLite 모델 벤치마크 (있는 경우)
        if self.tflite_interpreter:
            for image in sample_images:
                result = self.predict_single(image, use_tflite=True)
                tflite_times.append(result['inference_time_ms'])
        
        benchmark_results = {
            "keras_model": {
                "mean_inference_time_ms": np.mean(keras_times),
                "std_inference_time_ms": np.std(keras_times),
                "min_inference_time_ms": np.min(keras_times),
                "max_inference_time_ms": np.max(keras_times)
            }
        }
        
        if tflite_times:
            benchmark_results["tflite_model"] = {
                "mean_inference_time_ms": np.mean(tflite_times),
                "std_inference_time_ms": np.std(tflite_times),
                "min_inference_time_ms": np.min(tflite_times),
                "max_inference_time_ms": np.max(tflite_times)
            }
            
            benchmark_results["speedup_ratio"] = np.mean(keras_times) / np.mean(tflite_times)
        
        # 결과 로깅
        with mlflow.start_run(run_name="performance_benchmark"):
            for model_type, metrics in benchmark_results.items():
                if isinstance(metrics, dict):
                    for metric_name, value in metrics.items():
                        mlflow.log_metric(f"{model_type}_{metric_name}", value)
                else:
                    mlflow.log_metric(model_type, metrics)
        
        return benchmark_results
    
    def explain_prediction(self, image, top_k=3):
        """예측 결과 설명"""
        result = self.predict_single(image)
        
        # 상위 k개 예측 결과
        probabilities = np.array(result['probabilities'])
        top_indices = np.argsort(probabilities)[::-1][:top_k]
        
        explanations = []
        for idx in top_indices:
            explanations.append({
                "class": int(idx),
                "probability": float(probabilities[idx]),
                "percentage": float(probabilities[idx] * 100)
            })
        
        return {
            "predicted_class": result['predicted_class'],
            "confidence": result['confidence'],
            "top_predictions": explanations,
            "inference_time_ms": result['inference_time_ms']
        }

class OnDeviceModelManager:
    """온디바이스 모델 관리 클래스"""
    
    def __init__(self, model_registry_path="./data/models"):
        self.model_registry_path = model_registry_path
        self.model_metadata = {}
        self.load_model_registry()
    
    def load_model_registry(self):
        """모델 레지스트리 로드"""
        registry_file = os.path.join(self.model_registry_path, "model_registry.json")
        
        if os.path.exists(registry_file):
            with open(registry_file, 'r') as f:
                self.model_metadata = json.load(f)
        else:
            self.model_metadata = {"models": []}
    
    def register_model(self, model_info):
        """새 모델 등록"""
        model_info['registered_at'] = datetime.now().isoformat()
        model_info['model_id'] = f"mnist_cnn_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.model_metadata["models"].append(model_info)
        self._save_registry()
        
        return model_info['model_id']
    
    def get_best_model(self, metric="accuracy"):
        """최고 성능 모델 반환"""
        if not self.model_metadata["models"]:
            return None
        
        best_model = max(
            self.model_metadata["models"],
            key=lambda x: x.get(metric, 0)
        )
        
        return best_model
    
    def get_lightweight_model(self):
        """가장 경량화된 모델 반환"""
        if not self.model_metadata["models"]:
            return None
        
        lightweight_model = min(
            self.model_metadata["models"],
            key=lambda x: x.get("model_size_mb", float('inf'))
        )
        
        return lightweight_model
    
    def _save_registry(self):
        """모델 레지스트리 저장"""
        registry_file = os.path.join(self.model_registry_path, "model_registry.json")
        
        with open(registry_file, 'w') as f:
            json.dump(self.model_metadata, f, indent=2)
    
    def cleanup_old_models(self, keep_count=5):
        """오래된 모델 정리"""
        if len(self.model_metadata["models"]) <= keep_count:
            return
        
        # 등록일 기준 정렬
        sorted_models = sorted(
            self.model_metadata["models"],
            key=lambda x: x["registered_at"],
            reverse=True
        )
        
        # 유지할 모델들
        self.model_metadata["models"] = sorted_models[:keep_count]
        self._save_registry()
        
        print(f"모델 정리 완료: {keep_count}개 모델 유지")