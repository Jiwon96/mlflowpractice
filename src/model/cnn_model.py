import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import mlflow

class MNISTCNNModel(keras.Model):
    def __init__(self, num_classes=10, name="mnist_cnn"):
        super(MNISTCNNModel, self).__init__(name=name)
        
        # 온디바이스 모델을 위한 경량화된 CNN 아키텍처
        self.conv1 = layers.Conv2D(16, 3, activation='relu', padding='same')
        self.pool1 = layers.MaxPooling2D(2)
        self.conv2 = layers.Conv2D(32, 3, activation='relu', padding='same')
        self.pool2 = layers.MaxPooling2D(2)
        
        # 배치 정규화와 드롭아웃으로 안정성 향상
        self.bn1 = layers.BatchNormalization()
        self.dropout1 = layers.Dropout(0.25)
        
        # 전역 평균 풀링으로 파라미터 수 감소
        self.global_pool = layers.GlobalAveragePooling2D()
        
        # 완전연결층
        self.dense1 = layers.Dense(64, activation='relu')
        self.bn2 = layers.BatchNormalization()
        self.dropout2 = layers.Dropout(0.5)
        self.dense2 = layers.Dense(num_classes, activation='softmax')
        
    def call(self, inputs, training=None):
        x = self.conv1(inputs)
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = self.pool2(x)
        
        x = self.bn1(x, training=training)
        x = self.dropout1(x, training=training)
        
        x = self.global_pool(x)
        
        x = self.dense1(x)
        x = self.bn2(x, training=training)
        x = self.dropout2(x, training=training)
        
        return self.dense2(x)
    
    def get_model_summary(self):
        """모델 구조 정보 반환"""
        if not self.built:
            # input_shape을 명시적으로 지정해서 빌드
            # MNIST의 경우 (28, 28, 1) 또는 (None, 28, 28, 1)
            self.build(input_shape=(None, 28, 28, 1))  # 배치 크기는 None
        total_params = self.count_params()
        trainable_params = sum([tf.size(w).numpy() for w in self.trainable_weights])
        
        return {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "model_layers": len(self.layers),
            "model_size_mb": self._calculate_model_size()
        }
    
    def _calculate_model_size(self):
        """모델 크기 계산 (온디바이스 배포용)"""
        # 가중치 크기 계산
        total_size = 0
        for weight in self.weights:
            total_size += tf.size(weight).numpy() * 4  # float32 = 4 bytes
        
        return total_size / (1024 * 1024)  # MB 단위
    
    def convert_to_tflite(self, representative_dataset=None):
        """TensorFlow Lite 모델로 변환 (온디바이스 배포용)"""
        
        # TFLite 변환기 설정
        converter = tf.lite.TFLiteConverter.from_keras_model(self)
        
        # 양자화 설정 (모델 크기 감소)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        # 대표 데이터셋이 있으면 양자화 적용
        if representative_dataset:
            converter.representative_dataset = representative_dataset
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            converter.inference_input_type = tf.uint8
            converter.inference_output_type = tf.uint8
        
        # 변환 수행
        tflite_model = converter.convert()
        
        return tflite_model

def create_model_architecture(input_shape=(28, 28, 1), num_classes=10):
    """함수형 API를 사용한 모델 생성 (대안)"""
    
    inputs = keras.Input(shape=input_shape)
    
    # 첫 번째 블록
    x = layers.Conv2D(16, 3, activation='relu', padding='same')(inputs)
    x = layers.MaxPooling2D(2)(x)
    
    # 두 번째 블록
    x = layers.Conv2D(32, 3, activation='relu', padding='same')(x)
    x = layers.MaxPooling2D(2)(x)
    
    # 정규화 및 드롭아웃
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.25)(x)
    
    # 완전연결층
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = keras.Model(inputs, outputs, name='mnist_cnn_functional')
    
    return model