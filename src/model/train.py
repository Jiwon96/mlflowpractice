import tensorflow as tf
from tensorflow import keras
import mlflow
import mlflow.tensorflow
import os
import numpy as np
from datetime import datetime
import json

class ModelTrainer:
    def __init__(self, model, train_dataset, val_dataset, config):
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.config = config
        
        # 옵티마이저 및 손실 함수 설정
        self.optimizer = keras.optimizers.Adam(
            learning_rate=config.get('learning_rate', 0.001)
        )
        self.loss_fn = keras.losses.SparseCategoricalCrossentropy()
        
        # 메트릭 설정
        self.train_loss = keras.metrics.Mean(name='train_loss')
        self.train_accuracy = keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
        self.val_loss = keras.metrics.Mean(name='val_loss')
        self.val_accuracy = keras.metrics.SparseCategoricalAccuracy(name='val_accuracy')
        
    def compile_model(self):
        """모델 컴파일"""
        self.model.compile(
            optimizer=self.optimizer,
            loss=self.loss_fn,
            metrics=['accuracy']
        )
        
    @tf.function
    def train_step(self, images, labels):
        """훈련 스텝 (그래프 모드로 최적화)"""
        with tf.GradientTape() as tape:
            predictions = self.model(images, training=True)
            loss = self.loss_fn(labels, predictions)
            
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        
        self.train_loss(loss)
        self.train_accuracy(labels, predictions)
        
        return loss
    
    @tf.function
    def val_step(self, images, labels):
        """검증 스텝"""
        predictions = self.model(images, training=False)
        loss = self.loss_fn(labels, predictions)
        
        self.val_loss(loss)
        self.val_accuracy(labels, predictions)
        
        return loss
    
    def train_epoch(self):
        """한 에포크 훈련"""
        # 메트릭 초기화
        self.train_loss.reset_states()
        self.train_accuracy.reset_states()
        
        # 훈련 루프
        for step, (images, labels) in enumerate(self.train_dataset):
            loss = self.train_step(images, labels)
            
            # 진행상황 출력 (100 스텝마다)
            if step % 100 == 0:
                print(f"Step {step}: Loss = {loss:.4f}, "
                      f"Accuracy = {self.train_accuracy.result():.4f}")
        
        return self.train_loss.result(), self.train_accuracy.result()
    
    def validate_epoch(self):
        """한 에포크 검증"""
        # 메트릭 초기화
        self.val_loss.reset_states()
        self.val_accuracy.reset_states()
        
        # 검증 루프
        for images, labels in self.val_dataset:
            self.val_step(images, labels)
        
        return self.val_loss.result(), self.val_accuracy.result()
    
    def create_callbacks(self, model_save_path):
        """콜백 함수들 생성"""
        callbacks = []
        
        # 모델 체크포인트
        checkpoint_callback = keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(model_save_path, 'best_model.h5'),
            monitor='val_accuracy',
            save_best_only=True,
            save_weights_only=False,
            mode='max',
            verbose=1
        )
        callbacks.append(checkpoint_callback)
        
        # 조기 종료
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=5,
            restore_best_weights=True,
            verbose=1
        )
        callbacks.append(early_stopping)
        
        # 학습률 스케줄링
        lr_scheduler = keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1
        )
        callbacks.append(lr_scheduler)
        
        # TensorBoard 로깅
        tensorboard_callback = keras.callbacks.TensorBoard(
            log_dir=f"./logs/training/tensorboard_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            histogram_freq=1,
            write_graph=True,
            write_images=True
        )
        callbacks.append(tensorboard_callback)
        
        return callbacks
    
    def train(self, epochs=10):
        """전체 훈련 프로세스"""
        
        # 모델 저장 경로
        model_save_path = f"./data/models/{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(model_save_path, exist_ok=True)
        
        with mlflow.start_run(run_name=f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            
            # 모델 정보 로깅
            model_summary = self.model.get_model_summary()
            for key, value in model_summary.items():
                mlflow.log_param(key, value)
            
            # 하이퍼파라미터 로깅
            mlflow.log_param("epochs", epochs)
            mlflow.log_param("learning_rate", self.config.get('learning_rate', 0.001))
            mlflow.log_param("optimizer", "Adam")
            mlflow.log_param("loss_function", "SparseCategoricalCrossentropy")
            
            # 콜백 생성
            callbacks = self.create_callbacks(model_save_path)
            
            # 모델 컴파일
            self.compile_model()
            
            # 모델 구조 저장
            model_architecture = self.model.to_json()
            with open(os.path.join(model_save_path, 'model_architecture.json'), 'w') as f:
                json.dump(json.loads(model_architecture), f, indent=2)
            
            # 훈련 실행
            history = self.model.fit(
                self.train_dataset,
                epochs=epochs,
                validation_data=self.val_dataset,
                callbacks=callbacks,
                verbose=1
            )
            
            # 훈련 히스토리 로깅
            for epoch in range(len(history.history['loss'])):
                mlflow.log_metric("train_loss", history.history['loss'][epoch], step=epoch)
                mlflow.log_metric("train_accuracy", history.history['accuracy'][epoch], step=epoch)
                mlflow.log_metric("val_loss", history.history['val_loss'][epoch], step=epoch)
                mlflow.log_metric("val_accuracy", history.history['val_accuracy'][epoch], step=epoch)
                
                if 'lr' in history.history:
                    mlflow.log_metric("learning_rate", history.history['lr'][epoch], step=epoch)
            
            # 최종 메트릭
            best_val_accuracy = max(history.history['val_accuracy'])
            mlflow.log_metric("best_val_accuracy", best_val_accuracy)
            
            # 모델 저장 및 등록
            self._save_and_register_model(model_save_path, best_val_accuracy)
            
            return history, best_val_accuracy
    
    def _save_and_register_model(self, model_save_path, accuracy):
        """모델 저장 및 MLflow에 등록"""
        
        # Keras 모델 저장
        keras_model_path = os.path.join(model_save_path, 'keras_model.keras')
        self.model.save(keras_model_path)
        
        # TensorFlow Lite 모델 변환 및 저장
        tflite_model = self.model.convert_to_tflite()
        tflite_path = os.path.join(model_save_path, 'model.tflite')
        with open(tflite_path, 'wb') as f:
            f.write(tflite_model)
        
        # MLflow에 모델 등록
        mlflow.tensorflow.log_model(
            self.model,
            "model",
            registered_model_name="mnist_cnn_model"
        )
        
        # 아티팩트 로깅
        mlflow.log_artifacts(model_save_path)
        
        # 모델 크기 정보
        keras_size = sum(os.path.getsize(os.path.join(keras_model_path, f)) 
                        for f in os.listdir(keras_model_path) if os.path.isfile(os.path.join(keras_model_path, f)))
        tflite_size = os.path.getsize(tflite_path)
        
        mlflow.log_metric("keras_model_size_mb", keras_size / (1024**2))
        mlflow.log_metric("tflite_model_size_mb", tflite_size / (1024**2))
        mlflow.log_metric("size_reduction_ratio", tflite_size / keras_size)
        
        print(f"모델 저장 완료: {model_save_path}")
        print(f"Keras 모델 크기: {keras_size / (1024**2):.2f} MB")
        print(f"TFLite 모델 크기: {tflite_size / (1024**2):.2f} MB")
        print(f"크기 감소율: {(1 - tflite_size / keras_size) * 100:.1f}%")