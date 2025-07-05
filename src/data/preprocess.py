import tensorflow as tf
import mlflow
import logging

class DataPreprocessor:
    def __init__(self, batch_size=32, validation_split=0.2, buffer_size=1000):
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.buffer_size = buffer_size
        
    def normalize_img(self, image, label):
        """이미지 정규화"""
        # uint8 -> float32 변환 및 [0,1] 범위로 정규화
        image = tf.cast(image, tf.float32) / 255.0
        return image, label
    
    def augment_data(self, image, label):
        """데이터 증강 (훈련 데이터용)"""
        # 랜덤 회전 (-10도 ~ +10도)
        image = tf.image.rot90(image, k=tf.random.uniform([], 0, 4, dtype=tf.int32))
        
        # 약간의 노이즈 추가
        noise = tf.random.normal(shape=tf.shape(image), mean=0.0, stddev=0.01)
        image = tf.clip_by_value(image + noise, 0.0, 1.0)
        
        return image, label
    
    def create_data_pipeline(self, ds_train, ds_test, ds_info):
        """효율적인 데이터 파이프라인 생성"""
        
        # 훈련 데이터 크기 계산
        train_size = ds_info.splits['train'].num_examples
        val_size = int(train_size * self.validation_split)
        train_size = train_size - val_size
        
        with mlflow.start_run(run_name="data_preprocessing"):
            # 훈련/검증 데이터 분할
            ds_train_split = ds_train.take(train_size)
            ds_val = ds_train.skip(train_size)
            
            # 훈련 데이터 파이프라인
            ds_train_processed = (ds_train_split
                                .map(self.normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
                                .map(self.augment_data, num_parallel_calls=tf.data.AUTOTUNE)
                                .cache()
                                .shuffle(self.buffer_size)
                                .batch(self.batch_size)
                                .prefetch(tf.data.AUTOTUNE))
            
            # 검증 데이터 파이프라인
            ds_val_processed = (ds_val
                              .map(self.normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
                              .cache()
                              .batch(self.batch_size)
                              .prefetch(tf.data.AUTOTUNE))
            
            # 테스트 데이터 파이프라인
            ds_test_processed = (ds_test
                               .map(self.normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
                               .cache()
                               .batch(self.batch_size)
                               .prefetch(tf.data.AUTOTUNE))
            
            # MLflow 파라미터 로깅
            mlflow.log_param("batch_size", self.batch_size)
            mlflow.log_param("validation_split", self.validation_split)
            mlflow.log_param("buffer_size", self.buffer_size)
            mlflow.log_param("train_size", train_size)
            mlflow.log_param("val_size", val_size)
            mlflow.log_param("test_size", ds_info.splits['test'].num_examples)
            
            logging.info("데이터 파이프라인 생성 완료")
            logging.info(f"훈련: {train_size}, 검증: {val_size}, 테스트: {ds_info.splits['test'].num_examples}")
            
        return ds_train_processed, ds_val_processed, ds_test_processed
    
    def get_dataset_info(self, dataset):
        """데이터셋 정보 확인"""
        for batch in dataset.take(1):
            images, labels = batch
            print(f"배치 크기: {images.shape[0]}")
            print(f"이미지 형태: {images.shape}")
            print(f"레이블 형태: {labels.shape}")
            print(f"픽셀 값 범위: [{tf.reduce_min(images):.3f}, {tf.reduce_max(images):.3f}]")
            break