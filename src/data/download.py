import tensorflow as tf
import tensorflow_datasets as tfds
import mlflow
import logging
import os
from pathlib import Path

class MNISTDataLoader:
    def __init__(self, data_dir="./data/raw"):
        self.data_dir = data_dir
        Path(data_dir).mkdir(parents=True, exist_ok=True)
        
    def download_data(self):
        """MNIST 데이터 다운로드 및 로드"""
        try:
            # TensorFlow Datasets를 사용하여 MNIST 다운로드
            (ds_train, ds_test), ds_info = tfds.load(
                'mnist',
                split=['train', 'test'],
                shuffle_files=True,
                as_supervised=True,
                with_info=True,
                data_dir=self.data_dir
            )
            
            # MLflow로 데이터 정보 로깅
            with mlflow.start_run(run_name="data_download"):
                mlflow.log_param("train_size", ds_info.splits['train'].num_examples)
                mlflow.log_param("test_size", ds_info.splits['test'].num_examples)
                mlflow.log_param("num_classes", ds_info.features['label'].num_classes)
                mlflow.log_param("image_shape", ds_info.features['image'].shape)
                mlflow.log_param("dataset_size_mb", ds_info.dataset_size / (1024**2))
                
            logging.info(f"MNIST 데이터 다운로드 완료")
            logging.info(f"훈련 데이터: {ds_info.splits['train'].num_examples}개")
            logging.info(f"테스트 데이터: {ds_info.splits['test'].num_examples}개")
            
            return ds_train, ds_test, ds_info
            
        except Exception as e:
            logging.error(f"데이터 다운로드 실패: {str(e)}")
            raise

    def save_sample_images(self, dataset, num_samples=10):
        """샘플 이미지 저장 (데이터 품질 확인용)"""
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 5, figsize=(12, 6))
        axes = axes.ravel()
        
        for idx, (image, label) in enumerate(dataset.take(num_samples)):
            axes[idx].imshow(image.numpy().squeeze(), cmap='gray')
            axes[idx].set_title(f'Label: {label.numpy()}')
            axes[idx].axis('off')
            
        plt.tight_layout()
        sample_path = os.path.join(self.data_dir, 'sample_images.png')
        plt.savefig(sample_path)
        plt.close()
        
        return sample_path