import tensorflow as tf
import numpy as np
import mlflow
import matplotlib.pyplot as plt
import seaborn as sns

class DataValidator:
    def __init__(self):
        self.validation_results = {}
        
    def validate_data_quality(self, dataset, dataset_name="dataset"):
        """데이터 품질 검증"""
        
        total_samples = 0
        label_counts = np.zeros(10)
        pixel_values = []
        
        # 데이터 통계 수집
        for batch_images, batch_labels in dataset:
            total_samples += batch_images.shape[0]
            
            # 레이블 분포 계산
            for label in batch_labels:
                label_counts[label.numpy()] += 1
                
            # 픽셀 값 샘플링 (메모리 효율성을 위해 일부만)
            if len(pixel_values) < 10000:
                sample_pixels = tf.reshape(batch_images[:10], [-1]).numpy()
                pixel_values.extend(sample_pixels[:1000])
        
        pixel_values = np.array(pixel_values)
        
        # 검증 결과
        validation_results = {
            "total_samples": total_samples,
            "label_distribution": label_counts,
            "pixel_mean": np.mean(pixel_values),
            "pixel_std": np.std(pixel_values),
            "pixel_min": np.min(pixel_values),
            "pixel_max": np.max(pixel_values),
            "is_normalized": np.max(pixel_values) <= 1.0 and np.min(pixel_values) >= 0.0
        }
        
        # MLflow 로깅
        with mlflow.start_run(run_name=f"data_validation_{dataset_name}"):
            mlflow.log_metric(f"{dataset_name}_total_samples", total_samples)
            mlflow.log_metric(f"{dataset_name}_pixel_mean", validation_results["pixel_mean"])
            mlflow.log_metric(f"{dataset_name}_pixel_std", validation_results["pixel_std"])
            
            # 클래스 불균형 확인
            class_balance = np.min(label_counts) / np.max(label_counts)
            mlflow.log_metric(f"{dataset_name}_class_balance_ratio", class_balance)
            
            # 레이블 분포 시각화
            self._plot_label_distribution(label_counts, dataset_name)
            
        self.validation_results[dataset_name] = validation_results
        return validation_results
    
    def _plot_label_distribution(self, label_counts, dataset_name):
        """레이블 분포 시각화"""
        plt.figure(figsize=(10, 6))
        
        # 막대 그래프
        plt.subplot(1, 2, 1)
        plt.bar(range(10), label_counts)
        plt.title(f'{dataset_name} - Label Distribution')
        plt.xlabel('Digit')
        plt.ylabel('Count')
        
        # 비율 원형 차트
        plt.subplot(1, 2, 2)
        plt.pie(label_counts, labels=range(10), autopct='%1.1f%%')
        plt.title(f'{dataset_name} - Label Proportion')
        
        plt.tight_layout()
        
        # MLflow에 이미지 저장
        plot_path = f"./logs/training/{dataset_name}_label_distribution.png"
        plt.savefig(plot_path)
        mlflow.log_artifact(plot_path)
        plt.close()
    
    def check_data_leakage(self, train_dataset, val_dataset, num_samples=1000):
        """데이터 누출 검사"""
        
        # 샘플 이미지 수집
        train_samples = []
        val_samples = []
        
        for images, _ in train_dataset.take(num_samples // 32):
            train_samples.extend(images.numpy())
            
        for images, _ in val_dataset.take(num_samples // 32):
            val_samples.extend(images.numpy())
        
        train_samples = np.array(train_samples[:num_samples])
        val_samples = np.array(val_samples[:num_samples])
        
        # 중복 검사 (해시 기반)
        train_hashes = set()
        val_hashes = set()
        
        for img in train_samples:
            train_hashes.add(hash(img.tobytes()))
            
        for img in val_samples:
            val_hashes.add(hash(img.tobytes()))
        
        # 교집합 확인
        overlap = len(train_hashes.intersection(val_hashes))
        overlap_ratio = overlap / min(len(train_hashes), len(val_hashes))
        
        with mlflow.start_run(run_name="data_leakage_check"):
            mlflow.log_metric("data_overlap_count", overlap)
            mlflow.log_metric("data_overlap_ratio", overlap_ratio)
            mlflow.log_param("samples_checked", num_samples)
        
        return {
            "overlap_count": overlap,
            "overlap_ratio": overlap_ratio,
            "is_leakage_detected": overlap_ratio > 0.01  # 1% 이상이면 문제
        }