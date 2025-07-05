import tensorflow as tf
import numpy as np
import mlflow
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import os

class ModelEvaluator:
    def __init__(self, model, test_dataset):
        self.model = model
        self.test_dataset = test_dataset
        self.class_names = [str(i) for i in range(10)]  # MNIST 클래스명
        
    def evaluate_model(self):
        """모델 전체 평가"""
        
        with mlflow.start_run(run_name="model_evaluation"):
            # 기본 평가 메트릭
            test_loss, test_accuracy = self.model.evaluate(self.test_dataset, verbose=1)
            
            mlflow.log_metric("test_loss", test_loss)
            mlflow.log_metric("test_accuracy", test_accuracy)
            
            # 상세 분석
            predictions, true_labels = self._get_predictions()
            
            # 분류 리포트
            classification_rep = self._generate_classification_report(true_labels, predictions)
            
            # 혼동 행렬
            cm_path = self._plot_confusion_matrix(true_labels, predictions)
            
            # 클래스별 성능 분석
            class_metrics = self._analyze_class_performance(true_labels, predictions)
            
            # 예측 확신도 분석
            confidence_analysis = self._analyze_prediction_confidence(true_labels, predictions)
            
            # 오분류 샘플 분석
            error_analysis = self._analyze_misclassified_samples(true_labels, predictions)
            
            # MLflow 로깅
            mlflow.log_artifact(cm_path)
            
            for class_idx, metrics in class_metrics.items():
                mlflow.log_metric(f"class_{class_idx}_precision", metrics['precision'])
                mlflow.log_metric(f"class_{class_idx}_recall", metrics['recall'])
                mlflow.log_metric(f"class_{class_idx}_f1", metrics['f1'])
            
            return {
                "test_accuracy": test_accuracy,
                "test_loss": test_loss,
                "classification_report": classification_rep,
                "class_metrics": class_metrics,
                "confidence_analysis": confidence_analysis,
                "error_analysis": error_analysis
            }
    
    def _get_predictions(self):
        """테스트 데이터에 대한 예측 수행"""
        predictions = []
        true_labels = []
        
        for images, labels in self.test_dataset:
            batch_predictions = self.model.predict(images, verbose=0)
            predictions.extend(np.argmax(batch_predictions, axis=1))
            true_labels.extend(labels.numpy())
        
        return np.array(predictions), np.array(true_labels)
    
    def _generate_classification_report(self, true_labels, predictions):
        """분류 리포트 생성"""
        report = classification_report(
            true_labels, 
            predictions, 
            target_names=self.class_names,
            output_dict=True
        )
        
        # 텍스트 형태로도 저장
        report_text = classification_report(
            true_labels, 
            predictions, 
            target_names=self.class_names
        )
        
        report_path = "./logs/training/classification_report.txt"
        with open(report_path, 'w') as f:
            f.write(report_text)
        
        mlflow.log_artifact(report_path)
        
        return report
    
    def _plot_confusion_matrix(self, true_labels, predictions):
        """혼동 행렬 시각화"""
        cm = confusion_matrix(true_labels, predictions)
        
        plt.figure(figsize=(12, 10))
        
        # 정규화된 혼동 행렬
        plt.subplot(2, 2, 1)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=self.class_names, yticklabels=self.class_names)
        plt.title('Confusion Matrix (Raw Counts)')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        # 비율 혼동 행렬
        plt.subplot(2, 2, 2)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                   xticklabels=self.class_names, yticklabels=self.class_names)
        plt.title('Confusion Matrix (Normalized)')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        # 클래스별 정확도
        plt.subplot(2, 2, 3)
        class_accuracy = np.diag(cm) / np.sum(cm, axis=1)
        plt.bar(self.class_names, class_accuracy)
        plt.title('Per-Class Accuracy')
        plt.xlabel('Class')
        plt.ylabel('Accuracy')
        plt.xticks(rotation=45)
        
        # 클래스별 예측 분포
        plt.subplot(2, 2, 4)
        pred_counts = np.sum(cm, axis=0)
        true_counts = np.sum(cm, axis=1)
        
        x = np.arange(len(self.class_names))
        width = 0.35
        
        plt.bar(x - width/2, true_counts, width, label='True', alpha=0.7)
        plt.bar(x + width/2, pred_counts, width, label='Predicted', alpha=0.7)
        plt.xlabel('Class')
        plt.ylabel('Count')
        plt.title('True vs Predicted Distribution')
        plt.xticks(x, self.class_names)
        plt.legend()
        
        plt.tight_layout()
        
        cm_path = "./logs/training/confusion_matrix.png"
        plt.savefig(cm_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return cm_path
    
    def _analyze_class_performance(self, true_labels, predictions):
        """클래스별 성능 분석"""
        from sklearn.metrics import precision_recall_fscore_support
        
        precision, recall, f1, support = precision_recall_fscore_support(
            true_labels, predictions, average=None
        )
        
        class_metrics = {}
        for i in range(len(self.class_names)):
            class_metrics[i] = {
                'precision': precision[i],
                'recall': recall[i],
                'f1': f1[i],
                'support': support[i]
            }
        
        return class_metrics
    
    def _analyze_prediction_confidence(self, true_labels, predictions):
        """예측 확신도 분석"""
        confidences = []
        correct_confidences = []
        incorrect_confidences = []
        
        for images, labels in self.test_dataset:
            batch_probs = self.model.predict(images, verbose=0)
            batch_confidences = np.max(batch_probs, axis=1)
            batch_predictions = np.argmax(batch_probs, axis=1)
            
            confidences.extend(batch_confidences)
            
            for conf, pred, true_label in zip(batch_confidences, batch_predictions, labels.numpy()):
                if pred == true_label:
                    correct_confidences.append(conf)
                else:
                    incorrect_confidences.append(conf)
        
        return {
            "mean_confidence": np.mean(confidences),
            "correct_mean_confidence": np.mean(correct_confidences),
            "incorrect_mean_confidence": np.mean(incorrect_confidences),
            "confidence_std": np.std(confidences)
        }
    
    def _analyze_misclassified_samples(self, true_labels, predictions):
        """오분류 샘플 분석"""
        misclassified_indices = np.where(true_labels != predictions)[0]
        
        # 가장 많이 오분류된 클래스 쌍
        misclassification_pairs = {}
        for idx in misclassified_indices:
            pair = (true_labels[idx], predictions[idx])
            misclassification_pairs[pair] = misclassification_pairs.get(pair, 0) + 1
        
        # 상위 5개 오분류 패턴
        top_errors = sorted(misclassification_pairs.items(), 
                          key=lambda x: x[1], reverse=True)[:5]
        
        return {
            "total_misclassified": len(misclassified_indices),
            "misclassification_rate": len(misclassified_indices) / len(true_labels),
            "top_confusion_pairs": top_errors
        }
    
    def evaluate_tflite_model(self, tflite_model_path):
        """TensorFlow Lite 모델 평가"""
        
        # TFLite 인터프리터 로드
        interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
        interpreter.allocate_tensors()
        
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        correct_predictions = 0
        total_predictions = 0
        
        for images, labels in self.test_dataset:
            for i in range(len(images)):
                # 입력 데이터 설정
                input_data = np.expand_dims(images[i].numpy(), axis=0).astype(np.float32)
                interpreter.set_tensor(input_details[0]['index'], input_data)
                
                # 추론 실행
                interpreter.invoke()
                
                # 결과 가져오기
                output_data = interpreter.get_tensor(output_details[0]['index'])
                predicted_class = np.argmax(output_data[0])
                
                if predicted_class == labels[i].numpy():
                    correct_predictions += 1
                total_predictions += 1
        
        tflite_accuracy = correct_predictions / total_predictions
        
        with mlflow.start_run(run_name="tflite_evaluation"):
            mlflow.log_metric("tflite_accuracy", tflite_accuracy)
            
        return tflite_accuracy