# predict_fixed.py
import mlflow
import mlflow.pyfunc
import pandas as pd
import requests
import json
import numpy as np
from common import CONFIG, build_data

mlflow.set_tracking_uri(CONFIG['mlflow']['tracking_uri'])
print(f"MLflow Tracking URI: {mlflow.get_tracking_uri()}")

def predict_with_loaded_model():
    """직접 모델을 로드해서 예측"""
    model_name = CONFIG['project']['model_name']
    model_stage = CONFIG['mlflow']['model_stage']
    model_uri = f"models:/{model_name}/{model_stage}"
    
    try:
        # 모델 로드
        model = mlflow.pyfunc.load_model(model_uri)
        
        # 테스트 데이터 준비
        X_train, X_test, y_train, y_test = build_data()
        
        # 예측
        predictions = model.predict(X_test.head(5))
        
        print("Direct Model Predictions:")
        for i, pred in enumerate(predictions):
            actual = y_test.iloc[i].values[0]
            print(f"Sample {i+1}: Predicted={pred:.2f}, Actual={actual:.2f}")
        
        return predictions
    except Exception as e:
        print(f"Error in direct prediction: {e}")
        return None

def predict_with_api_v2():
    """MLflow 2.0+ 호환 API 예측"""
    
    # 테스트 데이터 준비
    X_train, X_test, y_train, y_test = build_data()
    
    # 첫 번째 샘플 사용
    sample_data = X_test.iloc[0].to_dict()
    feature_names = list(X_test.columns)
    sample_values = [X_test.iloc[0].tolist()]
    
    # MLflow 2.0+ 호환 형식들 (inputs 형식 제거 - 스키마 문제로 인해)
    request_formats = [
        {
            "name": "dataframe_split",
            "data": {
                "dataframe_split": {
                    "columns": feature_names,
                    "data": sample_values
                }
            }
        },
        {
            "name": "instances", 
            "data": {
                "instances": [sample_data]
            }
        }
        # inputs 형식은 제거 - MLflow 모델이 feature 이름을 요구함
    ]
    
    url = "http://localhost:8081/invocations"
    headers = {"Content-Type": "application/json"}
    
    successful_prediction = None
    
    for format_info in request_formats:
        try:
            print(f"\n=== Testing {format_info['name']} format ===")
            print(f"Request data preview: {format_info['name']} with {len(feature_names)} features")
            
            response = requests.post(url, json=format_info['data'], headers=headers, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                
                # 응답 구조 디버깅
                print(f"Raw response type: {type(result)}")
                print(f"Raw response: {result}")
                
                # 결과 파싱 (다양한 형태 처리)
                prediction = None
                
                if isinstance(result, list):
                    # [1.23] 또는 [[1.23]] 형태
                    if len(result) > 0:
                        if isinstance(result[0], (list, np.ndarray)):
                            prediction = result[0][0]  # [[1.23]] -> 1.23
                        else:
                            prediction = result[0]  # [1.23] -> 1.23
                elif isinstance(result, dict):
                    # {"prediction": 1.23} 또는 {"predictions": [1.23]} 형태
                    if 'prediction' in result:
                        prediction = result['prediction']
                    elif 'predictions' in result:
                        predictions = result['predictions']
                        if isinstance(predictions, list) and len(predictions) > 0:
                            prediction = predictions[0]
                    else:
                        # dict 자체가 예측 결과일 수도 있음 - 첫 번째 값 사용
                        values = list(result.values())
                        if values:
                            prediction = values[0]
                elif isinstance(result, (int, float)):
                    # 직접 숫자
                    prediction = result
                else:
                    print(f"Unexpected result type: {type(result)}")
                    print(f"Result content: {result}")
                    continue
                
                # 최종 숫자 변환
                if prediction is None:
                    print(f"Could not extract prediction from: {result}")
                    continue
                
                # 리스트나 배열이면 첫 번째 요소 추출
                if isinstance(prediction, (list, np.ndarray)):
                    if len(prediction) > 0:
                        prediction = prediction[0]
                    else:
                        print("Empty prediction array")
                        continue
                
                # 최종 float 변환
                try:
                    prediction = float(prediction)
                except (ValueError, TypeError) as e:
                    print(f"Cannot convert prediction to float: {prediction}, error: {e}")
                    continue
                
                actual = float(y_test.iloc[0].values[0])
                
                print(f"✅ {format_info['name']} format SUCCESS")
                print(f"Predicted: {prediction:.2f}")
                print(f"Actual: {actual:.2f}")
                print(f"Difference: {abs(prediction - actual):.2f}")
                
                successful_prediction = prediction
                break  # 성공하면 중단
                
            else:
                print(f"❌ {format_info['name']} format FAILED: {response.status_code}")
                error_msg = response.text[:200] + "..." if len(response.text) > 200 else response.text
                print(f"Error: {error_msg}")
                
        except requests.exceptions.ConnectionError:
            print("❌ Cannot connect to API server")
            print("Make sure the server is running: python serve_model.py")
            return None
        except requests.exceptions.Timeout:
            print(f"❌ {format_info['name']} format TIMEOUT")
        except ValueError as e:
            print(f"❌ {format_info['name']} format VALUE ERROR: {e}")
        except Exception as e:
            print(f"❌ {format_info['name']} format UNEXPECTED ERROR: {e}")
    
    return successful_prediction

def get_sample_request_format():
    """올바른 요청 형식 예시 생성"""
    try:
        X_train, X_test, y_train, y_test = build_data()
        sample = X_test.iloc[0]
        
        formats = {
            "dataframe_split": {
                "dataframe_split": {
                    "columns": list(sample.index),
                    "data": [sample.tolist()]
                }
            },
            "instances": {
                "instances": [sample.to_dict()]
            }
            # inputs 형식 제거 - feature 이름이 필요한 모델에서는 작동하지 않음
        }
        
        print("=== MLflow 2.0+ Compatible Request Formats ===")
        for name, format_data in formats.items():
            print(f"\n{name.upper()} format:")
            print(json.dumps(format_data, indent=2)[:300] + "..." if len(json.dumps(format_data, indent=2)) > 300 else json.dumps(format_data, indent=2))
        
        print(f"\nFeature names: {list(sample.index)}")
        print(f"Sample values: {sample.tolist()}")
        
        return formats
        
    except Exception as e:
        print(f"Error generating sample formats: {e}")
        return None

def test_api_connection():
    """API 서버 연결 테스트"""
    try:
        response = requests.get("http://localhost:8081/health", timeout=5)
        if response.status_code == 200:
            print("✅ API server is running")
            return True
        else:
            print(f"❌ API server returned: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ API server is not accessible")
        return False
    except Exception as e:
        print(f"❌ Connection test failed: {e}")
        return False

def compare_predictions():
    """직접 모델과 API 예측 결과 비교"""
    print("="*60)
    print("🔍 PREDICTION COMPARISON")
    print("="*60)
    
    # 1. 직접 모델 예측
    print("\n1️⃣ Direct Model Prediction:")
    direct_predictions = predict_with_loaded_model()
    
    if direct_predictions is None:
        print("❌ Direct prediction failed")
        return
    
    # 2. API 연결 확인
    print("\n2️⃣ API Connection Test:")
    if not test_api_connection():
        print("❌ API is not available. Make sure to run: python serve_model.py")
        return
    
    # 3. API 예측
    print("\n3️⃣ API Prediction:")
    api_prediction = predict_with_api_v2()
    
    # 4. 결과 비교
    if api_prediction is not None and direct_predictions is not None:
        direct_first = float(direct_predictions[0])
        api_first = float(api_prediction)
        
        print("\n" + "="*60)
        print("📊 RESULTS SUMMARY")
        print("="*60)
        print(f"Direct Model Prediction: {direct_first:.4f}")
        print(f"API Prediction:          {api_first:.4f}")
        print(f"Difference:              {abs(direct_first - api_first):.6f}")
        
        if abs(direct_first - api_first) < 0.001:
            print("✅ Predictions match! API is working correctly.")
        else:
            print("⚠️  Predictions differ. Check model versions or data preprocessing.")

if __name__ == "__main__":
    compare_predictions()