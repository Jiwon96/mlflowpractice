# debug_api.py - API 응답 구조 확인용
import requests
import json
from common import build_data

def debug_api_response():
    """API 응답 구조를 자세히 확인"""
    
    # 테스트 데이터 준비
    X_train, X_test, y_train, y_test = build_data()
    sample = X_test.iloc[0]
    
    # 간단한 요청 데이터
    request_data = {
        "dataframe_split": {
            "columns": list(sample.index),
            "data": [sample.tolist()]
        }
    }
    
    print("="*60)
    print("🔍 API DEBUG - 응답 구조 분석")
    print("="*60)
    
    print("\n📤 Request Data:")
    print(f"URL: http://localhost:8080/invocations")
    print(f"Columns ({len(sample.index)}): {list(sample.index)}")
    print(f"Sample data: {sample.tolist()[:3]}... (showing first 3 values)")
    print(f"Actual value: {y_test.iloc[0].values[0]}")
    
    try:
        # API 요청
        response = requests.post(
            "http://localhost:8080/invocations",
            json=request_data,
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        
        print(f"\n📥 Response Info:")
        print(f"Status Code: {response.status_code}")
        print(f"Content-Type: {response.headers.get('content-type', 'unknown')}")
        print(f"Response Size: {len(response.content)} bytes")
        
        if response.status_code == 200:
            result = response.json()
            
            print(f"\n📊 Response Analysis:")
            print(f"Response Type: {type(result)}")
            print(f"Response Content: {result}")
            
            if isinstance(result, list):
                print(f"List Length: {len(result)}")
                if len(result) > 0:
                    print(f"First Element Type: {type(result[0])}")
                    print(f"First Element: {result[0]}")
                    
                    if isinstance(result[0], list):
                        print(f"Nested List Length: {len(result[0])}")
                        if len(result[0]) > 0:
                            print(f"First Nested Element: {result[0][0]} (type: {type(result[0][0])})")
            
            elif isinstance(result, dict):
                print(f"Dict Keys: {list(result.keys())}")
                for key, value in result.items():
                    print(f"  {key}: {value} (type: {type(value)})")
            
            else:
                print(f"Direct Value: {result} (type: {type(result)})")
            
            # 예측값 추출 시도
            print(f"\n🎯 Prediction Extraction:")
            prediction = extract_prediction(result)
            if prediction is not None:
                print(f"✅ Successfully extracted: {prediction}")
                print(f"Type: {type(prediction)}")
            else:
                print("❌ Failed to extract prediction")
                
        else:
            print(f"\n❌ API Error:")
            print(f"Status: {response.status_code}")
            print(f"Response: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print("\n❌ Connection Error:")
        print("API server is not running. Start it with: python serve_model.py")
    except Exception as e:
        print(f"\n❌ Unexpected Error: {e}")

def extract_prediction(result):
    """응답에서 예측값 추출 시도"""
    try:
        if isinstance(result, list):
            if len(result) > 0:
                if isinstance(result[0], (list, tuple)):
                    return float(result[0][0])  # [[1.23]] -> 1.23
                else:
                    return float(result[0])  # [1.23] -> 1.23
        
        elif isinstance(result, dict):
            # 가능한 키들 확인
            possible_keys = ['prediction', 'predictions', 'result', 'output', 'value']
            for key in possible_keys:
                if key in result:
                    value = result[key]
                    if isinstance(value, list):
                        return float(value[0])
                    else:
                        return float(value)
            
            # 키가 없으면 첫 번째 값 시도
            values = list(result.values())
            if values:
                return float(values[0])
        
        elif isinstance(result, (int, float)):
            return float(result)
        
        else:
            return None
            
    except (ValueError, TypeError, IndexError):
        return None

def test_different_endpoints():
    """다른 엔드포인트들도 테스트"""
    
    endpoints = [
        ("GET", "http://localhost:8080/health", None),
        ("GET", "http://localhost:8080/ping", None),
    ]
    
    print(f"\n🌐 Testing Other Endpoints:")
    
    for method, url, data in endpoints:
        try:
            if method == "GET":
                response = requests.get(url, timeout=10)
            else:
                response = requests.post(url, json=data, timeout=10)
            
            print(f"\n{method} {url}")
            print(f"Status: {response.status_code}")
            
            if response.status_code == 200:
                try:
                    content = response.json()
                    print(f"Response: {content}")
                except:
                    print(f"Response (text): {response.text[:100]}...")
            else:
                print(f"Error: {response.text[:100]}...")
                
        except Exception as e:
            print(f"{method} {url} - Error: {e}")

if __name__ == "__main__":
    debug_api_response()
    test_different_endpoints()