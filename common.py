import yaml
import os
import json
import pandas as pd
import logging
from pathlib import Path
from sklearn.model_selection import train_test_split
import copy

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_config(config_name="config", config_dir="config"):
    """Load configuration from YAML file in config directory"""
    config_path = Path(config_dir) / f"{config_name}.yaml"
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def merge_configs(base_config, override_config):
    """딕셔너리 깊은 병합"""
    result = copy.deepcopy(base_config)
    
    def deep_merge(base_dict, override_dict):
        for key, value in override_dict.items():
            if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                deep_merge(base_dict[key], value)
            else:
                base_dict[key] = value
    
    deep_merge(result, override_config)
    return result

def load_config_by_env(env=None, config_dir="config"):
    """Load base config and override with environment-specific config"""
    if env is None:
        env = os.getenv("ENV", "dev")
    
    try:
        # 1. 기본 config 로드 (필수)
        base_config = load_config("config", config_dir)
        logger.info(f"Loaded base config from: config/config.yaml")
        
        # 2. 환경별 config가 있으면 오버라이드
        try:
            env_config = load_config(env, config_dir)
            final_config = merge_configs(base_config, env_config)
            logger.info(f"Applied overrides from: config/{env}.yaml")
            return final_config
        except FileNotFoundError:
            logger.info(f"No environment config found for '{env}', using base config only")
            return base_config
            
    except FileNotFoundError:
        raise FileNotFoundError(f"Base config file 'config.yaml' not found in {config_dir}/ directory")

def build_data(config=None):
    """데이터 로딩 및 분할"""
    if config is None:
        config = CONFIG
    
    data_config = config['data']
    data_path = data_config['path']
    col_label = data_config['target_column']
    test_size = data_config['test_size']
    random_state = data_config['random_state']
    
    # 파일 존재 확인
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    logger.info(f"Loading data from: {data_path}")
    data = pd.read_csv(data_path)
    
    # 데이터 검증
    if col_label not in data.columns:
        raise ValueError(f"Label column '{col_label}' not found in data")
    
    logger.info(f"Data shape: {data.shape}")
    logger.info(f"Target column: {col_label}")
    
    # 데이터 분할
    train, test = train_test_split(data, test_size=test_size, random_state=random_state)
    
    X_train = train.drop([col_label], axis=1)
    X_test = test.drop([col_label], axis=1)
    y_train = train[[col_label]]
    y_test = test[[col_label]]
    
    logger.info(f"Train set: {X_train.shape[0]} samples")
    logger.info(f"Test set: {X_test.shape[0]} samples")
    
    return X_train, X_test, y_train, y_test

def validate_data(X_train, X_test, y_train, y_test):
    """Basic data validation"""
    assert len(X_train) == len(y_train), "Training data length mismatch"
    assert len(X_test) == len(y_test), "Test data length mismatch"
    assert X_train.shape[1] == X_test.shape[1], "Feature count mismatch"
    
    # NaN 체크
    if X_train.isnull().sum().sum() > 0:
        logger.warning("Training data contains NaN values")
    if X_test.isnull().sum().sum() > 0:
        logger.warning("Test data contains NaN values")
    
    logger.info(f"Data validation passed. Train: {len(X_train)}, Test: {len(X_test)}")

def to_json(data_path=None, num_lines=3):
    """Convert CSV file to MLflow scoring format"""
    if data_path is None:
        data_path = CONFIG['data']['path']
    
    try:
        df = pd.read_csv(data_path)
        # quality 컬럼 제거 (target)
        target_col = CONFIG['data']['target_column']
        if target_col in df.columns:
            df = df.drop(target_col, axis=1)
        
        # 처음 num_lines만 선택
        sample_data = df.head(num_lines)
        
        result = {
            "columns": sample_data.columns.tolist(),
            "data": sample_data.values.tolist()
        }
        return json.dumps(result, indent=2)
    except Exception as e:
        logger.error(f"Error converting to JSON: {e}")
        raise

# 전역 config 객체
CONFIG = load_config_by_env()

# 기존 변수들 (하위 호환성)
data_path = CONFIG['data']['path']
experiment_name = CONFIG['project']['experiment_name']
model_name = CONFIG['project']['model_name']
model_uri = f"models:/{model_name}/{CONFIG['mlflow']['model_stage']}"
port = CONFIG['mlflow']['server_port']