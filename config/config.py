import yaml
import os

def load_config(config_path=None):
    """config.yml 파일을 로드하여 딕셔너리로 반환"""
    if config_path is None:
        # 현재 파일 기준으로 config.yml 경로 설정
        current_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(current_dir, 'config.yml')
    
    try:
        with open(config_path, 'r', encoding='utf-8') as file:
            config = yaml.safe_load(file)
        return config
    except FileNotFoundError:
        raise FileNotFoundError(f"설정 파일을 찾을 수 없습니다: {config_path}")
    except yaml.YAMLError as e:
        raise ValueError(f"YAML 파싱 오류: {e}")