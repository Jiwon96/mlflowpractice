import pytest
import yaml
import tempfile
import os
from pathlib import Path
from unittest.mock import patch, mock_open

# utils.common을 import (프로젝트 구조에 맞게 조정)
import sys
sys.path.append('..')  # 또는 적절한 경로
from ex2.utils.common import load_config, merge_config


class TestLoadConfig:
    
    def test_load_config_default_config(self):
        """기본 config.yaml 파일 로드 테스트"""
        config_dir = "../config"
        if Path(config_dir, "config.yaml").exists():
            config_result = load_config(config_name="config", config_dir=config_dir)
        
            assert 'project' in config_result
            assert 'mlflow' in config_result
            assert 'logging' in config_result
            assert config_result['project']['name'] == 'ml-pipeline'
            assert isinstance(config_result['mlflow']['server_port'], int)

    def test_load_config_dev_config(self):
        """dev.yaml 파일 로드 테스트"""
        config_dir = "../config"
        if Path(config_dir, "dev.yaml").exists():
            dev_result = load_config(config_name="dev", config_dir=config_dir)
            
            # 필수 구조만 검증
            assert 'data' in dev_result
            assert 'mlflow' in dev_result
            assert dev_result['logging']['level'] == 'DEBUG'
            assert dev_result['data']['path'].endswith('.csv')

    def test_override_config_default_config(self):
        """기본 """
    
    def test_load_config_file_not_found(self):
        """존재하지 않는 파일에 대한 예외 처리 테스트"""
        with tempfile.TemporaryDirectory() as temp_dir:
            with pytest.raises(FileNotFoundError, match="Config file not found"):
                load_config(config_name="nonexistent", config_dir=temp_dir)

    def test_load_config_invalid_yaml(self):
        """잘못된 YAML 형식에 대한 예외 처리 테스트"""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_file = Path(temp_dir) / "invalid.yaml"
            
            # 잘못된 YAML 내용
            config_file.write_text("invalid: yaml: [", encoding='utf-8')
            
            with pytest.raises(yaml.YAMLError):
                load_config(config_name="invalid", config_dir=temp_dir)

    @pytest.mark.parametrize("config_name,expected_level", [
        ("config", "INFO"),
        ("dev", "DEBUG"),
    ])
    def test_logging_levels(self, config_name, expected_level):
        """로깅 레벨 검증 (파라미터화)"""
        config_dir = "config"
        config_file = Path(config_dir) / f"{config_name}.yaml"
        
        if not config_file.exists():
            pytest.skip(f"Config file not found: {config_file}")
        
        result = load_config(config_name=config_name, config_dir=config_dir)
        assert result['logging']['level'] == expected_level

class TestMergeConfig:
    
    def test_merge_config_simple(self):
        """간단한 딕셔너리 병합 테스트"""
        base = {'a': 1, 'b': 2}
        override = {'b': 3, 'c': 4}
        
        result = merge_config(base, override)
        
        assert result == {'a': 1, 'b': 3, 'c': 4}
        # 원본이 변경되지 않았는지 확인
        assert base == {'a': 1, 'b': 2}

    def test_merge_config_nested(self):
        """중첩된 딕셔너리 병합 테스트"""
        base = {
            'database': {'host': 'localhost', 'port': 5432},
            'app': {'name': 'test'}
        }
        override = {
            'database': {'port': 3306, 'user': 'admin'},
            'logging': {'level': 'DEBUG'}
        }
        
        result = merge_config(base, override)
        
        expected = {
            'database': {'host': 'localhost', 'port': 3306, 'user': 'admin'},
            'app': {'name': 'test'},
            'logging': {'level': 'DEBUG'}
        }
        assert result == expected

    def test_merge_config_real_scenario(self):
        """실제 config 파일 시나리오 테스트"""
        base_config = {
            'project': {'name': 'ml-pipeline'},
            'mlflow': {'tracking_uri': 'http://localhost:5000'},
            'logging': {'level': 'INFO'}
        }
        
        dev_override = {
            'logging': {'level': 'DEBUG'},
            'data': {'path': './data/dev/train.csv'}
        }
        
        result = merge_config(base_config, dev_override)
        
        # 기존 값 유지
        assert result['project']['name'] == 'ml-pipeline'
        assert result['mlflow']['tracking_uri'] == 'http://localhost:5000'
        
        # 새로운 값 추가
        assert result['data']['path'] == './data/dev/train.csv'
        
        # 기존 값 덮어쓰기
        assert result['logging']['level'] == 'DEBUG'

    def test_merge_config_empty_dicts(self):
        """빈 딕셔너리 처리 테스트"""
        assert merge_config({}, {'a': 1}) == {'a': 1}
        assert merge_config({'a': 1}, {}) == {'a': 1}
        assert merge_config({}, {}) == {}

    def test_merge_config_list_override(self):
        """리스트는 덮어쓰기되는지 테스트"""
        base = {'features': ['a', 'b'], 'settings': {'debug': True}}
        override = {'features': ['c', 'd']}
        
        result = merge_config(base, override)
        
        # 리스트는 병합되지 않고 덮어쓰기됨
        assert result['features'] == ['c', 'd']
        assert result['settings']['debug'] == True

if __name__ == "__main__":
    # 직접 실행 시 pytest 실행
    pytest.main([__file__, "-v"])