import yaml
import os
import json
import pandas as pd
import logging
from pathlib import Path
from sklearn.model_selection import train_test_split
import copy

def load_config(config_name="config", config_dir="config"):
    """Load configuration from YAML file in config directory"""
    config_path = Path(config_dir) / f"{config_name}.yaml"

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def merge_config(base_config, override_config):
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
    if env is None:
        env = os.getenv("ENV", "dev")

    try:
        base_config = load_config("config", config_dir)
        logger.info(f"Loaded base config from: config/config.yaml")

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

CONFIG = load_config_by_env()
data_path = CONFIG['data']['path']
experiment_name = CONFIG['project']['experiment_name']
model_name = CONFIG['project']['model_name']
model_uri = f"models:/{model_name}/{CONFIG['mlflow']['model_stage']}"
port = CONFIG['mlflow']['server_port']
        