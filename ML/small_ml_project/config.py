import os
from pathlib import Path

class Config:
    BASE_DIR = Path(__file__).parent
    DATA_DIR = BASE_DIR / "data"
    MODELS_DIR = BASE_DIR / "models"
    RESULTS_DIR = BASE_DIR / "results"
    LOGS_DIR = BASE_DIR / "logs"
    
    RANDOM_STATE = 42
    TEST_SIZE = 0.2
    VALIDATION_SIZE = 0.2
    
    MODEL_PARAMS = {
        'random_forest': {
            'n_estimators': 100,
            'max_depth': 10,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'random_state': RANDOM_STATE
        },
        'xgboost': {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': RANDOM_STATE
        },
        'lightgbm': {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': RANDOM_STATE
        },
        'neural_network': {
            'hidden_layers': [64, 32],
            'dropout_rate': 0.3,
            'learning_rate': 0.001,
            'epochs': 50,
            'batch_size': 32
        }
    }
    
    @classmethod
    def create_directories(cls):
        for directory in [cls.DATA_DIR, cls.MODELS_DIR, cls.RESULTS_DIR, cls.LOGS_DIR]:
            directory.mkdir(exist_ok=True)
