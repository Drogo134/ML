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
    
    FEATURE_GROUPS = {
        'demographic': ['age', 'gender', 'education', 'income', 'location'],
        'behavioral': ['session_duration', 'page_views', 'click_rate', 'time_on_site'],
        'temporal': ['hour', 'day_of_week', 'month', 'season'],
        'contextual': ['device_type', 'browser', 'referrer', 'campaign']
    }
    
    MODEL_PARAMS = {
        'xgboost': {
            'n_estimators': 1000,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': RANDOM_STATE
        },
        'lightgbm': {
            'n_estimators': 1000,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': RANDOM_STATE
        },
        'neural_network': {
            'hidden_layers': [128, 64, 32],
            'dropout_rate': 0.3,
            'learning_rate': 0.001,
            'epochs': 100,
            'batch_size': 32
        }
    }
    
    @classmethod
    def create_directories(cls):
        for directory in [cls.DATA_DIR, cls.MODELS_DIR, cls.RESULTS_DIR, cls.LOGS_DIR]:
            directory.mkdir(exist_ok=True)
