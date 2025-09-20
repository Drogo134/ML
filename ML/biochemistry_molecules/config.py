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
    
    MOLECULAR_FEATURES = {
        'descriptors': [
            'MW', 'LogP', 'HBD', 'HBA', 'TPSA', 'RotBonds', 'AromaticRings',
            'HeavyAtoms', 'FormalCharge', 'NumHDonors', 'NumHAcceptors'
        ],
        'fingerprints': ['Morgan', 'MACCS', 'RDKit', 'ECFP4', 'ECFP6'],
        'graph_features': ['node_features', 'edge_features', 'graph_features']
    }
    
    MODEL_PARAMS = {
        'gcn': {
            'hidden_dim': 64,
            'num_layers': 3,
            'dropout': 0.3,
            'learning_rate': 0.001,
            'epochs': 100,
            'batch_size': 32
        },
        'gat': {
            'hidden_dim': 64,
            'num_heads': 8,
            'num_layers': 3,
            'dropout': 0.3,
            'learning_rate': 0.001,
            'epochs': 100,
            'batch_size': 32
        },
        'transformer': {
            'd_model': 128,
            'nhead': 8,
            'num_layers': 6,
            'dropout': 0.1,
            'learning_rate': 0.001,
            'epochs': 100,
            'batch_size': 32
        },
        'xgboost': {
            'n_estimators': 1000,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': RANDOM_STATE
        }
    }
    
    DATASETS = {
        'tox21': {
            'url': 'https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/tox21.csv.gz',
            'tasks': ['NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 'NR-ER', 'NR-ER-LBD', 'NR-PPAR-gamma', 'SR-ARE', 'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53']
        },
        'bace': {
            'url': 'https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/bace.csv',
            'tasks': ['Class']
        },
        'bbbp': {
            'url': 'https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/bbbp.csv',
            'tasks': ['p_np']
        },
        'clintox': {
            'url': 'https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/clintox.csv',
            'tasks': ['CT_TOX', 'FDA_APPROVED']
        }
    }
    
    @classmethod
    def create_directories(cls):
        for directory in [cls.DATA_DIR, cls.MODELS_DIR, cls.RESULTS_DIR, cls.LOGS_DIR]:
            directory.mkdir(exist_ok=True)
