import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression, make_blobs
from sklearn.model_selection import train_test_split
from typing import Tuple, Dict, Any
from config import Config

class DataGenerator:
    def __init__(self, config):
        self.config = config
        
    def generate_classification_data(self, n_samples: int = 1000, n_features: int = 20, 
                                   n_classes: int = 2, n_clusters_per_class: int = 1,
                                   class_sep: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
        X, y = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=int(n_features * 0.7),
            n_redundant=int(n_features * 0.2),
            n_repeated=int(n_features * 0.1),
            n_classes=n_classes,
            n_clusters_per_class=n_clusters_per_class,
            class_sep=class_sep,
            random_state=self.config.RANDOM_STATE
        )
        
        return X, y
    
    def generate_regression_data(self, n_samples: int = 1000, n_features: int = 20,
                               noise: float = 0.1, n_informative: int = None) -> Tuple[np.ndarray, np.ndarray]:
        if n_informative is None:
            n_informative = int(n_features * 0.7)
            
        X, y = make_regression(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=n_informative,
            noise=noise,
            random_state=self.config.RANDOM_STATE
        )
        
        return X, y
    
    def generate_clustering_data(self, n_samples: int = 1000, n_features: int = 2,
                               centers: int = 3, cluster_std: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
        X, y = make_blobs(
            n_samples=n_samples,
            centers=centers,
            n_features=n_features,
            cluster_std=cluster_std,
            random_state=self.config.RANDOM_STATE
        )
        
        return X, y
    
    def generate_anomaly_data(self, n_samples: int = 1000, n_features: int = 20,
                            contamination: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
        from sklearn.datasets import make_blobs
        
        n_outliers = int(n_samples * contamination)
        n_inliers = n_samples - n_outliers
        
        X_inliers, _ = make_blobs(
            n_samples=n_inliers,
            centers=1,
            n_features=n_features,
            cluster_std=1.0,
            random_state=self.config.RANDOM_STATE
        )
        
        X_outliers = np.random.uniform(
            low=X_inliers.min() - 2,
            high=X_inliers.max() + 2,
            size=(n_outliers, n_features)
        )
        
        X = np.vstack([X_inliers, X_outliers])
        y = np.hstack([np.zeros(n_inliers), np.ones(n_outliers)])
        
        indices = np.random.permutation(len(X))
        return X[indices], y[indices]
    
    def generate_time_series_data(self, n_samples: int = 1000, n_features: int = 5,
                                trend: bool = True, seasonal: bool = True,
                                noise_level: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
        t = np.arange(n_samples)
        
        X = np.random.randn(n_samples, n_features)
        
        y = np.zeros(n_samples)
        
        if trend:
            y += 0.01 * t
        
        if seasonal:
            y += 0.5 * np.sin(2 * np.pi * t / 50)
            y += 0.3 * np.sin(2 * np.pi * t / 100)
        
        y += X[:, 0] * 0.5
        y += X[:, 1] * 0.3
        y += X[:, 2] * 0.2
        
        y += np.random.normal(0, noise_level, n_samples)
        
        return X, y
    
    def create_dataset(self, task_type: str = 'classification', **kwargs) -> pd.DataFrame:
        if task_type == 'classification':
            X, y = self.generate_classification_data(**kwargs)
        elif task_type == 'regression':
            X, y = self.generate_regression_data(**kwargs)
        elif task_type == 'clustering':
            X, y = self.generate_clustering_data(**kwargs)
        elif task_type == 'anomaly':
            X, y = self.generate_anomaly_data(**kwargs)
        elif task_type == 'time_series':
            X, y = self.generate_time_series_data(**kwargs)
        else:
            raise ValueError(f"Unknown task type: {task_type}")
        
        feature_names = [f'feature_{i}' for i in range(X.shape[1])]
        df = pd.DataFrame(X, columns=feature_names)
        df['target'] = y
        
        return df
    
    def add_noise(self, df: pd.DataFrame, noise_level: float = 0.1) -> pd.DataFrame:
        df_noisy = df.copy()
        
        for col in df.columns:
            if col != 'target':
                noise = np.random.normal(0, noise_level * df[col].std(), len(df))
                df_noisy[col] += noise
        
        return df_noisy
    
    def add_missing_values(self, df: pd.DataFrame, missing_ratio: float = 0.1) -> pd.DataFrame:
        df_missing = df.copy()
        
        n_missing = int(len(df) * missing_ratio)
        missing_indices = np.random.choice(len(df), n_missing, replace=False)
        
        for idx in missing_indices:
            col = np.random.choice([c for c in df.columns if c != 'target'])
            df_missing.loc[idx, col] = np.nan
        
        return df_missing
    
    def create_imbalanced_dataset(self, df: pd.DataFrame, imbalance_ratio: float = 0.1) -> pd.DataFrame:
        if 'target' not in df.columns:
            return df
        
        df_imbalanced = df.copy()
        
        unique_classes = df['target'].unique()
        if len(unique_classes) != 2:
            return df
        
        class_0 = df[df['target'] == unique_classes[0]]
        class_1 = df[df['target'] == unique_classes[1]]
        
        n_class_0 = len(class_0)
        n_class_1 = int(n_class_0 * imbalance_ratio)
        
        class_1_sampled = class_1.sample(n=n_class_1, random_state=self.config.RANDOM_STATE)
        
        df_imbalanced = pd.concat([class_0, class_1_sampled], ignore_index=True)
        df_imbalanced = df_imbalanced.sample(frac=1, random_state=self.config.RANDOM_STATE).reset_index(drop=True)
        
        return df_imbalanced
