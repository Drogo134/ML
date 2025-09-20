import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random
from typing import Dict, List, Tuple
from config import Config

class HumanBehaviorDataGenerator:
    def __init__(self, random_state: int = 42):
        np.random.seed(random_state)
        random.seed(random_state)
        
    def generate_demographic_features(self, n_samples: int) -> Dict[str, List]:
        return {
            'age': np.random.normal(35, 12, n_samples).astype(int),
            'gender': np.random.choice(['Male', 'Female', 'Other'], n_samples, p=[0.48, 0.48, 0.04]),
            'education': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], n_samples, p=[0.3, 0.4, 0.25, 0.05]),
            'income': np.random.lognormal(10, 0.5, n_samples).astype(int),
            'location': np.random.choice(['Urban', 'Suburban', 'Rural'], n_samples, p=[0.6, 0.3, 0.1])
        }
    
    def generate_behavioral_features(self, n_samples: int) -> Dict[str, List]:
        return {
            'session_duration': np.random.exponential(300, n_samples),
            'page_views': np.random.poisson(8, n_samples),
            'click_rate': np.random.beta(2, 5, n_samples),
            'time_on_site': np.random.gamma(2, 100, n_samples),
            'bounce_rate': np.random.beta(1, 3, n_samples),
            'conversion_rate': np.random.beta(1, 9, n_samples)
        }
    
    def generate_temporal_features(self, n_samples: int) -> Dict[str, List]:
        start_date = datetime.now() - timedelta(days=365)
        dates = [start_date + timedelta(days=np.random.randint(0, 365)) for _ in range(n_samples)]
        
        return {
            'hour': [d.hour for d in dates],
            'day_of_week': [d.weekday() for d in dates],
            'month': [d.month for d in dates],
            'season': [self._get_season(d.month) for d in dates],
            'is_weekend': [d.weekday() >= 5 for d in dates]
        }
    
    def generate_contextual_features(self, n_samples: int) -> Dict[str, List]:
        return {
            'device_type': np.random.choice(['Desktop', 'Mobile', 'Tablet'], n_samples, p=[0.4, 0.5, 0.1]),
            'browser': np.random.choice(['Chrome', 'Firefox', 'Safari', 'Edge'], n_samples, p=[0.6, 0.2, 0.15, 0.05]),
            'referrer': np.random.choice(['Direct', 'Search', 'Social', 'Email', 'Other'], n_samples, p=[0.3, 0.3, 0.2, 0.1, 0.1]),
            'campaign': np.random.choice(['None', 'Summer', 'Holiday', 'BlackFriday'], n_samples, p=[0.6, 0.15, 0.15, 0.1])
        }
    
    def generate_psychological_features(self, n_samples: int) -> Dict[str, List]:
        return {
            'risk_tolerance': np.random.beta(2, 2, n_samples),
            'impulsiveness': np.random.beta(2, 3, n_samples),
            'patience': np.random.beta(3, 2, n_samples),
            'curiosity': np.random.beta(2.5, 2, n_samples),
            'social_orientation': np.random.beta(2, 2.5, n_samples)
        }
    
    def generate_target_variables(self, features: pd.DataFrame) -> Dict[str, List]:
        n_samples = len(features)
        
        purchase_probability = self._calculate_purchase_probability(features)
        churn_probability = self._calculate_churn_probability(features)
        engagement_score = self._calculate_engagement_score(features)
        
        return {
            'will_purchase': np.random.binomial(1, purchase_probability, n_samples),
            'will_churn': np.random.binomial(1, churn_probability, n_samples),
            'engagement_level': np.random.normal(engagement_score, 0.1, n_samples),
            'lifetime_value': np.random.exponential(engagement_score * 100, n_samples)
        }
    
    def _get_season(self, month: int) -> str:
        if month in [12, 1, 2]:
            return 'Winter'
        elif month in [3, 4, 5]:
            return 'Spring'
        elif month in [6, 7, 8]:
            return 'Summer'
        else:
            return 'Fall'
    
    def _calculate_purchase_probability(self, features: pd.DataFrame) -> np.ndarray:
        base_prob = 0.1
        age_factor = np.where(features['age'] > 30, 0.05, -0.02)
        income_factor = features['income'] / 100000 * 0.1
        session_factor = features['session_duration'] / 1000 * 0.05
        return np.clip(base_prob + age_factor + income_factor + session_factor, 0, 1)
    
    def _calculate_churn_probability(self, features: pd.DataFrame) -> np.ndarray:
        base_prob = 0.15
        session_factor = np.where(features['session_duration'] < 60, 0.1, -0.05)
        bounce_factor = features['bounce_rate'] * 0.2
        return np.clip(base_prob + session_factor + bounce_factor, 0, 1)
    
    def _calculate_engagement_score(self, features: pd.DataFrame) -> np.ndarray:
        base_score = 0.5
        page_views_factor = features['page_views'] / 20
        time_factor = features['time_on_site'] / 1000
        click_factor = features['click_rate'] * 0.5
        return np.clip(base_score + page_views_factor + time_factor + click_factor, 0, 1)
    
    def generate_dataset(self, n_samples: int = 10000) -> pd.DataFrame:
        demographic = self.generate_demographic_features(n_samples)
        behavioral = self.generate_behavioral_features(n_samples)
        temporal = self.generate_temporal_features(n_samples)
        contextual = self.generate_contextual_features(n_samples)
        psychological = self.generate_psychological_features(n_samples)
        
        all_features = {**demographic, **behavioral, **temporal, **contextual, **psychological}
        features_df = pd.DataFrame(all_features)
        
        targets = self.generate_target_variables(features_df)
        targets_df = pd.DataFrame(targets)
        
        dataset = pd.concat([features_df, targets_df], axis=1)
        return dataset

if __name__ == "__main__":
    generator = HumanBehaviorDataGenerator()
    dataset = generator.generate_dataset(10000)
    print(f"Generated dataset with shape: {dataset.shape}")
    print(f"Columns: {list(dataset.columns)}")
    print(dataset.head())
