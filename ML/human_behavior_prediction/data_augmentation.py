import numpy as np
import pandas as pd
from sklearn.utils import resample
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
from imblearn.under_sampling import RandomUnderSampler, TomekLinks
from imblearn.combine import SMOTETomek, SMOTEENN
import warnings
warnings.filterwarnings('ignore')

class DataAugmentation:
    def __init__(self, random_state=42):
        self.random_state = random_state
        np.random.seed(random_state)
        
    def smote_oversampling(self, X, y, sampling_strategy='auto', k_neighbors=5):
        smote = SMOTE(
            sampling_strategy=sampling_strategy,
            k_neighbors=k_neighbors,
            random_state=self.random_state
        )
        X_resampled, y_resampled = smote.fit_resample(X, y)
        return X_resampled, y_resampled
    
    def adasyn_oversampling(self, X, y, sampling_strategy='auto', n_neighbors=5):
        adasyn = ADASYN(
            sampling_strategy=sampling_strategy,
            n_neighbors=n_neighbors,
            random_state=self.random_state
        )
        X_resampled, y_resampled = adasyn.fit_resample(X, y)
        return X_resampled, y_resampled
    
    def borderline_smote(self, X, y, sampling_strategy='auto', k_neighbors=5):
        borderline_smote = BorderlineSMOTE(
            sampling_strategy=sampling_strategy,
            k_neighbors=k_neighbors,
            random_state=self.random_state
        )
        X_resampled, y_resampled = borderline_smote.fit_resample(X, y)
        return X_resampled, y_resampled
    
    def random_undersampling(self, X, y, sampling_strategy='auto'):
        undersampler = RandomUnderSampler(
            sampling_strategy=sampling_strategy,
            random_state=self.random_state
        )
        X_resampled, y_resampled = undersampler.fit_resample(X, y)
        return X_resampled, y_resampled
    
    def tomek_links(self, X, y):
        tomek = TomekLinks()
        X_resampled, y_resampled = tomek.fit_resample(X, y)
        return X_resampled, y_resampled
    
    def smote_tomek(self, X, y, sampling_strategy='auto'):
        smote_tomek = SMOTETomek(
            sampling_strategy=sampling_strategy,
            random_state=self.random_state
        )
        X_resampled, y_resampled = smote_tomek.fit_resample(X, y)
        return X_resampled, y_resampled
    
    def smote_enn(self, X, y, sampling_strategy='auto'):
        smote_enn = SMOTEENN(
            sampling_strategy=sampling_strategy,
            random_state=self.random_state
        )
        X_resampled, y_resampled = smote_enn.fit_resample(X, y)
        return X_resampled, y_resampled
    
    def add_noise(self, X, noise_factor=0.01):
        noise = np.random.normal(0, noise_factor, X.shape)
        return X + noise
    
    def gaussian_noise_augmentation(self, X, y, noise_factor=0.01, n_samples=1000):
        X_augmented = []
        y_augmented = []
        
        for _ in range(n_samples):
            idx = np.random.randint(0, len(X))
            x_noisy = X[idx] + np.random.normal(0, noise_factor, X[idx].shape)
            X_augmented.append(x_noisy)
            y_augmented.append(y[idx])
        
        X_augmented = np.array(X_augmented)
        y_augmented = np.array(y_augmented)
        
        return np.vstack([X, X_augmented]), np.hstack([y, y_augmented])
    
    def bootstrap_sampling(self, X, y, n_samples=None):
        if n_samples is None:
            n_samples = len(X)
        
        indices = np.random.choice(len(X), size=n_samples, replace=True)
        return X[indices], y[indices]
    
    def create_synthetic_samples(self, X, y, n_synthetic=1000, method='interpolation'):
        X_synthetic = []
        y_synthetic = []
        
        for _ in range(n_synthetic):
            if method == 'interpolation':
                idx1, idx2 = np.random.choice(len(X), 2, replace=False)
                alpha = np.random.random()
                x_synthetic = alpha * X[idx1] + (1 - alpha) * X[idx2]
                y_synthetic.append(y[idx1] if np.random.random() < alpha else y[idx2])
            elif method == 'gaussian':
                idx = np.random.randint(0, len(X))
                x_synthetic = X[idx] + np.random.normal(0, 0.1, X[idx].shape)
                y_synthetic.append(y[idx])
            
            X_synthetic.append(x_synthetic)
        
        X_synthetic = np.array(X_synthetic)
        y_synthetic = np.array(y_synthetic)
        
        return np.vstack([X, X_synthetic]), np.hstack([y, y_synthetic])
    
    def augment_dataset(self, X, y, method='smote', **kwargs):
        if method == 'smote':
            return self.smote_oversampling(X, y, **kwargs)
        elif method == 'adasyn':
            return self.adasyn_oversampling(X, y, **kwargs)
        elif method == 'borderline_smote':
            return self.borderline_smote(X, y, **kwargs)
        elif method == 'random_undersampling':
            return self.random_undersampling(X, y, **kwargs)
        elif method == 'tomek_links':
            return self.tomek_links(X, y)
        elif method == 'smote_tomek':
            return self.smote_tomek(X, y, **kwargs)
        elif method == 'smote_enn':
            return self.smote_enn(X, y, **kwargs)
        elif method == 'gaussian_noise':
            return self.gaussian_noise_augmentation(X, y, **kwargs)
        elif method == 'bootstrap':
            return self.bootstrap_sampling(X, y, **kwargs)
        elif method == 'synthetic':
            return self.create_synthetic_samples(X, y, **kwargs)
        else:
            raise ValueError(f"Unknown augmentation method: {method}")
    
    def compare_augmentation_methods(self, X, y, methods=['smote', 'adasyn', 'borderline_smote']):
        results = {}
        
        for method in methods:
            try:
                X_aug, y_aug = self.augment_dataset(X, y, method=method)
                results[method] = {
                    'original_size': len(X),
                    'augmented_size': len(X_aug),
                    'class_distribution': np.bincount(y_aug),
                    'X': X_aug,
                    'y': y_aug
                }
            except Exception as e:
                print(f"Error with {method}: {e}")
                results[method] = None
        
        return results
