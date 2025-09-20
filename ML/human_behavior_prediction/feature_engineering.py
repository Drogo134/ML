import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from feature_engine.encoding import RareLabelEncoder, MeanEncoder
from feature_engine.outliers import Winsorizer
from feature_engine.selection import DropCorrelatedFeatures
import warnings
warnings.filterwarnings('ignore')

class FeatureEngineer:
    def __init__(self, config):
        self.config = config
        self.scalers = {}
        self.encoders = {}
        self.feature_selector = None
        self.pca = None
        self.outlier_treaters = {}
        
    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        df['age_income_interaction'] = df['age'] * df['income'] / 1000
        df['session_click_interaction'] = df['session_duration'] * df['click_rate']
        df['page_views_per_minute'] = df['page_views'] / (df['session_duration'] / 60 + 1)
        df['engagement_efficiency'] = df['click_rate'] / (df['bounce_rate'] + 0.001)
        
        return df
    
    def create_polynomial_features(self, df: pd.DataFrame, columns: list, degree: int = 2) -> pd.DataFrame:
        df = df.copy()
        
        for col in columns:
            if col in df.columns:
                for d in range(2, degree + 1):
                    df[f'{col}_poly_{d}'] = df[col] ** d
                    
        return df
    
    def create_rolling_features(self, df: pd.DataFrame, group_cols: list, value_cols: list, windows: list) -> pd.DataFrame:
        df = df.copy()
        df_sorted = df.sort_values(group_cols + ['hour'])
        
        for group_col in group_cols:
            for value_col in value_cols:
                for window in windows:
                    df_sorted[f'{value_col}_rolling_mean_{window}'] = (
                        df_sorted.groupby(group_col)[value_col]
                        .rolling(window=window, min_periods=1)
                        .mean()
                        .reset_index(0, drop=True)
                    )
                    
        return df_sorted
    
    def create_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        return df
    
    def create_clustering_features(self, df: pd.DataFrame, n_clusters: int = 5) -> pd.DataFrame:
        df = df.copy()
        
        behavioral_cols = ['session_duration', 'page_views', 'click_rate', 'time_on_site']
        behavioral_data = df[behavioral_cols].fillna(df[behavioral_cols].mean())
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=self.config.RANDOM_STATE)
        df['behavioral_cluster'] = kmeans.fit_predict(behavioral_data)
        
        demographic_cols = ['age', 'income']
        demographic_data = df[demographic_cols].fillna(df[demographic_cols].mean())
        
        kmeans_demo = KMeans(n_clusters=n_clusters, random_state=self.config.RANDOM_STATE)
        df['demographic_cluster'] = kmeans_demo.fit_predict(demographic_data)
        
        return df
    
    def handle_outliers(self, df: pd.DataFrame, method: str = 'winsorize') -> pd.DataFrame:
        df = df.copy()
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if col not in ['will_purchase', 'will_churn', 'engagement_level', 'lifetime_value']:
                if method == 'winsorize':
                    winsorizer = Winsorizer(capping_method='iqr', tail='both')
                    df[col] = winsorizer.fit_transform(df[[col]]).flatten()
                elif method == 'clip':
                    q1, q3 = df[col].quantile([0.25, 0.75])
                    iqr = q3 - q1
                    lower_bound = q1 - 1.5 * iqr
                    upper_bound = q3 + 1.5 * iqr
                    df[col] = df[col].clip(lower_bound, upper_bound)
                    
        return df
    
    def encode_categorical_features(self, df: pd.DataFrame, target_col: str = None) -> pd.DataFrame:
        df = df.copy()
        categorical_cols = df.select_dtypes(include=['object']).columns
        
        for col in categorical_cols:
            if col in df.columns:
                if target_col and col != target_col:
                    encoder = MeanEncoder()
                    df[col] = encoder.fit_transform(df[[col]], df[target_col])
                else:
                    le = LabelEncoder()
                    df[col] = le.fit_transform(df[col].astype(str))
                    self.encoders[col] = le
                    
        return df
    
    def scale_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        df = df.copy()
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        target_cols = ['will_purchase', 'will_churn', 'engagement_level', 'lifetime_value']
        feature_cols = [col for col in numeric_cols if col not in target_cols]
        
        if fit:
            scaler = StandardScaler()
            df[feature_cols] = scaler.fit_transform(df[feature_cols])
            self.scalers['standard'] = scaler
        else:
            if 'standard' in self.scalers:
                df[feature_cols] = self.scalers['standard'].transform(df[feature_cols])
                
        return df
    
    def select_features(self, X: pd.DataFrame, y: pd.Series, method: str = 'mutual_info', k: int = 20) -> pd.DataFrame:
        if method == 'mutual_info':
            selector = SelectKBest(score_func=mutual_info_classif, k=k)
        elif method == 'f_classif':
            selector = SelectKBest(score_func=f_classif, k=k)
        else:
            return X
            
        X_selected = selector.fit_transform(X, y)
        selected_features = X.columns[selector.get_support()]
        
        self.feature_selector = selector
        return pd.DataFrame(X_selected, columns=selected_features, index=X.index)
    
    def apply_pca(self, X: pd.DataFrame, n_components: int = 10, fit: bool = True) -> pd.DataFrame:
        if fit:
            self.pca = PCA(n_components=n_components, random_state=self.config.RANDOM_STATE)
            X_pca = self.pca.fit_transform(X)
        else:
            if self.pca:
                X_pca = self.pca.transform(X)
            else:
                return X
                
        pca_columns = [f'pca_{i}' for i in range(X_pca.shape[1])]
        return pd.DataFrame(X_pca, columns=pca_columns, index=X.index)
    
    def drop_correlated_features(self, df: pd.DataFrame, threshold: float = 0.95) -> pd.DataFrame:
        df = df.copy()
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        target_cols = ['will_purchase', 'will_churn', 'engagement_level', 'lifetime_value']
        feature_cols = [col for col in numeric_cols if col not in target_cols]
        
        corr_selector = DropCorrelatedFeatures(variables=feature_cols, threshold=threshold)
        df_processed = corr_selector.fit_transform(df)
        
        return df_processed
    
    def engineer_features(self, df: pd.DataFrame, target_col: str = None, fit: bool = True) -> pd.DataFrame:
        df = df.copy()
        
        df = self.create_interaction_features(df)
        df = self.create_time_features(df)
        df = self.create_polynomial_features(df, ['age', 'income', 'session_duration'], degree=2)
        df = self.create_clustering_features(df)
        df = self.handle_outliers(df)
        df = self.encode_categorical_features(df, target_col)
        df = self.scale_features(df, fit=fit)
        
        if target_col and fit:
            df = self.drop_correlated_features(df)
            
        return df
