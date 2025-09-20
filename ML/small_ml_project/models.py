import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import lightgbm as lgb
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import optuna
import joblib
import os
from typing import Dict, List, Tuple, Any
from config import Config

class ModelTrainer:
    def __init__(self, config):
        self.config = config
        self.models = {}
        self.scalers = {}
        self.results = {}
        
    def prepare_data(self, X, y, test_size=0.2, validation_size=0.2, task_type='classification'):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.config.RANDOM_STATE, 
            stratify=y if task_type == 'classification' else None
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=validation_size, random_state=self.config.RANDOM_STATE,
            stratify=y_train if task_type == 'classification' else None
        )
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def train_random_forest(self, X_train, y_train, task_type='classification'):
        if task_type == 'classification':
            model = RandomForestClassifier(**self.config.MODEL_PARAMS['random_forest'])
        else:
            model = RandomForestRegressor(**self.config.MODEL_PARAMS['random_forest'])
        
        model.fit(X_train, y_train)
        self.models['random_forest'] = model
        return model
    
    def train_xgboost(self, X_train, y_train, task_type='classification'):
        if task_type == 'classification':
            model = xgb.XGBClassifier(**self.config.MODEL_PARAMS['xgboost'])
        else:
            model = xgb.XGBRegressor(**self.config.MODEL_PARAMS['xgboost'])
        
        model.fit(X_train, y_train)
        self.models['xgboost'] = model
        return model
    
    def train_lightgbm(self, X_train, y_train, task_type='classification'):
        if task_type == 'classification':
            model = lgb.LGBMClassifier(**self.config.MODEL_PARAMS['lightgbm'])
        else:
            model = lgb.LGBMRegressor(**self.config.MODEL_PARAMS['lightgbm'])
        
        model.fit(X_train, y_train)
        self.models['lightgbm'] = model
        return model
    
    def train_neural_network(self, X_train, y_train, X_val, y_val, task_type='classification'):
        model = Sequential()
        
        for i, units in enumerate(self.config.MODEL_PARAMS['neural_network']['hidden_layers']):
            if i == 0:
                model.add(Dense(units, activation='relu', input_shape=(X_train.shape[1],)))
            else:
                model.add(Dense(units, activation='relu'))
            model.add(BatchNormalization())
            model.add(Dropout(self.config.MODEL_PARAMS['neural_network']['dropout_rate']))
        
        if task_type == 'classification':
            model.add(Dense(1, activation='sigmoid'))
            model.compile(
                optimizer=Adam(learning_rate=self.config.MODEL_PARAMS['neural_network']['learning_rate']),
                loss='binary_crossentropy',
                metrics=['accuracy']
            )
        else:
            model.add(Dense(1, activation='linear'))
            model.compile(
                optimizer=Adam(learning_rate=self.config.MODEL_PARAMS['neural_network']['learning_rate']),
                loss='mse',
                metrics=['mae']
            )
        
        callbacks = [
            EarlyStopping(patience=10, restore_best_weights=True),
            ReduceLROnPlateau(factor=0.5, patience=5)
        ]
        
        model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=self.config.MODEL_PARAMS['neural_network']['epochs'],
            batch_size=self.config.MODEL_PARAMS['neural_network']['batch_size'],
            callbacks=callbacks,
            verbose=0
        )
        
        self.models['neural_network'] = model
        return model
    
    def train_svm(self, X_train, y_train, task_type='classification'):
        if task_type == 'classification':
            model = SVC(probability=True, random_state=self.config.RANDOM_STATE)
        else:
            model = SVR()
        
        model.fit(X_train, y_train)
        self.models['svm'] = model
        return model
    
    def train_logistic_regression(self, X_train, y_train, task_type='classification'):
        if task_type == 'classification':
            model = LogisticRegression(random_state=self.config.RANDOM_STATE, max_iter=1000)
        else:
            model = LinearRegression()
        
        model.fit(X_train, y_train)
        self.models['logistic_regression'] = model
        return model
    
    def optimize_hyperparameters(self, X_train, y_train, model_name='xgboost', 
                                task_type='classification', n_trials=50):
        def objective(trial):
            if model_name == 'xgboost':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    'random_state': self.config.RANDOM_STATE
                }
                if task_type == 'classification':
                    model = xgb.XGBClassifier(**params)
                else:
                    model = xgb.XGBRegressor(**params)
            elif model_name == 'lightgbm':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    'random_state': self.config.RANDOM_STATE
                }
                if task_type == 'classification':
                    model = lgb.LGBMClassifier(**params)
                else:
                    model = lgb.LGBMRegressor(**params)
            
            scores = cross_val_score(model, X_train, y_train, cv=3, scoring='accuracy' if task_type == 'classification' else 'r2')
            return scores.mean()
        
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials)
        
        best_params = study.best_params
        best_params['random_state'] = self.config.RANDOM_STATE
        
        if model_name == 'xgboost':
            if task_type == 'classification':
                best_model = xgb.XGBClassifier(**best_params)
            else:
                best_model = xgb.XGBRegressor(**best_params)
        elif model_name == 'lightgbm':
            if task_type == 'classification':
                best_model = lgb.LGBMClassifier(**best_params)
            else:
                best_model = lgb.LGBMRegressor(**best_params)
        
        best_model.fit(X_train, y_train)
        self.models[f'{model_name}_optimized'] = best_model
        
        return best_model, best_params
    
    def evaluate_model(self, model, X_test, y_test, task_type='classification'):
        y_pred = model.predict(X_test)
        
        if task_type == 'classification':
            y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
            
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, average='weighted'),
                'recall': recall_score(y_test, y_pred, average='weighted'),
                'f1': f1_score(y_test, y_pred, average='weighted'),
                'auc': roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else None
            }
        else:
            metrics = {
                'mse': mean_squared_error(y_test, y_pred),
                'mae': mean_absolute_error(y_test, y_pred),
                'r2': r2_score(y_test, y_pred)
            }
        
        return metrics
    
    def train_all_models(self, X_train, y_train, X_val, y_val, task_type='classification'):
        print("Training all models...")
        
        self.train_random_forest(X_train, y_train, task_type)
        self.train_xgboost(X_train, y_train, task_type)
        self.train_lightgbm(X_train, y_train, task_type)
        self.train_neural_network(X_train, y_train, X_val, y_val, task_type)
        self.train_svm(X_train, y_train, task_type)
        self.train_logistic_regression(X_train, y_train, task_type)
        
        print(f"Trained {len(self.models)} models")
    
    def evaluate_all_models(self, X_test, y_test, task_type='classification'):
        print("Evaluating all models...")
        
        for name, model in self.models.items():
            metrics = self.evaluate_model(model, X_test, y_test, task_type)
            self.results[name] = metrics
            
            if task_type == 'classification':
                print(f"{name}: Accuracy={metrics['accuracy']:.4f}, AUC={metrics.get('auc', 'N/A')}")
            else:
                print(f"{name}: MSE={metrics['mse']:.4f}, R²={metrics['r2']:.4f}")
        
        return self.results
    
    def save_models(self, save_dir=None):
        if save_dir is None:
            save_dir = self.config.MODELS_DIR
        
        os.makedirs(save_dir, exist_ok=True)
        
        for name, model in self.models.items():
            if name == 'neural_network':
                model.save(os.path.join(save_dir, f'{name}.h5'))
            else:
                joblib.dump(model, os.path.join(save_dir, f'{name}.pkl'))
        
        print(f"Models saved to {save_dir}")
    
    def load_models(self, load_dir=None):
        if load_dir is None:
            load_dir = self.config.MODELS_DIR
        
        for file in os.listdir(load_dir):
            if file.endswith('.pkl'):
                name = file.replace('.pkl', '')
                self.models[name] = joblib.load(os.path.join(load_dir, file))
            elif file.endswith('.h5'):
                name = file.replace('.h5', '')
                self.models[name] = tf.keras.models.load_model(os.path.join(load_dir, file))
        
        print(f"Loaded {len(self.models)} models from {load_dir}")
