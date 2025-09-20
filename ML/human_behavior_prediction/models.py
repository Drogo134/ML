import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import optuna
import joblib
import os
from config import Config

class ModelTrainer:
    def __init__(self, config):
        self.config = config
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        
    def prepare_data(self, X, y, test_size=0.2, validation_size=0.2):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.config.RANDOM_STATE, stratify=y
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=validation_size, random_state=self.config.RANDOM_STATE, stratify=y_train
        )
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)
        
        self.scalers['standard'] = scaler
        
        return (X_train_scaled, X_val_scaled, X_test_scaled, 
                X_train, X_val, X_test, y_train, y_val, y_test)
    
    def train_xgboost(self, X_train, y_train, X_val, y_val, task_type='classification'):
        if task_type == 'classification':
            model = xgb.XGBClassifier(**self.config.MODEL_PARAMS['xgboost'])
        else:
            model = xgb.XGBRegressor(**self.config.MODEL_PARAMS['xgboost'])
            
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=50,
            verbose=False
        )
        
        self.models['xgboost'] = model
        self.feature_importance['xgboost'] = model.feature_importances_
        return model
    
    def train_lightgbm(self, X_train, y_train, X_val, y_val, task_type='classification'):
        if task_type == 'classification':
            model = lgb.LGBMClassifier(**self.config.MODEL_PARAMS['lightgbm'])
        else:
            model = lgb.LGBMRegressor(**self.config.MODEL_PARAMS['lightgbm'])
            
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
        )
        
        self.models['lightgbm'] = model
        self.feature_importance['lightgbm'] = model.feature_importances_
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
            EarlyStopping(patience=20, restore_best_weights=True),
            ReduceLROnPlateau(factor=0.5, patience=10)
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
    
    def train_ensemble(self, X_train, y_train, X_val, y_val, task_type='classification'):
        models = {}
        
        if task_type == 'classification':
            models['rf'] = RandomForestClassifier(n_estimators=100, random_state=self.config.RANDOM_STATE)
            models['gb'] = GradientBoostingClassifier(n_estimators=100, random_state=self.config.RANDOM_STATE)
            models['lr'] = LogisticRegression(random_state=self.config.RANDOM_STATE, max_iter=1000)
            models['svm'] = SVC(probability=True, random_state=self.config.RANDOM_STATE)
        else:
            from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
            from sklearn.linear_model import LinearRegression
            from sklearn.svm import SVR
            
            models['rf'] = RandomForestRegressor(n_estimators=100, random_state=self.config.RANDOM_STATE)
            models['gb'] = GradientBoostingRegressor(n_estimators=100, random_state=self.config.RANDOM_STATE)
            models['lr'] = LinearRegression()
            models['svm'] = SVR()
        
        for name, model in models.items():
            model.fit(X_train, y_train)
            self.models[f'ensemble_{name}'] = model
            
        return models
    
    def optimize_hyperparameters(self, X_train, y_train, X_val, y_val, model_name='xgboost', n_trials=100):
        def objective(trial):
            if model_name == 'xgboost':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    'random_state': self.config.RANDOM_STATE
                }
                model = xgb.XGBClassifier(**params)
            elif model_name == 'lightgbm':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    'random_state': self.config.RANDOM_STATE
                }
                model = lgb.LGBMClassifier(**params)
            
            model.fit(X_train, y_train)
            score = model.score(X_val, y_val)
            return score
        
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials)
        
        best_params = study.best_params
        best_params['random_state'] = self.config.RANDOM_STATE
        
        if model_name == 'xgboost':
            best_model = xgb.XGBClassifier(**best_params)
        elif model_name == 'lightgbm':
            best_model = lgb.LGBMClassifier(**best_params)
        
        best_model.fit(X_train, y_train)
        self.models[f'{model_name}_optimized'] = best_model
        
        return best_model, best_params
    
    def evaluate_model(self, model, X_test, y_test, task_type='classification'):
        y_pred = model.predict(X_test)
        
        if task_type == 'classification':
            y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
            
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'auc': roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else None,
                'classification_report': classification_report(y_test, y_pred),
                'confusion_matrix': confusion_matrix(y_test, y_pred)
            }
        else:
            from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
            metrics = {
                'mse': mean_squared_error(y_test, y_pred),
                'mae': mean_absolute_error(y_test, y_pred),
                'r2': r2_score(y_test, y_pred)
            }
        
        return metrics
    
    def save_models(self, save_dir=None):
        if save_dir is None:
            save_dir = self.config.MODELS_DIR
        
        os.makedirs(save_dir, exist_ok=True)
        
        for name, model in self.models.items():
            if name == 'neural_network':
                model.save(os.path.join(save_dir, f'{name}.h5'))
            else:
                joblib.dump(model, os.path.join(save_dir, f'{name}.pkl'))
        
        for name, scaler in self.scalers.items():
            joblib.dump(scaler, os.path.join(save_dir, f'scaler_{name}.pkl'))
        
        if self.feature_importance:
            joblib.dump(self.feature_importance, os.path.join(save_dir, 'feature_importance.pkl'))
    
    def load_models(self, load_dir=None):
        if load_dir is None:
            load_dir = self.config.MODELS_DIR
        
        for file in os.listdir(load_dir):
            if file.endswith('.pkl') and not file.startswith('scaler_'):
                name = file.replace('.pkl', '')
                self.models[name] = joblib.load(os.path.join(load_dir, file))
            elif file.startswith('scaler_'):
                name = file.replace('scaler_', '').replace('.pkl', '')
                self.scalers[name] = joblib.load(os.path.join(load_dir, file))
            elif file == 'feature_importance.pkl':
                self.feature_importance = joblib.load(os.path.join(load_dir, file))
        
        if 'neural_network.h5' in os.listdir(load_dir):
            self.models['neural_network'] = tf.keras.models.load_model(
                os.path.join(load_dir, 'neural_network.h5')
            )
