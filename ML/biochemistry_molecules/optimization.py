import numpy as np
import pandas as pd
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
import torch
import torch.nn as nn
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score, mean_squared_error, r2_score
from typing import Dict, List, Tuple, Any, Optional
import logging
import joblib
import os
from config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HyperparameterOptimizer:
    def __init__(self, config):
        self.config = config
        self.study = None
        self.best_params = {}
        self.optimization_history = []
        
    def optimize_xgboost(self, X, y, task_type='classification', n_trials=100):
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 1000),
                'max_depth': trial.suggest_int('max_depth', 3, 12),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'gamma': trial.suggest_float('gamma', 0, 5),
                'random_state': self.config.RANDOM_STATE
            }
            
            if task_type == 'classification':
                from xgboost import XGBClassifier
                model = XGBClassifier(**params)
                scoring = 'accuracy'
            else:
                from xgboost import XGBRegressor
                model = XGBRegressor(**params)
                scoring = 'r2'
            
            cv_scores = cross_val_score(model, X, y, cv=5, scoring=scoring)
            return cv_scores.mean()
        
        study = optuna.create_study(
            direction='maximize',
            sampler=TPESampler(seed=self.config.RANDOM_STATE),
            pruner=MedianPruner(n_startup_trials=10, n_warmup_steps=5)
        )
        
        study.optimize(objective, n_trials=n_trials)
        
        self.best_params['xgboost'] = study.best_params
        self.optimization_history.append({
            'model': 'xgboost',
            'best_score': study.best_value,
            'best_params': study.best_params,
            'n_trials': n_trials
        })
        
        logger.info(f"XGBoost optimization completed. Best score: {study.best_value:.4f}")
        return study.best_params
    
    def optimize_lightgbm(self, X, y, task_type='classification', n_trials=100):
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 1000),
                'max_depth': trial.suggest_int('max_depth', 3, 12),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
                'num_leaves': trial.suggest_int('num_leaves', 10, 300),
                'random_state': self.config.RANDOM_STATE
            }
            
            if task_type == 'classification':
                import lightgbm as lgb
                model = lgb.LGBMClassifier(**params)
                scoring = 'accuracy'
            else:
                import lightgbm as lgb
                model = lgb.LGBMRegressor(**params)
                scoring = 'r2'
            
            cv_scores = cross_val_score(model, X, y, cv=5, scoring=scoring)
            return cv_scores.mean()
        
        study = optuna.create_study(
            direction='maximize',
            sampler=TPESampler(seed=self.config.RANDOM_STATE),
            pruner=MedianPruner(n_startup_trials=10, n_warmup_steps=5)
        )
        
        study.optimize(objective, n_trials=n_trials)
        
        self.best_params['lightgbm'] = study.best_params
        self.optimization_history.append({
            'model': 'lightgbm',
            'best_score': study.best_value,
            'best_params': study.best_params,
            'n_trials': n_trials
        })
        
        logger.info(f"LightGBM optimization completed. Best score: {study.best_value:.4f}")
        return study.best_params
    
    def optimize_neural_network(self, X, y, task_type='classification', n_trials=50):
        def objective(trial):
            hidden_layers = trial.suggest_int('n_layers', 1, 4)
            layers = []
            
            for i in range(hidden_layers):
                units = trial.suggest_int(f'n_units_l{i}', 32, 256)
                layers.append(units)
            
            dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)
            learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
            batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128])
            
            model = self._build_neural_network(
                input_dim=X.shape[1],
                hidden_layers=layers,
                output_dim=1,
                dropout_rate=dropout_rate,
                learning_rate=learning_rate,
                task_type=task_type
            )
            
            cv_scores = self._evaluate_neural_network(model, X, y, task_type, batch_size)
            return cv_scores.mean()
        
        study = optuna.create_study(
            direction='maximize',
            sampler=TPESampler(seed=self.config.RANDOM_STATE),
            pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=3)
        )
        
        study.optimize(objective, n_trials=n_trials)
        
        self.best_params['neural_network'] = study.best_params
        self.optimization_history.append({
            'model': 'neural_network',
            'best_score': study.best_value,
            'best_params': study.best_params,
            'n_trials': n_trials
        })
        
        logger.info(f"Neural Network optimization completed. Best score: {study.best_value:.4f}")
        return study.best_params
    
    def _build_neural_network(self, input_dim, hidden_layers, output_dim, 
                            dropout_rate, learning_rate, task_type):
        import tensorflow as tf
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
        from tensorflow.keras.optimizers import Adam
        
        model = Sequential()
        
        for i, units in enumerate(hidden_layers):
            if i == 0:
                model.add(Dense(units, activation='relu', input_shape=(input_dim,)))
            else:
                model.add(Dense(units, activation='relu'))
            model.add(BatchNormalization())
            model.add(Dropout(dropout_rate))
        
        if task_type == 'classification':
            model.add(Dense(output_dim, activation='sigmoid'))
            model.compile(
                optimizer=Adam(learning_rate=learning_rate),
                loss='binary_crossentropy',
                metrics=['accuracy']
            )
        else:
            model.add(Dense(output_dim, activation='linear'))
            model.compile(
                optimizer=Adam(learning_rate=learning_rate),
                loss='mse',
                metrics=['mae']
            )
        
        return model
    
    def _evaluate_neural_network(self, model, X, y, task_type, batch_size):
        from sklearn.model_selection import KFold
        from sklearn.preprocessing import StandardScaler
        
        kf = KFold(n_splits=5, shuffle=True, random_state=self.config.RANDOM_STATE)
        scores = []
        
        for train_idx, val_idx in kf.split(X):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)
            
            model.fit(
                X_train_scaled, y_train,
                validation_data=(X_val_scaled, y_val),
                epochs=20,
                batch_size=batch_size,
                verbose=0
            )
            
            if task_type == 'classification':
                y_pred = (model.predict(X_val_scaled) > 0.5).astype(int)
                score = accuracy_score(y_val, y_pred)
            else:
                y_pred = model.predict(X_val_scaled)
                score = r2_score(y_val, y_pred)
            
            scores.append(score)
        
        return np.array(scores)
    
    def optimize_graph_models(self, train_loader, val_loader, model_type='gcn', n_trials=30):
        def objective(trial):
            if model_type == 'gcn':
                hidden_dim = trial.suggest_int('hidden_dim', 32, 128)
                num_layers = trial.suggest_int('num_layers', 2, 5)
                dropout = trial.suggest_float('dropout', 0.1, 0.5)
                learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
                
                from graph_models import GCNModel
                model = GCNModel(
                    input_dim=7,
                    hidden_dim=hidden_dim,
                    output_dim=1,
                    num_layers=num_layers,
                    dropout=dropout
                )
            
            elif model_type == 'gat':
                hidden_dim = trial.suggest_int('hidden_dim', 32, 128)
                num_heads = trial.suggest_int('num_heads', 4, 16)
                num_layers = trial.suggest_int('num_layers', 2, 5)
                dropout = trial.suggest_float('dropout', 0.1, 0.5)
                learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
                
                from graph_models import GATModel
                model = GATModel(
                    input_dim=7,
                    hidden_dim=hidden_dim,
                    output_dim=1,
                    num_heads=num_heads,
                    num_layers=num_layers,
                    dropout=dropout
                )
            
            else:
                raise ValueError(f"Unknown model type: {model_type}")
            
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model = model.to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
            
            best_val_score = 0
            for epoch in range(20):
                model.train()
                train_loss = 0
                
                for batch in train_loader:
                    batch = batch.to(device)
                    optimizer.zero_grad()
                    
                    out = model(batch.x, batch.edge_index, batch.batch)
                    loss = torch.nn.functional.binary_cross_entropy_with_logits(
                        out.squeeze(), batch.y.float()
                    )
                    
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item()
                
                model.eval()
                val_scores = []
                with torch.no_grad():
                    for batch in val_loader:
                        batch = batch.to(device)
                        out = model(batch.x, batch.edge_index, batch.batch)
                        pred = (torch.sigmoid(out.squeeze()) > 0.5).float()
                        score = (pred == batch.y.float()).float().mean()
                        val_scores.append(score.item())
                
                val_score = np.mean(val_scores)
                if val_score > best_val_score:
                    best_val_score = val_score
            
            return best_val_score
        
        study = optuna.create_study(
            direction='maximize',
            sampler=TPESampler(seed=self.config.RANDOM_STATE)
        )
        
        study.optimize(objective, n_trials=n_trials)
        
        self.best_params[f'graph_{model_type}'] = study.best_params
        self.optimization_history.append({
            'model': f'graph_{model_type}',
            'best_score': study.best_value,
            'best_params': study.best_params,
            'n_trials': n_trials
        })
        
        logger.info(f"Graph {model_type} optimization completed. Best score: {study.best_value:.4f}")
        return study.best_params
    
    def optimize_all_models(self, X, y, task_type='classification', 
                          graph_data=None, n_trials_per_model=50):
        logger.info("Starting optimization of all models...")
        
        results = {}
        
        if X is not None:
            results['xgboost'] = self.optimize_xgboost(X, y, task_type, n_trials_per_model)
            results['lightgbm'] = self.optimize_lightgbm(X, y, task_type, n_trials_per_model)
            results['neural_network'] = self.optimize_neural_network(X, y, task_type, n_trials_per_model)
        
        if graph_data is not None:
            train_loader, val_loader = graph_data
            results['graph_gcn'] = self.optimize_graph_models(train_loader, val_loader, 'gcn', n_trials_per_model)
            results['graph_gat'] = self.optimize_graph_models(train_loader, val_loader, 'gat', n_trials_per_model)
        
        self.save_optimization_results()
        return results
    
    def save_optimization_results(self):
        results_file = self.config.RESULTS_DIR / 'optimization_results.json'
        
        import json
        with open(results_file, 'w') as f:
            json.dump({
                'best_params': self.best_params,
                'optimization_history': self.optimization_history
            }, f, indent=2)
        
        logger.info(f"Optimization results saved to {results_file}")
    
    def load_optimization_results(self):
        results_file = self.config.RESULTS_DIR / 'optimization_results.json'
        
        if os.path.exists(results_file):
            import json
            with open(results_file, 'r') as f:
                data = json.load(f)
                self.best_params = data.get('best_params', {})
                self.optimization_history = data.get('optimization_history', [])
            
            logger.info("Optimization results loaded successfully")
        else:
            logger.warning("No optimization results found")
    
    def get_optimization_summary(self):
        if not self.optimization_history:
            return "No optimization history available"
        
        summary = "Optimization Summary:\n"
        summary += "=" * 50 + "\n"
        
        for record in self.optimization_history:
            summary += f"Model: {record['model']}\n"
            summary += f"Best Score: {record['best_score']:.4f}\n"
            summary += f"Trials: {record['n_trials']}\n"
            summary += f"Best Params: {record['best_params']}\n"
            summary += "-" * 30 + "\n"
        
        return summary
