import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, roc_auc_score, mean_squared_error, r2_score
import xgboost as xgb
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from typing import List, Dict, Tuple
import os

from config import Config
from molecular_data_loader import MolecularDataLoader
from graph_models import MolecularGraphTrainer

warnings.filterwarnings('ignore')

class MolecularPropertyPredictionPipeline:
    def __init__(self):
        self.config = Config()
        self.config.create_directories()
        
        self.data_loader = MolecularDataLoader(self.config)
        self.graph_trainer = MolecularGraphTrainer(self.config)
        
        self.data = None
        self.results = {}
        
    def load_and_process_data(self, dataset_name: str = 'tox21'):
        print(f"Loading and processing {dataset_name} dataset...")
        
        self.data = self.data_loader.create_molecular_dataset(dataset_name)
        
        print(f"Dataset shape: {self.data.shape}")
        print(f"Columns: {list(self.data.columns)}")
        
        return self.data
    
    def prepare_traditional_features(self, target_column: str) -> Tuple[np.ndarray, np.ndarray]:
        descriptor_cols = [col for col in self.data.columns if col in self.config.MOLECULAR_FEATURES['descriptors']]
        
        X = self.data[descriptor_cols].fillna(0).values
        y = self.data[target_column].fillna(0).values
        
        return X, y
    
    def prepare_fingerprint_features(self, target_column: str) -> Tuple[np.ndarray, np.ndarray]:
        fingerprint_cols = []
        for fp_name in self.config.MOLECULAR_FEATURES['fingerprints']:
            fp_cols = [col for col in self.data.columns if col.startswith(f'{fp_name}_')]
            fingerprint_cols.extend(fp_cols)
        
        X = self.data[fingerprint_cols].fillna(0).values
        y = self.data[target_column].fillna(0).values
        
        return X, y
    
    def prepare_graph_data(self, target_column: str) -> Tuple[List[Dict], np.ndarray]:
        graphs = self.data['graph_data'].tolist()
        y = self.data[target_column].fillna(0).values
        
        valid_indices = [i for i, graph in enumerate(graphs) if graph is not None]
        graphs = [graphs[i] for i in valid_indices]
        y = y[valid_indices]
        
        return graphs, y
    
    def train_traditional_models(self, X, y, task_type: str = 'classification'):
        print("Training traditional ML models...")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.config.TEST_SIZE, random_state=self.config.RANDOM_STATE, stratify=y if task_type == 'classification' else None
        )
        
        models = {}
        
        if task_type == 'classification':
            models['rf'] = RandomForestClassifier(n_estimators=100, random_state=self.config.RANDOM_STATE)
            models['xgboost'] = xgb.XGBClassifier(**self.config.MODEL_PARAMS['xgboost'])
            models['lightgbm'] = lgb.LGBMClassifier(**self.config.MODEL_PARAMS['xgboost'])
        else:
            models['rf'] = RandomForestRegressor(n_estimators=100, random_state=self.config.RANDOM_STATE)
            models['xgboost'] = xgb.XGBRegressor(**self.config.MODEL_PARAMS['xgboost'])
            models['lightgbm'] = lgb.LGBMRegressor(**self.config.MODEL_PARAMS['xgboost'])
        
        for name, model in models.items():
            print(f"Training {name}...")
            model.fit(X_train, y_train)
            
            y_pred = model.predict(X_test)
            
            if task_type == 'classification':
                y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
                accuracy = accuracy_score(y_test, y_pred)
                auc = roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else None
                
                self.results[f'traditional_{name}'] = {
                    'accuracy': accuracy,
                    'auc': auc,
                    'predictions': y_pred,
                    'probabilities': y_pred_proba,
                    'targets': y_test
                }
                
                print(f"{name} - Accuracy: {accuracy:.4f}, AUC: {auc:.4f if auc else 'N/A'}")
            else:
                mse = mean_squared_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                
                self.results[f'traditional_{name}'] = {
                    'mse': mse,
                    'r2': r2,
                    'predictions': y_pred,
                    'targets': y_test
                }
                
                print(f"{name} - MSE: {mse:.4f}, R²: {r2:.4f}")
        
        return X_test, y_test
    
    def train_graph_models(self, graphs, y, task_type: str = 'classification'):
        print("Training graph neural network models...")
        
        train_indices, test_indices = train_test_split(
            range(len(graphs)), test_size=self.config.TEST_SIZE, 
            random_state=self.config.RANDOM_STATE, stratify=y if task_type == 'classification' else None
        )
        
        train_graphs = [graphs[i] for i in train_indices]
        train_y = y[train_indices]
        test_graphs = [graphs[i] for i in test_indices]
        test_y = y[test_indices]
        
        train_loader = self.graph_trainer.create_data_loader(train_graphs, train_y, batch_size=32)
        test_loader = self.graph_trainer.create_data_loader(test_graphs, test_y, batch_size=32, shuffle=False)
        
        val_indices, _ = train_test_split(
            range(len(train_graphs)), test_size=self.config.VALIDATION_SIZE,
            random_state=self.config.RANDOM_STATE, stratify=train_y if task_type == 'classification' else None
        )
        
        val_graphs = [train_graphs[i] for i in val_indices]
        val_y = train_y[val_indices]
        val_loader = self.graph_trainer.create_data_loader(val_graphs, val_y, batch_size=32, shuffle=False)
        
        graph_models = ['gcn', 'gat', 'transformer']
        
        for model_name in graph_models:
            print(f"Training {model_name}...")
            self.graph_trainer.train_model(model_name, train_loader, val_loader, task_type)
            
            results = self.graph_trainer.evaluate_model(model_name, test_loader, task_type)
            self.results[f'graph_{model_name}'] = results
            
            if task_type == 'classification':
                print(f"{model_name} - Accuracy: {results['accuracy']:.4f}")
            else:
                print(f"{model_name} - MSE: {results['mse']:.4f}, R²: {results['r2']:.4f}")
        
        return test_loader
    
    def create_visualizations(self, target_column: str):
        print("Creating visualizations...")
        
        results_dir = self.config.RESULTS_DIR
        
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        
        model_names = list(self.results.keys())
        
        if 'accuracy' in self.results[model_names[0]]:
            accuracies = [self.results[name].get('accuracy', 0) for name in model_names]
            aucs = [self.results[name].get('auc', 0) for name in model_names]
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            ax1.bar(model_names, accuracies)
            ax1.set_title('Model Accuracy Comparison')
            ax1.set_ylabel('Accuracy')
            ax1.tick_params(axis='x', rotation=45)
            
            ax2.bar(model_names, aucs)
            ax2.set_title('Model AUC Comparison')
            ax2.set_ylabel('AUC')
            ax2.tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            plt.savefig(results_dir / f'model_comparison_{target_column}.png', dpi=300, bbox_inches='tight')
            plt.show()
        else:
            mses = [self.results[name].get('mse', 0) for name in model_names]
            r2s = [self.results[name].get('r2', 0) for name in model_names]
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            ax1.bar(model_names, mses)
            ax1.set_title('Model MSE Comparison')
            ax1.set_ylabel('MSE')
            ax1.tick_params(axis='x', rotation=45)
            
            ax2.bar(model_names, r2s)
            ax2.set_title('Model R² Comparison')
            ax2.set_ylabel('R²')
            ax2.tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            plt.savefig(results_dir / f'model_comparison_{target_column}.png', dpi=300, bbox_inches='tight')
            plt.show()
    
    def run_full_pipeline(self, dataset_name: str = 'tox21', target_column: str = None):
        print("Starting Molecular Property Prediction Pipeline...")
        print("=" * 60)
        
        self.load_and_process_data(dataset_name)
        
        if target_column is None:
            dataset_info = self.config.DATASETS[dataset_name]
            target_column = dataset_info['tasks'][0]
        
        print(f"Target column: {target_column}")
        
        task_type = 'classification' if self.data[target_column].dtype in ['object', 'category'] or self.data[target_column].nunique() <= 2 else 'regression'
        print(f"Task type: {task_type}")
        
        X_desc, y_desc = self.prepare_traditional_features(target_column)
        X_fp, y_fp = self.prepare_fingerprint_features(target_column)
        graphs, y_graph = self.prepare_graph_data(target_column)
        
        print(f"Descriptor features shape: {X_desc.shape}")
        print(f"Fingerprint features shape: {X_fp.shape}")
        print(f"Graph data samples: {len(graphs)}")
        
        self.train_traditional_models(X_desc, y_desc, task_type)
        self.train_traditional_models(X_fp, y_fp, task_type)
        self.train_graph_models(graphs, y_graph, task_type)
        
        self.create_visualizations(target_column)
        
        print("\n" + "=" * 60)
        print("Pipeline completed successfully!")
        
        results_df = pd.DataFrame({
            name: {k: v for k, v in results.items() if k not in ['predictions', 'probabilities', 'targets']}
            for name, results in self.results.items()
        }).T
        
        print("\nFinal Results:")
        print(results_df)
        
        results_df.to_csv(self.config.RESULTS_DIR / f"molecular_results_{target_column}.csv")
        
        return results_df

def main():
    pipeline = MolecularPropertyPredictionPipeline()
    
    results = pipeline.run_full_pipeline(dataset_name='tox21', target_column='NR-AR')
    
    print("\nResults saved to:")
    print(f"- Data: {pipeline.config.DATA_DIR}")
    print(f"- Models: {pipeline.config.MODELS_DIR}")
    print(f"- Results: {pipeline.config.RESULTS_DIR}")

if __name__ == "__main__":
    main()
