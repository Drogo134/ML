import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import warnings
from typing import Dict, List, Tuple
import os

from config import Config
from data_generator import DataGenerator
from models import ModelTrainer

warnings.filterwarnings('ignore')

class MLPipeline:
    def __init__(self):
        self.config = Config()
        self.config.create_directories()
        
        self.data_generator = DataGenerator(self.config)
        self.model_trainer = ModelTrainer(self.config)
        
        self.data = None
        self.results = {}
        
    def generate_data(self, task_type='classification', **kwargs):
        print(f"Generating {task_type} dataset...")
        
        self.data = self.data_generator.create_dataset(task_type=task_type, **kwargs)
        
        print(f"Dataset shape: {self.data.shape}")
        print(f"Target distribution:")
        print(self.data['target'].value_counts())
        
        return self.data
    
    def preprocess_data(self, target_column='target'):
        print("Preprocessing data...")
        
        X = self.data.drop(columns=[target_column])
        y = self.data[target_column]
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)
        
        print(f"Preprocessed features shape: {X_scaled_df.shape}")
        return X_scaled_df, y
    
    def run_pipeline(self, task_type='classification', **kwargs):
        print(f"Starting ML Pipeline for {task_type}...")
        print("=" * 50)
        
        self.generate_data(task_type=task_type, **kwargs)
        X, y = self.preprocess_data()
        
        X_train, X_val, X_test, y_train, y_val, y_test = self.model_trainer.prepare_data(
            X, y, task_type=task_type
        )
        
        self.model_trainer.train_all_models(X_train, y_train, X_val, y_val, task_type)
        self.model_trainer.evaluate_all_models(X_test, y_test, task_type)
        
        self.create_visualizations(X_test, y_test, task_type)
        
        print("\n" + "=" * 50)
        print("Pipeline completed successfully!")
        
        results_df = pd.DataFrame(self.model_trainer.results).T
        print("\nFinal Results:")
        print(results_df)
        
        results_df.to_csv(self.config.RESULTS_DIR / f"ml_results_{task_type}.csv")
        
        self.model_trainer.save_models()
        
        return results_df
    
    def create_visualizations(self, X_test, y_test, task_type):
        print("Creating visualizations...")
        
        results_dir = self.config.RESULTS_DIR
        
        if task_type == 'classification':
            self.plot_classification_results(y_test, task_type)
        else:
            self.plot_regression_results(y_test, task_type)
        
        self.plot_model_comparison(task_type)
    
    def plot_classification_results(self, y_test, task_type):
        results_dir = self.config.RESULTS_DIR
        
        for name, model in self.model_trainer.models.items():
            y_pred = model.predict(X_test)
            
            plt.figure(figsize=(12, 4))
            
            plt.subplot(1, 2, 1)
            cm = confusion_matrix(y_test, y_pred)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title(f'Confusion Matrix - {name}')
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            
            plt.subplot(1, 2, 2)
            if hasattr(model, 'predict_proba'):
                y_pred_proba = model.predict_proba(X_test)[:, 1]
                plt.hist(y_pred_proba, bins=20, alpha=0.7)
                plt.title(f'Prediction Probabilities - {name}')
                plt.xlabel('Probability')
                plt.ylabel('Frequency')
            
            plt.tight_layout()
            plt.savefig(results_dir / f'classification_results_{name}.png', dpi=300, bbox_inches='tight')
            plt.show()
    
    def plot_regression_results(self, y_test, task_type):
        results_dir = self.config.RESULTS_DIR
        
        for name, model in self.model_trainer.models.items():
            y_pred = model.predict(X_test)
            
            plt.figure(figsize=(12, 4))
            
            plt.subplot(1, 2, 1)
            plt.scatter(y_test, y_pred, alpha=0.6)
            plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
            plt.xlabel('Actual')
            plt.ylabel('Predicted')
            plt.title(f'Actual vs Predicted - {name}')
            
            plt.subplot(1, 2, 2)
            residuals = y_test - y_pred
            plt.scatter(y_pred, residuals, alpha=0.6)
            plt.axhline(y=0, color='r', linestyle='--')
            plt.xlabel('Predicted')
            plt.ylabel('Residuals')
            plt.title(f'Residuals Plot - {name}')
            
            plt.tight_layout()
            plt.savefig(results_dir / f'regression_results_{name}.png', dpi=300, bbox_inches='tight')
            plt.show()
    
    def plot_model_comparison(self, task_type):
        results_dir = self.config.RESULTS_DIR
        
        if task_type == 'classification':
            metrics = ['accuracy', 'precision', 'recall', 'f1']
        else:
            metrics = ['mse', 'mae', 'r2']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, metric in enumerate(metrics):
            if i < len(axes):
                values = [self.model_trainer.results[name].get(metric, 0) for name in self.model_trainer.results.keys()]
                names = list(self.model_trainer.results.keys())
                
                axes[i].bar(names, values)
                axes[i].set_title(f'Model Comparison - {metric.upper()}')
                axes[i].set_ylabel(metric.upper())
                axes[i].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(results_dir / f'model_comparison_{task_type}.png', dpi=300, bbox_inches='tight')
        plt.show()

def main():
    pipeline = MLPipeline()
    
    print("ML Pipeline Demo")
    print("=" * 50)
    
    print("\n1. Classification Task")
    results_clf = pipeline.run_pipeline(
        task_type='classification',
        n_samples=1000,
        n_features=20,
        n_classes=2
    )
    
    print("\n2. Regression Task")
    results_reg = pipeline.run_pipeline(
        task_type='regression',
        n_samples=1000,
        n_features=20
    )
    
    print("\n3. Clustering Task")
    results_cluster = pipeline.run_pipeline(
        task_type='clustering',
        n_samples=1000,
        n_features=2,
        centers=3
    )
    
    print("\nResults saved to:")
    print(f"- Models: {pipeline.config.MODELS_DIR}")
    print(f"- Results: {pipeline.config.RESULTS_DIR}")

if __name__ == "__main__":
    main()
