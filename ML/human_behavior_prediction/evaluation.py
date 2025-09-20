import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, precision_recall_curve, confusion_matrix,
    mean_squared_error, mean_absolute_error, r2_score
)
import shap
import lime
import lime.lime_tabular
from sklearn.inspection import permutation_importance
import warnings
warnings.filterwarnings('ignore')

class ModelEvaluator:
    def __init__(self, config):
        self.config = config
        self.results = {}
        
    def calculate_classification_metrics(self, y_true, y_pred, y_pred_proba=None):
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted'),
            'recall': recall_score(y_true, y_pred, average='weighted'),
            'f1': f1_score(y_true, y_pred, average='weighted')
        }
        
        if y_pred_proba is not None:
            metrics['auc'] = roc_auc_score(y_true, y_pred_proba)
            
        return metrics
    
    def calculate_regression_metrics(self, y_true, y_pred):
        return {
            'mse': mean_squared_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred)
        }
    
    def plot_confusion_matrix(self, y_true, y_pred, model_name, save_path=None):
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {model_name}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_roc_curve(self, y_true, y_pred_proba, model_name, save_path=None):
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        auc = roc_auc_score(y_true, y_pred_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc:.3f})')
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {model_name}')
        plt.legend()
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_precision_recall_curve(self, y_true, y_pred_proba, model_name, save_path=None):
        precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, label=f'{model_name}')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall Curve - {model_name}')
        plt.legend()
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_feature_importance(self, feature_names, importance_scores, model_name, top_n=20, save_path=None):
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance_scores
        }).sort_values('importance', ascending=False).head(top_n)
        
        plt.figure(figsize=(10, 8))
        sns.barplot(data=importance_df, x='importance', y='feature')
        plt.title(f'Feature Importance - {model_name}')
        plt.xlabel('Importance')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_residuals(self, y_true, y_pred, model_name, save_path=None):
        residuals = y_true - y_pred
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        axes[0].scatter(y_pred, residuals, alpha=0.6)
        axes[0].axhline(y=0, color='r', linestyle='--')
        axes[0].set_xlabel('Predicted Values')
        axes[0].set_ylabel('Residuals')
        axes[0].set_title(f'Residuals Plot - {model_name}')
        axes[0].grid(True)
        
        axes[1].hist(residuals, bins=30, alpha=0.7)
        axes[1].set_xlabel('Residuals')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title(f'Residuals Distribution - {model_name}')
        axes[1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_interactive_dashboard(self, results_dict, save_path=None):
        models = list(results_dict.keys())
        metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc']
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Accuracy', 'Precision', 'Recall', 'F1-Score'),
            specs=[[{"type": "bar"}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "bar"}]]
        )
        
        for i, metric in enumerate(metrics[:4]):
            row = i // 2 + 1
            col = i % 2 + 1
            
            values = [results_dict[model].get(metric, 0) for model in models]
            
            fig.add_trace(
                go.Bar(x=models, y=values, name=metric),
                row=row, col=col
            )
        
        fig.update_layout(
            title="Model Performance Comparison",
            showlegend=False,
            height=600
        )
        
        if save_path:
            fig.write_html(save_path)
        
        fig.show()
    
    def explain_predictions_shap(self, model, X, feature_names, model_name, max_samples=100):
        if hasattr(model, 'predict_proba'):
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X[:max_samples])
            
            plt.figure(figsize=(10, 8))
            shap.summary_plot(shap_values, X[:max_samples], feature_names=feature_names, show=False)
            plt.title(f'SHAP Summary - {model_name}')
            plt.tight_layout()
            plt.show()
            
            return explainer, shap_values
        else:
            print(f"SHAP not supported for {model_name}")
            return None, None
    
    def explain_predictions_lime(self, model, X, y, feature_names, model_name, instance_idx=0):
        try:
            explainer = lime.lime_tabular.LimeTabularExplainer(
                X, feature_names=feature_names, class_names=['No', 'Yes']
            )
            
            explanation = explainer.explain_instance(
                X[instance_idx], 
                model.predict_proba, 
                num_features=10
            )
            
            explanation.show_in_notebook(show_table=True)
            return explanation
        except Exception as e:
            print(f"LIME explanation failed for {model_name}: {e}")
            return None
    
    def cross_validate_model(self, model, X, y, cv=5, scoring='accuracy'):
        from sklearn.model_selection import cross_val_score
        
        scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
        
        return {
            'mean_score': scores.mean(),
            'std_score': scores.std(),
            'scores': scores
        }
    
    def compare_models(self, models_dict, X_test, y_test, task_type='classification'):
        results = {}
        
        for name, model in models_dict.items():
            y_pred = model.predict(X_test)
            
            if task_type == 'classification':
                y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
                results[name] = self.calculate_classification_metrics(y_test, y_pred, y_pred_proba)
            else:
                results[name] = self.calculate_regression_metrics(y_test, y_pred)
        
        results_df = pd.DataFrame(results).T
        return results_df
    
    def generate_report(self, results_dict, save_path=None):
        report = f"""
# Model Evaluation Report

## Summary
Total models evaluated: {len(results_dict)}

## Results
"""
        
        for model_name, metrics in results_dict.items():
            report += f"\n### {model_name}\n"
            for metric, value in metrics.items():
                if isinstance(value, float):
                    report += f"- {metric}: {value:.4f}\n"
                else:
                    report += f"- {metric}: {value}\n"
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report)
        
        return report
