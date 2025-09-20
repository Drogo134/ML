import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw, Descriptors
import networkx as nx
from typing import List, Dict, Any, Optional
import warnings
warnings.filterwarnings('ignore')

class MolecularVisualization:
    def __init__(self, config):
        self.config = config
        self.colors = px.colors.qualitative.Set3
        
    def plot_molecular_descriptors_distribution(self, df: pd.DataFrame, 
                                              descriptor_cols: List[str] = None):
        if descriptor_cols is None:
            descriptor_cols = ['MW', 'LogP', 'TPSA', 'HBD', 'HBA']
        
        available_cols = [col for col in descriptor_cols if col in df.columns]
        
        if not available_cols:
            print("No descriptor columns found")
            return
        
        n_cols = min(3, len(available_cols))
        n_rows = (len(available_cols) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
        if n_rows == 1:
            axes = [axes] if n_cols == 1 else axes
        else:
            axes = axes.flatten()
        
        for i, col in enumerate(available_cols):
            if i < len(axes):
                axes[i].hist(df[col].dropna(), bins=30, alpha=0.7, color=self.colors[i % len(self.colors)])
                axes[i].set_title(f'Distribution of {col}')
                axes[i].set_xlabel(col)
                axes[i].set_ylabel('Frequency')
                axes[i].grid(True, alpha=0.3)
        
        for i in range(len(available_cols), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.show()
    
    def plot_molecular_property_correlation(self, df: pd.DataFrame, 
                                          descriptor_cols: List[str] = None):
        if descriptor_cols is None:
            descriptor_cols = ['MW', 'LogP', 'TPSA', 'HBD', 'HBA', 'RotBonds', 'AromaticRings']
        
        available_cols = [col for col in descriptor_cols if col in df.columns]
        
        if len(available_cols) < 2:
            print("Not enough descriptor columns for correlation plot")
            return
        
        correlation_matrix = df[available_cols].corr()
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                   square=True, fmt='.2f')
        plt.title('Molecular Descriptors Correlation Matrix')
        plt.tight_layout()
        plt.show()
    
    def plot_molecular_scatter(self, df: pd.DataFrame, x_col: str, y_col: str, 
                             color_col: str = None, size_col: str = None):
        if x_col not in df.columns or y_col not in df.columns:
            print(f"Columns {x_col} or {y_col} not found in dataframe")
            return
        
        plt.figure(figsize=(10, 8))
        
        if color_col and color_col in df.columns:
            scatter = plt.scatter(df[x_col], df[y_col], c=df[color_col], 
                                alpha=0.6, cmap='viridis')
            plt.colorbar(scatter, label=color_col)
        else:
            plt.scatter(df[x_col], df[y_col], alpha=0.6)
        
        if size_col and size_col in df.columns:
            sizes = df[size_col] * 100
            plt.scatter(df[x_col], df[y_col], s=sizes, alpha=0.6)
        
        plt.xlabel(x_col)
        plt.ylabel(y_col)
        plt.title(f'{y_col} vs {x_col}')
        plt.grid(True, alpha=0.3)
        plt.show()
    
    def plot_model_performance_comparison(self, results: Dict[str, Dict], 
                                        task_type: str = 'classification'):
        if task_type == 'classification':
            metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc']
        else:
            metrics = ['mse', 'mae', 'r2']
        
        available_metrics = [m for m in metrics if any(m in r for r in results.values())]
        
        if not available_metrics:
            print("No metrics available for plotting")
            return
        
        n_metrics = len(available_metrics)
        n_cols = min(3, n_metrics)
        n_rows = (n_metrics + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
        if n_rows == 1:
            axes = [axes] if n_cols == 1 else axes
        else:
            axes = axes.flatten()
        
        for i, metric in enumerate(available_metrics):
            if i < len(axes):
                model_names = list(results.keys())
                metric_values = [results[model].get(metric, 0) for model in model_names]
                
                bars = axes[i].bar(model_names, metric_values, 
                                 color=self.colors[:len(model_names)])
                axes[i].set_title(f'Model Comparison - {metric.upper()}')
                axes[i].set_ylabel(metric.upper())
                axes[i].tick_params(axis='x', rotation=45)
                
                for bar, value in zip(bars, metric_values):
                    axes[i].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                               f'{value:.3f}', ha='center', va='bottom')
        
        for i in range(len(available_metrics), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.show()
    
    def plot_roc_curves(self, results: Dict[str, Dict], save_path: str = None):
        plt.figure(figsize=(10, 8))
        
        for model_name, result in results.items():
            if 'y_true' in result and 'y_pred_proba' in result:
                from sklearn.metrics import roc_curve, auc
                
                fpr, tpr, _ = roc_curve(result['y_true'], result['y_pred_proba'])
                roc_auc = auc(fpr, tpr)
                
                plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.3f})')
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_confusion_matrices(self, results: Dict[str, Dict], save_path: str = None):
        n_models = len(results)
        n_cols = min(3, n_models)
        n_rows = (n_models + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
        if n_rows == 1:
            axes = [axes] if n_cols == 1 else axes
        else:
            axes = axes.flatten()
        
        for i, (model_name, result) in enumerate(results.items()):
            if i < len(axes) and 'y_true' in result and 'y_pred' in result:
                from sklearn.metrics import confusion_matrix
                
                cm = confusion_matrix(result['y_true'], result['y_pred'])
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i])
                axes[i].set_title(f'Confusion Matrix - {model_name}')
                axes[i].set_xlabel('Predicted')
                axes[i].set_ylabel('Actual')
        
        for i in range(len(results), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_feature_importance(self, feature_importance: Dict[str, np.ndarray], 
                              feature_names: List[str], top_n: int = 20):
        fig, axes = plt.subplots(1, len(feature_importance), figsize=(6 * len(feature_importance), 6))
        if len(feature_importance) == 1:
            axes = [axes]
        
        for i, (model_name, importance) in enumerate(feature_importance.items()):
            if i < len(axes):
                importance_df = pd.DataFrame({
                    'feature': feature_names,
                    'importance': importance
                }).sort_values('importance', ascending=False).head(top_n)
                
                sns.barplot(data=importance_df, x='importance', y='feature', ax=axes[i])
                axes[i].set_title(f'Feature Importance - {model_name}')
                axes[i].set_xlabel('Importance')
        
        plt.tight_layout()
        plt.show()
    
    def plot_molecular_structures(self, smiles_list: List[str], 
                                labels: List[str] = None, 
                                max_mols: int = 20):
        if len(smiles_list) > max_mols:
            smiles_list = smiles_list[:max_mols]
            if labels:
                labels = labels[:max_mols]
        
        mols = [Chem.MolFromSmiles(smiles) for smiles in smiles_list]
        mols = [mol for mol in mols if mol is not None]
        
        if not mols:
            print("No valid molecules found")
            return
        
        img = Draw.MolsToGridImage(mols, molsPerRow=4, subImgSize=(300, 300))
        
        if labels:
            print("Molecule labels:", labels[:len(mols)])
        
        return img
    
    def plot_molecular_network(self, smiles_list: List[str], 
                             similarity_threshold: float = 0.7):
        from rdkit import DataStructs
        from rdkit.Chem import rdMolDescriptors
        
        mols = [Chem.MolFromSmiles(smiles) for smiles in smiles_list]
        mols = [mol for mol in mols if mol is not None]
        
        if len(mols) < 2:
            print("Need at least 2 valid molecules for network plot")
            return
        
        fingerprints = [rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, 2) for mol in mols]
        
        G = nx.Graph()
        
        for i, mol in enumerate(mols):
            G.add_node(i, smiles=smiles_list[i])
        
        for i in range(len(mols)):
            for j in range(i + 1, len(mols)):
                similarity = DataStructs.TanimotoSimilarity(fingerprints[i], fingerprints[j])
                if similarity >= similarity_threshold:
                    G.add_edge(i, j, weight=similarity)
        
        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(G, k=1, iterations=50)
        
        nx.draw_networkx_nodes(G, pos, node_color='lightblue', 
                              node_size=500, alpha=0.7)
        nx.draw_networkx_edges(G, pos, alpha=0.5)
        nx.draw_networkx_labels(G, pos, font_size=8)
        
        plt.title(f'Molecular Similarity Network (threshold: {similarity_threshold})')
        plt.axis('off')
        plt.show()
    
    def plot_learning_curves(self, history: Dict[str, List[float]], 
                           save_path: str = None):
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(history['train_loss'], label='Training Loss')
        plt.plot(history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        plt.plot(history['train_acc'], label='Training Accuracy')
        plt.plot(history['val_acc'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_interactive_dashboard(self, results: Dict[str, Dict], 
                                  save_path: str = None):
        models = list(results.keys())
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Accuracy', 'Precision', 'Recall', 'F1-Score'),
            specs=[[{"type": "bar"}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "bar"}]]
        )
        
        metrics = ['accuracy', 'precision', 'recall', 'f1']
        
        for i, metric in enumerate(metrics):
            row = i // 2 + 1
            col = i % 2 + 1
            
            values = [results[model].get(metric, 0) for model in models]
            
            fig.add_trace(
                go.Bar(x=models, y=values, name=metric, showlegend=False),
                row=row, col=col
            )
        
        fig.update_layout(
            title="Molecular Property Prediction - Model Performance",
            height=600,
            showlegend=False
        )
        
        if save_path:
            fig.write_html(save_path)
        
        fig.show()
    
    def plot_dataset_statistics(self, df: pd.DataFrame, 
                              target_column: str = None):
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        if target_column and target_column in df.columns:
            target_counts = df[target_column].value_counts()
            axes[0, 0].pie(target_counts.values, labels=target_counts.index, autopct='%1.1f%%')
            axes[0, 0].set_title(f'Target Distribution - {target_column}')
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            df[numeric_cols].hist(ax=axes[0, 1], bins=20, alpha=0.7)
            axes[0, 1].set_title('Numeric Features Distribution')
        
        if len(numeric_cols) > 1:
            correlation_matrix = df[numeric_cols].corr()
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', 
                       center=0, ax=axes[1, 0])
            axes[1, 0].set_title('Feature Correlation Matrix')
        
        missing_data = df.isnull().sum()
        missing_data = missing_data[missing_data > 0]
        if len(missing_data) > 0:
            missing_data.plot(kind='bar', ax=axes[1, 1])
            axes[1, 1].set_title('Missing Data by Feature')
            axes[1, 1].tick_params(axis='x', rotation=45)
        else:
            axes[1, 1].text(0.5, 0.5, 'No Missing Data', ha='center', va='center')
            axes[1, 1].set_title('Missing Data Analysis')
        
        plt.tight_layout()
        plt.show()
