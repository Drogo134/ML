import pandas as pd
import numpy as np
import os
import warnings
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import sys

# Добавляем путь к AI интеграции
sys.path.append('../shared')
from ai_integration import AIEnhancer

from config import Config
from data_generator import HumanBehaviorDataGenerator
from feature_engineering import FeatureEngineer
from models import ModelTrainer
from evaluation import ModelEvaluator

warnings.filterwarnings('ignore')

class HumanBehaviorPredictionPipeline:
    def __init__(self):
        self.config = Config()
        self.config.create_directories()
        
        self.data_generator = HumanBehaviorDataGenerator()
        self.feature_engineer = FeatureEngineer(self.config)
        self.model_trainer = ModelTrainer(self.config)
        self.evaluator = ModelEvaluator(self.config)
        self.ai_enhancer = AIEnhancer()
        
        self.data = None
        self.processed_data = None
        self.results = {}
        
    def generate_data(self, n_samples=10000, save_data=True):
        print("Generating synthetic human behavior data...")
        self.data = self.data_generator.generate_dataset(n_samples)
        
        if save_data:
            data_path = self.config.DATA_DIR / "human_behavior_data.csv"
            self.data.to_csv(data_path, index=False)
            print(f"Data saved to {data_path}")
        
        print(f"Generated dataset with shape: {self.data.shape}")
        return self.data
    
    def load_data(self, file_path=None):
        if file_path is None:
            file_path = self.config.DATA_DIR / "human_behavior_data.csv"
        
        if os.path.exists(file_path):
            self.data = pd.read_csv(file_path)
            print(f"Loaded data with shape: {self.data.shape}")
        else:
            print("Data file not found. Generating new data...")
            self.generate_data()
        
        return self.data
    
    def prepare_features(self, target_column='will_purchase'):
        print("Engineering features...")
        
        feature_columns = [col for col in self.data.columns if col not in 
                          ['will_purchase', 'will_churn', 'engagement_level', 'lifetime_value']]
        
        X = self.data[feature_columns]
        y = self.data[target_column]
        
        self.processed_data = self.feature_engineer.engineer_features(
            self.data, target_col=target_column, fit=True
        )
        
        processed_feature_columns = [col for col in self.processed_data.columns 
                                   if col not in ['will_purchase', 'will_churn', 'engagement_level', 'lifetime_value']]
        
        X_processed = self.processed_data[processed_feature_columns]
        y_processed = self.processed_data[target_column]
        
        print(f"Processed features shape: {X_processed.shape}")
        return X_processed, y_processed
    
    def train_models(self, X, y, target_type='classification'):
        print("Training models...")
        
        X_train_scaled, X_val_scaled, X_test_scaled, X_train, X_val, X_test, y_train, y_val, y_test = \
            self.model_trainer.prepare_data(X, y)
        
        models_to_train = ['xgboost', 'lightgbm', 'neural_network']
        
        for model_name in models_to_train:
            print(f"Training {model_name}...")
            
            if model_name == 'neural_network':
                model = self.model_trainer.train_neural_network(
                    X_train_scaled, y_train, X_val_scaled, y_val, target_type
                )
            elif model_name == 'xgboost':
                model = self.model_trainer.train_xgboost(
                    X_train, y_train, X_val, y_val, target_type
                )
            elif model_name == 'lightgbm':
                model = self.model_trainer.train_lightgbm(
                    X_train, y_train, X_val, y_val, target_type
                )
            
            if model_name == 'neural_network':
                y_pred = (model.predict(X_test_scaled) > 0.5).astype(int)
                y_pred_proba = model.predict(X_test_scaled)
            else:
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
            
            metrics = self.evaluator.calculate_classification_metrics(y_test, y_pred, y_pred_proba)
            self.results[model_name] = metrics
            
            print(f"{model_name} - Accuracy: {metrics['accuracy']:.4f}, AUC: {metrics.get('auc', 'N/A')}")
        
        self.model_trainer.save_models()
        return X_test, y_test
    
    def evaluate_models(self, X_test, y_test):
        print("Evaluating models...")
        
        for model_name, model in self.model_trainer.models.items():
            if model_name == 'neural_network':
                y_pred = (model.predict(X_test) > 0.5).astype(int)
                y_pred_proba = model.predict(X_test)
            else:
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
            
            metrics = self.evaluator.calculate_classification_metrics(y_test, y_pred, y_pred_proba)
            self.results[model_name] = metrics
            
            print(f"\n{model_name} Results:")
            for metric, value in metrics.items():
                print(f"  {metric}: {value:.4f}")
    
    def create_visualizations(self, X_test, y_test):
        print("Creating visualizations...")
        
        results_dir = self.config.RESULTS_DIR
        
        for model_name, model in self.model_trainer.models.items():
            if model_name == 'neural_network':
                y_pred = (model.predict(X_test) > 0.5).astype(int)
                y_pred_proba = model.predict(X_test)
            else:
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
            
            self.evaluator.plot_confusion_matrix(
                y_test, y_pred, model_name, 
                save_path=results_dir / f"confusion_matrix_{model_name}.png"
            )
            
            if y_pred_proba is not None:
                self.evaluator.plot_roc_curve(
                    y_test, y_pred_proba, model_name,
                    save_path=results_dir / f"roc_curve_{model_name}.png"
                )
                
                self.evaluator.plot_precision_recall_curve(
                    y_test, y_pred_proba, model_name,
                    save_path=results_dir / f"precision_recall_{model_name}.png"
                )
            
            if hasattr(model, 'feature_importances_'):
                feature_names = [f"feature_{i}" for i in range(X_test.shape[1])]
                self.evaluator.plot_feature_importance(
                    feature_names, model.feature_importances_, model_name,
                    save_path=results_dir / f"feature_importance_{model_name}.png"
                )
    
    def run_full_pipeline(self, n_samples=10000, target_column='will_purchase'):
        print("Starting Human Behavior Prediction Pipeline...")
        print("=" * 50)
        
        self.generate_data(n_samples)
        X, y = self.prepare_features(target_column)
        X_test, y_test = self.train_models(X, y)
        self.evaluate_models(X_test, y_test)
        self.create_visualizations(X_test, y_test)
        
        print("\n" + "=" * 50)
        print("Pipeline completed successfully!")
        
        results_df = pd.DataFrame(self.results).T
        print("\nFinal Results:")
        print(results_df)
        
        results_df.to_csv(self.config.RESULTS_DIR / "model_results.csv")
        
        return results_df
    
    def enhance_with_ai(self, X_test, y_test, predictions=None):
        """Улучшение результатов с помощью AI"""
        print("\n🤖 AI Enhancement for Human Behavior Prediction")
        print("=" * 50)
        
        try:
            # Подготавливаем данные для AI анализа
            sample_data = {
                "data_shape": X_test.shape,
                "target_distribution": np.bincount(y_test).tolist(),
                "feature_statistics": {
                    "mean": np.mean(X_test, axis=0).tolist()[:5],  # Первые 5 признаков
                    "std": np.std(X_test, axis=0).tolist()[:5]
                }
            }
            
            # AI анализ данных
            ai_analysis = self.ai_enhancer.enhance_human_behavior_prediction(
                sample_data, predictions if predictions is not None else []
            )
            
            if ai_analysis:
                print("✅ AI анализ завершен")
                
                # Сохраняем AI анализ
                ai_results_dir = self.config.RESULTS_DIR / "ai_analysis"
                ai_results_dir.mkdir(exist_ok=True)
                
                with open(ai_results_dir / "behavior_analysis.txt", "w", encoding="utf-8") as f:
                    f.write(ai_analysis.get("ai_analysis", "AI анализ недоступен"))
                
                print(f"AI анализ сохранен в {ai_results_dir}")
                return ai_analysis
            else:
                print("❌ Ошибка AI анализа")
                return None
                
        except Exception as e:
            print(f"❌ Ошибка AI интеграции: {e}")
            return None
    
    def generate_ai_insights(self, model_performance):
        """Генерация AI инсайтов о производительности модели"""
        try:
            insights = self.ai_enhancer.enhance_ml_results(model_performance)
            
            if insights and insights.get("insights"):
                print("✅ AI инсайты сгенерированы")
                
                # Сохраняем инсайты
                ai_results_dir = self.config.RESULTS_DIR / "ai_analysis"
                ai_results_dir.mkdir(exist_ok=True)
                
                with open(ai_results_dir / "model_insights.txt", "w", encoding="utf-8") as f:
                    f.write(insights["insights"])
                
                return insights
            else:
                print("❌ Ошибка генерации AI инсайтов")
                return None
                
        except Exception as e:
            print(f"❌ Ошибка генерации AI инсайтов: {e}")
            return None
    
    def create_ai_report(self, results):
        """Создание AI отчета о проекте"""
        try:
            report = self.ai_enhancer.create_project_report(
                "Human Behavior Prediction System", results
            )
            
            if report:
                print("✅ AI отчет создан")
                
                # Сохраняем отчет
                ai_results_dir = self.config.RESULTS_DIR / "ai_analysis"
                ai_results_dir.mkdir(exist_ok=True)
                
                with open(ai_results_dir / "ai_report.md", "w", encoding="utf-8") as f:
                    f.write(report)
                
                return report
            else:
                print("❌ Ошибка создания AI отчета")
                return None
                
        except Exception as e:
            print(f"❌ Ошибка создания AI отчета: {e}")
            return None

def main():
    pipeline = HumanBehaviorPredictionPipeline()
    
    results = pipeline.run_full_pipeline(n_samples=10000)
    
    print("\nResults saved to:")
    print(f"- Data: {pipeline.config.DATA_DIR}")
    print(f"- Models: {pipeline.config.MODELS_DIR}")
    print(f"- Results: {pipeline.config.RESULTS_DIR}")
    
    # AI интеграция
    print("\n" + "=" * 50)
    print("AI INTEGRATION")
    print("=" * 50)
    
    # Генерируем AI инсайты
    pipeline.generate_ai_insights(results.to_dict())
    
    # Создаем AI отчет
    pipeline.create_ai_report(results.to_dict())

if __name__ == "__main__":
    main()
