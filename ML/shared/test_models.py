#!/usr/bin/env python3
"""
Скрипт для тестирования обученных моделей
"""

import os
import sys
import json
import logging
from datetime import datetime
from pathlib import Path

# Добавляем пути к проектам
sys.path.append('human_behavior_prediction')
sys.path.append('biochemistry_molecules')
sys.path.append('small_ml_project')

from human_behavior_prediction.main import HumanBehaviorPredictionPipeline
from biochemistry_molecules.main import MolecularPropertyPredictionPipeline
from small_ml_project.main import MLPipeline

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('model_testing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ModelTester:
    def __init__(self):
        self.test_results = {}
    
    def test_human_behavior_models(self):
        """Тестирование моделей прогноза поведения"""
        logger.info("=" * 60)
        logger.info("ТЕСТИРОВАНИЕ МОДЕЛЕЙ ПРОГНОЗА ПОВЕДЕНИЯ")
        logger.info("=" * 60)
        
        try:
            pipeline = HumanBehaviorPredictionPipeline()
            
            # Загружаем данные
            pipeline.load_data()
            X, y = pipeline.prepare_features('will_purchase')
            
            # Загружаем модели
            pipeline.model_trainer.load_models()
            
            # Подготавливаем тестовые данные
            X_train, X_val, X_test, y_train, y_val, y_test = pipeline.model_trainer.prepare_data(X, y)
            
            # Тестируем каждую модель
            for model_name, model in pipeline.model_trainer.models.items():
                logger.info(f"Тестирование модели: {model_name}")
                
                # Предсказания
                if model_name == 'neural_network':
                    y_pred = (model.predict(X_test) > 0.5).astype(int)
                    y_pred_proba = model.predict(X_test)
                else:
                    y_pred = model.predict(X_test)
                    y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
                
                # Оценка
                metrics = pipeline.evaluator.calculate_classification_metrics(y_test, y_pred, y_pred_proba)
                
                self.test_results[f'human_behavior_{model_name}'] = metrics
                
                logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
                if 'auc' in metrics and metrics['auc']:
                    logger.info(f"AUC: {metrics['auc']:.4f}")
            
            logger.info("✅ Тестирование моделей прогноза поведения завершено!")
            return True
            
        except Exception as e:
            logger.error(f"❌ Ошибка при тестировании моделей прогноза поведения: {e}")
            return False
    
    def test_molecular_models(self):
        """Тестирование моделей молекулярных свойств"""
        logger.info("=" * 60)
        logger.info("ТЕСТИРОВАНИЕ МОДЕЛЕЙ МОЛЕКУЛЯРНЫХ СВОЙСТВ")
        logger.info("=" * 60)
        
        try:
            pipeline = MolecularPropertyPredictionPipeline()
            
            # Загружаем данные
            pipeline.load_and_process_data('tox21')
            
            # Подготавливаем признаки
            X_desc, y_desc = pipeline.prepare_traditional_features('NR-AR')
            X_fp, y_fp = pipeline.prepare_fingerprint_features('NR-AR')
            graphs, y_graph = pipeline.prepare_graph_data('NR-AR')
            
            # Загружаем модели
            pipeline.model_trainer.load_models()
            
            # Тестируем традиционные модели
            for model_name, model in pipeline.model_trainer.models.items():
                if not model_name.startswith('graph_'):
                    logger.info(f"Тестирование модели: {model_name}")
                    
                    # Выбираем соответствующие данные
                    if 'fp' in model_name.lower():
                        X_test, y_test = X_fp, y_fp
                    else:
                        X_test, y_test = X_desc, y_desc
                    
                    # Предсказания
                    y_pred = model.predict(X_test)
                    y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
                    
                    # Оценка
                    metrics = pipeline.evaluator.calculate_classification_metrics(y_test, y_pred, y_pred_proba)
                    
                    self.test_results[f'molecular_{model_name}'] = metrics
                    
                    logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
                    if 'auc' in metrics and metrics['auc']:
                        logger.info(f"AUC: {metrics['auc']:.4f}")
            
            logger.info("✅ Тестирование моделей молекулярных свойств завершено!")
            return True
            
        except Exception as e:
            logger.error(f"❌ Ошибка при тестировании моделей молекулярных свойств: {e}")
            return False
    
    def test_small_ml_models(self):
        """Тестирование моделей малого ML проекта"""
        logger.info("=" * 60)
        logger.info("ТЕСТИРОВАНИЕ МОДЕЛЕЙ МАЛОГО ML ПРОЕКТА")
        logger.info("=" * 60)
        
        try:
            pipeline = MLPipeline()
            
            task_types = ['classification', 'regression']
            
            for task_type in task_types:
                logger.info(f"Тестирование для задачи: {task_type}")
                
                # Генерируем тестовые данные
                pipeline.generate_data(task_type=task_type, n_samples=1000)
                X, y = pipeline.preprocess_data()
                
                # Загружаем модели
                pipeline.model_trainer.load_models()
                
                # Подготавливаем тестовые данные
                X_train, X_val, X_test, y_train, y_val, y_test = pipeline.model_trainer.prepare_data(X, y, task_type=task_type)
                
                # Тестируем каждую модель
                for model_name, model in pipeline.model_trainer.models.items():
                    logger.info(f"Тестирование модели: {model_name}")
                    
                    # Предсказания
                    if model_name == 'neural_network':
                        if task_type == 'classification':
                            y_pred = (model.predict(X_test) > 0.5).astype(int)
                            y_pred_proba = model.predict(X_test)
                        else:
                            y_pred = model.predict(X_test)
                            y_pred_proba = None
                    else:
                        y_pred = model.predict(X_test)
                        y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
                    
                    # Оценка
                    if task_type == 'classification':
                        metrics = pipeline.evaluator.calculate_classification_metrics(y_test, y_pred, y_pred_proba)
                    else:
                        metrics = pipeline.evaluator.calculate_regression_metrics(y_test, y_pred)
                    
                    self.test_results[f'small_ml_{task_type}_{model_name}'] = metrics
                    
                    if task_type == 'classification':
                        logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
                        if 'auc' in metrics and metrics['auc']:
                            logger.info(f"AUC: {metrics['auc']:.4f}")
                    else:
                        logger.info(f"MSE: {metrics['mse']:.4f}")
                        logger.info(f"R²: {metrics['r2']:.4f}")
            
            logger.info("✅ Тестирование моделей малого ML проекта завершено!")
            return True
            
        except Exception as e:
            logger.error(f"❌ Ошибка при тестировании моделей малого ML проекта: {e}")
            return False
    
    def run_all_tests(self):
        """Запуск всех тестов"""
        logger.info("🚀 НАЧАЛО ТЕСТИРОВАНИЯ МОДЕЛЕЙ")
        logger.info("=" * 80)
        
        # Тестируем все проекты
        self.test_human_behavior_models()
        self.test_molecular_models()
        self.test_small_ml_models()
        
        # Генерируем отчет
        self.generate_test_report()
        
        logger.info("=" * 80)
        logger.info("✅ ТЕСТИРОВАНИЕ МОДЕЛЕЙ ЗАВЕРШЕНО")
        logger.info("=" * 80)
    
    def generate_test_report(self):
        """Генерация отчета о тестировании"""
        logger.info("Генерация отчета о тестировании...")
        
        report = f"""
# Отчет о тестировании моделей ML
Сгенерирован: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Результаты тестирования

"""
        
        for model_name, metrics in self.test_results.items():
            report += f"### {model_name}\n"
            for metric, value in metrics.items():
                if isinstance(value, float):
                    report += f"- {metric}: {value:.4f}\n"
                else:
                    report += f"- {metric}: {value}\n"
            report += "\n"
        
        # Сохраняем отчет
        with open('model_test_report.md', 'w', encoding='utf-8') as f:
            f.write(report)
        
        logger.info("Отчет сохранен в model_test_report.md")

def main():
    """Основная функция"""
    tester = ModelTester()
    tester.run_all_tests()

if __name__ == "__main__":
    main()
