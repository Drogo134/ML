#!/usr/bin/env python3
"""
Скрипт для обучения всех моделей с сохранением и загрузкой
"""

import os
import sys
import json
import time
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
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ModelTrainer:
    def __init__(self):
        self.training_status = {
            'human_behavior': {'status': 'not_started', 'models': []},
            'molecular': {'status': 'not_started', 'models': []},
            'small_ml': {'status': 'not_started', 'models': []}
        }
        self.results = {}
        
    def save_training_status(self):
        """Сохранение статуса обучения"""
        with open('training_status.json', 'w') as f:
            json.dump(self.training_status, f, indent=2)
    
    def load_training_status(self):
        """Загрузка статуса обучения"""
        if os.path.exists('training_status.json'):
            with open('training_status.json', 'r') as f:
                self.training_status = json.load(f)
    
    def train_human_behavior_models(self, n_samples=10000):
        """Обучение моделей прогноза поведения"""
        logger.info("=" * 60)
        logger.info("ОБУЧЕНИЕ МОДЕЛЕЙ ПРОГНОЗА ПОВЕДЕНИЯ")
        logger.info("=" * 60)
        
        try:
            pipeline = HumanBehaviorPredictionPipeline()
            
            # Проверяем, есть ли уже обученные модели
            models_dir = Path('human_behavior_prediction/models')
            existing_models = list(models_dir.glob('*.pkl')) + list(models_dir.glob('*.h5'))
            
            if existing_models and self.training_status['human_behavior']['status'] == 'completed':
                logger.info("Модели уже обучены, загружаем существующие...")
                pipeline.model_trainer.load_models()
                self.training_status['human_behavior']['status'] = 'loaded'
                return True
            
            # Генерируем данные
            logger.info("Генерация данных...")
            pipeline.generate_data(n_samples=n_samples, save_data=True)
            
            # Подготавливаем признаки
            logger.info("Подготовка признаков...")
            X, y = pipeline.prepare_features('will_purchase')
            
            # Обучаем модели
            logger.info("Обучение моделей...")
            X_test, y_test = pipeline.train_models(X, y)
            
            # Сохраняем модели
            pipeline.model_trainer.save_models()
            
            # Оцениваем модели
            pipeline.evaluate_models(X_test, y_test)
            
            # Создаем визуализации
            pipeline.create_visualizations(X_test, y_test)
            
            self.training_status['human_behavior']['status'] = 'completed'
            self.training_status['human_behavior']['models'] = list(pipeline.model_trainer.models.keys())
            self.results['human_behavior'] = pipeline.results
            
            logger.info("✅ Модели прогноза поведения обучены успешно!")
            return True
            
        except Exception as e:
            logger.error(f"❌ Ошибка при обучении моделей прогноза поведения: {e}")
            self.training_status['human_behavior']['status'] = 'failed'
            return False
    
    def train_molecular_models(self, dataset_name='tox21'):
        """Обучение моделей молекулярных свойств"""
        logger.info("=" * 60)
        logger.info("ОБУЧЕНИЕ МОДЕЛЕЙ МОЛЕКУЛЯРНЫХ СВОЙСТВ")
        logger.info("=" * 60)
        
        try:
            pipeline = MolecularPropertyPredictionPipeline()
            
            # Проверяем, есть ли уже обученные модели
            models_dir = Path('biochemistry_molecules/models')
            existing_models = list(models_dir.glob('*.pkl')) + list(models_dir.glob('*.h5'))
            
            if existing_models and self.training_status['molecular']['status'] == 'completed':
                logger.info("Модели уже обучены, загружаем существующие...")
                self.training_status['molecular']['status'] = 'loaded'
                return True
            
            # Загружаем и обрабатываем данные
            logger.info("Загрузка и обработка данных...")
            pipeline.load_and_process_data(dataset_name)
            
            # Подготавливаем признаки
            logger.info("Подготовка признаков...")
            X_desc, y_desc = pipeline.prepare_traditional_features('NR-AR')
            X_fp, y_fp = pipeline.prepare_fingerprint_features('NR-AR')
            graphs, y_graph = pipeline.prepare_graph_data('NR-AR')
            
            # Обучаем традиционные модели
            logger.info("Обучение традиционных моделей...")
            pipeline.train_traditional_models(X_desc, y_desc, 'classification')
            pipeline.train_traditional_models(X_fp, y_fp, 'classification')
            
            # Обучаем графовые модели
            logger.info("Обучение графовых моделей...")
            pipeline.train_graph_models(graphs, y_graph, 'classification')
            
            # Создаем визуализации
            pipeline.create_visualizations('NR-AR')
            
            self.training_status['molecular']['status'] = 'completed'
            self.training_status['molecular']['models'] = list(pipeline.model_trainer.models.keys())
            self.results['molecular'] = pipeline.results
            
            logger.info("✅ Модели молекулярных свойств обучены успешно!")
            return True
            
        except Exception as e:
            logger.error(f"❌ Ошибка при обучении моделей молекулярных свойств: {e}")
            self.training_status['molecular']['status'] = 'failed'
            return False
    
    def train_small_ml_models(self, task_types=['classification', 'regression']):
        """Обучение моделей малого ML проекта"""
        logger.info("=" * 60)
        logger.info("ОБУЧЕНИЕ МОДЕЛЕЙ МАЛОГО ML ПРОЕКТА")
        logger.info("=" * 60)
        
        try:
            pipeline = MLPipeline()
            
            # Проверяем, есть ли уже обученные модели
            models_dir = Path('small_ml_project/models')
            existing_models = list(models_dir.glob('*.pkl')) + list(models_dir.glob('*.h5'))
            
            if existing_models and self.training_status['small_ml']['status'] == 'completed':
                logger.info("Модели уже обучены, загружаем существующие...")
                pipeline.model_trainer.load_models()
                self.training_status['small_ml']['status'] = 'loaded'
                return True
            
            results = {}
            
            for task_type in task_types:
                logger.info(f"Обучение моделей для задачи: {task_type}")
                
                # Генерируем данные
                pipeline.generate_data(task_type=task_type, n_samples=2000)
                
                # Подготавливаем данные
                X, y = pipeline.preprocess_data()
                
                # Обучаем модели
                X_train, X_val, X_test, y_train, y_val, y_test = pipeline.model_trainer.prepare_data(
                    X, y, task_type=task_type
                )
                
                pipeline.model_trainer.train_all_models(X_train, y_train, X_val, y_val, task_type)
                pipeline.model_trainer.evaluate_all_models(X_test, y_test, task_type)
                
                results[task_type] = pipeline.model_trainer.results
                
                # Сохраняем модели
                pipeline.model_trainer.save_models()
            
            self.training_status['small_ml']['status'] = 'completed'
            self.training_status['small_ml']['models'] = list(pipeline.model_trainer.models.keys())
            self.results['small_ml'] = results
            
            logger.info("✅ Модели малого ML проекта обучены успешно!")
            return True
            
        except Exception as e:
            logger.error(f"❌ Ошибка при обучении моделей малого ML проекта: {e}")
            self.training_status['small_ml']['status'] = 'failed'
            return False
    
    def continue_training(self, project_name, additional_epochs=10):
        """Продолжение обучения существующих моделей"""
        logger.info(f"Продолжение обучения для проекта: {project_name}")
        
        if project_name == 'human_behavior':
            pipeline = HumanBehaviorPredictionPipeline()
            pipeline.model_trainer.load_models()
            
            # Загружаем данные
            pipeline.load_data()
            X, y = pipeline.prepare_features('will_purchase')
            
            # Продолжаем обучение нейронных сетей
            for model_name, model in pipeline.model_trainer.models.items():
                if model_name == 'neural_network':
                    # Продолжаем обучение нейронной сети
                    X_train, X_val, X_test, y_train, y_val, y_test = pipeline.model_trainer.prepare_data(X, y)
                    
                    # Дополнительные эпохи
                    model.fit(
                        X_train, y_train,
                        validation_data=(X_val, y_val),
                        epochs=additional_epochs,
                        verbose=1
                    )
                    
                    # Сохраняем обновленную модель
                    model.save(f'human_behavior_prediction/models/{model_name}_continued.h5')
            
            logger.info("✅ Продолжение обучения завершено!")
            
        elif project_name == 'molecular':
            pipeline = MolecularPropertyPredictionPipeline()
            pipeline.model_trainer.load_models()
            
            # Продолжаем обучение графовых моделей
            logger.info("Продолжение обучения графовых моделей...")
            
        elif project_name == 'small_ml':
            pipeline = MLPipeline()
            pipeline.model_trainer.load_models()
            
            # Продолжаем обучение
            logger.info("Продолжение обучения моделей малого ML проекта...")
    
    def generate_training_report(self):
        """Генерация отчета об обучении"""
        logger.info("Генерация отчета об обучении...")
        
        report = f"""
# Отчет об обучении моделей ML
Сгенерирован: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Статус обучения

### Human Behavior Prediction
- Статус: {self.training_status['human_behavior']['status']}
- Модели: {', '.join(self.training_status['human_behavior']['models'])}

### Molecular Property Prediction
- Статус: {self.training_status['molecular']['status']}
- Модели: {', '.join(self.training_status['molecular']['models'])}

### Small ML Project
- Статус: {self.training_status['small_ml']['status']}
- Модели: {', '.join(self.training_status['small_ml']['models'])}

## Результаты

"""
        
        for project, results in self.results.items():
            report += f"### {project}\n"
            if isinstance(results, dict):
                for model_name, metrics in results.items():
                    report += f"- {model_name}: "
                    if 'accuracy' in metrics:
                        report += f"Accuracy={metrics['accuracy']:.4f}"
                    elif 'mse' in metrics:
                        report += f"MSE={metrics['mse']:.4f}"
                    report += "\n"
            report += "\n"
        
        # Сохраняем отчет
        with open('training_report.md', 'w', encoding='utf-8') as f:
            f.write(report)
        
        logger.info("Отчет сохранен в training_report.md")
    
    def run_full_training(self, n_samples=10000, dataset_name='tox21'):
        """Полное обучение всех моделей"""
        logger.info("🚀 НАЧАЛО ПОЛНОГО ОБУЧЕНИЯ МОДЕЛЕЙ")
        logger.info("=" * 80)
        
        start_time = time.time()
        
        # Загружаем существующий статус
        self.load_training_status()
        
        # Обучаем модели прогноза поведения
        if self.training_status['human_behavior']['status'] != 'completed':
            self.train_human_behavior_models(n_samples)
            self.save_training_status()
        
        # Обучаем модели молекулярных свойств
        if self.training_status['molecular']['status'] != 'completed':
            self.train_molecular_models(dataset_name)
            self.save_training_status()
        
        # Обучаем модели малого ML проекта
        if self.training_status['small_ml']['status'] != 'completed':
            self.train_small_ml_models()
            self.save_training_status()
        
        # Генерируем отчет
        self.generate_training_report()
        
        total_time = time.time() - start_time
        logger.info("=" * 80)
        logger.info(f"✅ ПОЛНОЕ ОБУЧЕНИЕ ЗАВЕРШЕНО за {total_time:.2f} секунд")
        logger.info("=" * 80)
        
        return True

def main():
    """Основная функция"""
    trainer = ModelTrainer()
    
    # Полное обучение
    trainer.run_full_training()
    
    # Пример продолжения обучения
    # trainer.continue_training('human_behavior', additional_epochs=20)

if __name__ == "__main__":
    main()

