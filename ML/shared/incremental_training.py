#!/usr/bin/env python3
"""
Скрипт для инкрементального обучения моделей
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
        logging.FileHandler('incremental_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class IncrementalTrainer:
    def __init__(self):
        self.training_history = {}
        self.load_training_history()
    
    def load_training_history(self):
        """Загрузка истории обучения"""
        if os.path.exists('training_history.json'):
            with open('training_history.json', 'r') as f:
                self.training_history = json.load(f)
        else:
            self.training_history = {
                'human_behavior': {'cycles': 0, 'last_training': None},
                'molecular': {'cycles': 0, 'last_training': None},
                'small_ml': {'cycles': 0, 'last_training': None}
            }
    
    def save_training_history(self):
        """Сохранение истории обучения"""
        with open('training_history.json', 'w') as f:
            json.dump(self.training_history, f, indent=2)
    
    def incremental_train_human_behavior(self, n_samples=5000, cycles=3):
        """Инкрементальное обучение моделей прогноза поведения"""
        logger.info("=" * 60)
        logger.info("ИНКРЕМЕНТАЛЬНОЕ ОБУЧЕНИЕ МОДЕЛЕЙ ПРОГНОЗА ПОВЕДЕНИЯ")
        logger.info("=" * 60)
        
        try:
            pipeline = HumanBehaviorPredictionPipeline()
            
            for cycle in range(cycles):
                logger.info(f"Цикл обучения {cycle + 1}/{cycles}")
                
                # Генерируем новые данные
                new_data = pipeline.data_generator.generate_dataset(n_samples=n_samples)
                
                # Загружаем существующие данные если есть
                existing_data_path = Path('human_behavior_prediction/data/human_behavior_data.csv')
                if existing_data_path.exists():
                    existing_data = pipeline.load_data()
                    # Объединяем данные
                    combined_data = pipeline.data_generator.data_generator.create_dataset(
                        n_samples=len(existing_data) + len(new_data)
                    )
                else:
                    combined_data = new_data
                
                # Сохраняем объединенные данные
                combined_data.to_csv(existing_data_path, index=False)
                
                # Подготавливаем признаки
                X, y = pipeline.prepare_features('will_purchase')
                
                # Загружаем существующие модели если есть
                models_dir = Path('human_behavior_prediction/models')
                if models_dir.exists() and any(models_dir.glob('*.pkl')):
                    pipeline.model_trainer.load_models()
                
                # Обучаем модели
                X_train, X_val, X_test, y_train, y_val, y_test = pipeline.model_trainer.prepare_data(X, y)
                
                # Обучаем каждую модель
                for model_name in ['xgboost', 'lightgbm', 'neural_network']:
                    if model_name == 'neural_network':
                        model = pipeline.model_trainer.train_neural_network(
                            X_train, y_train, X_val, y_val, 'classification'
                        )
                    elif model_name == 'xgboost':
                        model = pipeline.model_trainer.train_xgboost(
                            X_train, y_train, X_val, y_val, 'classification'
                        )
                    elif model_name == 'lightgbm':
                        model = pipeline.model_trainer.train_lightgbm(
                            X_train, y_train, X_val, y_val, 'classification'
                        )
                
                # Сохраняем модели
                pipeline.model_trainer.save_models()
                
                # Оцениваем модели
                pipeline.evaluate_models(X_test, y_test)
                
                # Обновляем историю
                self.training_history['human_behavior']['cycles'] += 1
                self.training_history['human_behavior']['last_training'] = datetime.now().isoformat()
                
                logger.info(f"✅ Цикл {cycle + 1} завершен")
            
            self.save_training_history()
            logger.info("✅ Инкрементальное обучение моделей прогноза поведения завершено!")
            return True
            
        except Exception as e:
            logger.error(f"❌ Ошибка при инкрементальном обучении: {e}")
            return False
    
    def incremental_train_molecular(self, dataset_name='tox21', cycles=2):
        """Инкрементальное обучение моделей молекулярных свойств"""
        logger.info("=" * 60)
        logger.info("ИНКРЕМЕНТАЛЬНОЕ ОБУЧЕНИЕ МОДЕЛЕЙ МОЛЕКУЛЯРНЫХ СВОЙСТВ")
        logger.info("=" * 60)
        
        try:
            pipeline = MolecularPropertyPredictionPipeline()
            
            for cycle in range(cycles):
                logger.info(f"Цикл обучения {cycle + 1}/{cycles}")
                
                # Загружаем данные
                pipeline.load_and_process_data(dataset_name)
                
                # Подготавливаем признаки
                X_desc, y_desc = pipeline.prepare_traditional_features('NR-AR')
                X_fp, y_fp = pipeline.prepare_fingerprint_features('NR-AR')
                graphs, y_graph = pipeline.prepare_graph_data('NR-AR')
                
                # Загружаем существующие модели если есть
                models_dir = Path('biochemistry_molecules/models')
                if models_dir.exists() and any(models_dir.glob('*.pkl')):
                    pipeline.model_trainer.load_models()
                
                # Обучаем традиционные модели
                pipeline.train_traditional_models(X_desc, y_desc, 'classification')
                pipeline.train_traditional_models(X_fp, y_fp, 'classification')
                
                # Обучаем графовые модели
                pipeline.train_graph_models(graphs, y_graph, 'classification')
                
                # Сохраняем модели
                pipeline.model_trainer.save_models()
                
                # Обновляем историю
                self.training_history['molecular']['cycles'] += 1
                self.training_history['molecular']['last_training'] = datetime.now().isoformat()
                
                logger.info(f"✅ Цикл {cycle + 1} завершен")
            
            self.save_training_history()
            logger.info("✅ Инкрементальное обучение моделей молекулярных свойств завершено!")
            return True
            
        except Exception as e:
            logger.error(f"❌ Ошибка при инкрементальном обучении: {e}")
            return False
    
    def incremental_train_small_ml(self, task_types=['classification', 'regression'], cycles=2):
        """Инкрементальное обучение моделей малого ML проекта"""
        logger.info("=" * 60)
        logger.info("ИНКРЕМЕНТАЛЬНОЕ ОБУЧЕНИЕ МОДЕЛЕЙ МАЛОГО ML ПРОЕКТА")
        logger.info("=" * 60)
        
        try:
            pipeline = MLPipeline()
            
            for cycle in range(cycles):
                logger.info(f"Цикл обучения {cycle + 1}/{cycles}")
                
                for task_type in task_types:
                    logger.info(f"Обучение для задачи: {task_type}")
                    
                    # Генерируем данные
                    pipeline.generate_data(task_type=task_type, n_samples=1000)
                    
                    # Подготавливаем данные
                    X, y = pipeline.preprocess_data()
                    
                    # Загружаем существующие модели если есть
                    models_dir = Path('small_ml_project/models')
                    if models_dir.exists() and any(models_dir.glob('*.pkl')):
                        pipeline.model_trainer.load_models()
                    
                    # Обучаем модели
                    X_train, X_val, X_test, y_train, y_val, y_test = pipeline.model_trainer.prepare_data(
                        X, y, task_type=task_type
                    )
                    
                    pipeline.model_trainer.train_all_models(X_train, y_train, X_val, y_val, task_type)
                    pipeline.model_trainer.evaluate_all_models(X_test, y_test, task_type)
                    
                    # Сохраняем модели
                    pipeline.model_trainer.save_models()
                
                # Обновляем историю
                self.training_history['small_ml']['cycles'] += 1
                self.training_history['small_ml']['last_training'] = datetime.now().isoformat()
                
                logger.info(f"✅ Цикл {cycle + 1} завершен")
            
            self.save_training_history()
            logger.info("✅ Инкрементальное обучение моделей малого ML проекта завершено!")
            return True
            
        except Exception as e:
            logger.error(f"❌ Ошибка при инкрементальном обучении: {e}")
            return False
    
    def run_incremental_training(self, cycles=3):
        """Запуск инкрементального обучения всех проектов"""
        logger.info("🚀 НАЧАЛО ИНКРЕМЕНТАЛЬНОГО ОБУЧЕНИЯ")
        logger.info("=" * 80)
        
        start_time = time.time()
        
        # Инкрементальное обучение моделей прогноза поведения
        self.incremental_train_human_behavior(cycles=cycles)
        
        # Инкрементальное обучение моделей молекулярных свойств
        self.incremental_train_molecular(cycles=cycles)
        
        # Инкрементальное обучение моделей малого ML проекта
        self.incremental_train_small_ml(cycles=cycles)
        
        total_time = time.time() - start_time
        logger.info("=" * 80)
        logger.info(f"✅ ИНКРЕМЕНТАЛЬНОЕ ОБУЧЕНИЕ ЗАВЕРШЕНО за {total_time:.2f} секунд")
        logger.info("=" * 80)
        
        # Генерируем отчет
        self.generate_incremental_report()
    
    def generate_incremental_report(self):
        """Генерация отчета об инкрементальном обучении"""
        logger.info("Генерация отчета об инкрементальном обучении...")
        
        report = f"""
# Отчет об инкрементальном обучении моделей ML
Сгенерирован: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## История обучения

### Human Behavior Prediction
- Количество циклов: {self.training_history['human_behavior']['cycles']}
- Последнее обучение: {self.training_history['human_behavior']['last_training']}

### Molecular Property Prediction
- Количество циклов: {self.training_history['molecular']['cycles']}
- Последнее обучение: {self.training_history['molecular']['last_training']}

### Small ML Project
- Количество циклов: {self.training_history['small_ml']['cycles']}
- Последнее обучение: {self.training_history['small_ml']['last_training']}

## Рекомендации

1. Продолжайте инкрементальное обучение для улучшения качества моделей
2. Мониторьте производительность моделей на новых данных
3. Регулярно сохраняйте резервные копии обученных моделей
4. Анализируйте результаты каждого цикла обучения

## Следующие шаги

1. Запустите тестирование моделей
2. Проверьте качество предсказаний
3. При необходимости проведите дополнительное обучение
"""
        
        # Сохраняем отчет
        with open('incremental_training_report.md', 'w', encoding='utf-8') as f:
            f.write(report)
        
        logger.info("Отчет сохранен в incremental_training_report.md")

def main():
    """Основная функция"""
    trainer = IncrementalTrainer()
    
    # Инкрементальное обучение
    trainer.run_incremental_training(cycles=3)

if __name__ == "__main__":
    main()