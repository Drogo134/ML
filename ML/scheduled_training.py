#!/usr/bin/env python3
"""
Скрипт для автоматического обучения моделей по расписанию
"""

import os
import sys
import json
import time
import logging
import schedule
from datetime import datetime, timedelta
from pathlib import Path

# Добавляем пути к проектам
sys.path.append('human_behavior_prediction')
sys.path.append('biochemistry_molecules')
sys.path.append('small_ml_project')

from train_models import ModelTrainer
from incremental_training import IncrementalTrainer

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('scheduled_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ScheduledTrainer:
    def __init__(self):
        self.config = {
            'full_training_interval': 'weekly',  # daily, weekly, monthly
            'incremental_training_interval': 'daily',  # hourly, daily
            'incremental_cycles': 2,
            'new_samples': 1000,
            'auto_retrain_threshold': 0.05,  # 5% снижение производительности
            'max_retries': 3
        }
        self.load_config()
        
    def load_config(self):
        """Загрузка конфигурации"""
        if os.path.exists('training_config.json'):
            with open('training_config.json', 'r') as f:
                self.config.update(json.load(f))
    
    def save_config(self):
        """Сохранение конфигурации"""
        with open('training_config.json', 'w') as f:
            json.dump(self.config, f, indent=2)
    
    def check_model_performance(self, project_name):
        """Проверка производительности моделей"""
        try:
            if project_name == 'human_behavior':
                from human_behavior_prediction.main import HumanBehaviorPredictionPipeline
                pipeline = HumanBehaviorPredictionPipeline()
                pipeline.model_trainer.load_models()
                
                if not pipeline.model_trainer.models:
                    return None
                
                # Загружаем тестовые данные
                pipeline.load_data()
                X, y = pipeline.prepare_features('will_purchase')
                X_train, X_val, X_test, y_train, y_val, y_test = pipeline.model_trainer.prepare_data(X, y)
                
                # Оцениваем производительность
                performance = pipeline.model_trainer.evaluate_all_models(X_test, y_test)
                
                return performance
            
            elif project_name == 'molecular':
                from biochemistry_molecules.main import MolecularPropertyPredictionPipeline
                pipeline = MolecularPropertyPredictionPipeline()
                pipeline.model_trainer.load_models()
                
                if not pipeline.model_trainer.models:
                    return None
                
                # Загружаем тестовые данные
                pipeline.load_and_process_data('tox21')
                X_desc, y_desc = pipeline.prepare_traditional_features('NR-AR')
                X_train, X_val, X_test, y_train, y_val, y_test = pipeline.model_trainer.prepare_data(X_desc, y_desc)
                
                # Оцениваем производительность
                performance = pipeline.model_trainer.evaluate_all_models(X_test, y_test)
                
                return performance
            
            elif project_name == 'small_ml':
                from small_ml_project.main import MLPipeline
                pipeline = MLPipeline()
                pipeline.model_trainer.load_models()
                
                if not pipeline.model_trainer.models:
                    return None
                
                # Генерируем тестовые данные
                pipeline.generate_data(task_type='classification', n_samples=1000)
                X, y = pipeline.preprocess_data()
                X_train, X_val, X_test, y_train, y_val, y_test = pipeline.model_trainer.prepare_data(X, y)
                
                # Оцениваем производительность
                performance = pipeline.model_trainer.evaluate_all_models(X_test, y_test)
                
                return performance
            
        except Exception as e:
            logger.error(f"Ошибка при проверке производительности {project_name}: {e}")
            return None
    
    def needs_retraining(self, project_name):
        """Проверка необходимости переобучения"""
        try:
            # Загружаем историю производительности
            history_file = f'{project_name}_performance_history.json'
            if not os.path.exists(history_file):
                return True  # Первое обучение
            
            with open(history_file, 'r') as f:
                history = json.load(f)
            
            if not history or len(history) < 2:
                return True
            
            # Получаем текущую производительность
            current_performance = self.check_model_performance(project_name)
            if not current_performance:
                return True
            
            # Сравниваем с предыдущей производительностью
            last_performance = history[-1]['performance']
            
            for model_name, current_metrics in current_performance.items():
                if model_name in last_performance:
                    last_metrics = last_performance[model_name]
                    
                    # Проверяем снижение производительности
                    if 'accuracy' in current_metrics and 'accuracy' in last_metrics:
                        accuracy_drop = last_metrics['accuracy'] - current_metrics['accuracy']
                        if accuracy_drop > self.config['auto_retrain_threshold']:
                            logger.info(f"Обнаружено снижение точности {model_name}: {accuracy_drop:.4f}")
                            return True
                    
                    if 'mse' in current_metrics and 'mse' in last_metrics:
                        mse_increase = current_metrics['mse'] - last_metrics['mse']
                        if mse_increase > self.config['auto_retrain_threshold']:
                            logger.info(f"Обнаружено увеличение MSE {model_name}: {mse_increase:.4f}")
                            return True
            
            return False
            
        except Exception as e:
            logger.error(f"Ошибка при проверке необходимости переобучения {project_name}: {e}")
            return True
    
    def save_performance_history(self, project_name, performance):
        """Сохранение истории производительности"""
        history_file = f'{project_name}_performance_history.json'
        
        if os.path.exists(history_file):
            with open(history_file, 'r') as f:
                history = json.load(f)
        else:
            history = []
        
        history.append({
            'timestamp': datetime.now().isoformat(),
            'performance': performance
        })
        
        # Ограничиваем историю последними 50 записями
        if len(history) > 50:
            history = history[-50:]
        
        with open(history_file, 'w') as f:
            json.dump(history, f, indent=2)
    
    def full_training_job(self):
        """Задача полного обучения"""
        logger.info("🔄 ЗАПУСК ПОЛНОГО ОБУЧЕНИЯ ПО РАСПИСАНИЮ")
        
        try:
            trainer = ModelTrainer()
            success = trainer.run_full_training()
            
            if success:
                logger.info("✅ Полное обучение завершено успешно")
            else:
                logger.error("❌ Ошибка при полном обучении")
                
        except Exception as e:
            logger.error(f"❌ Критическая ошибка при полном обучении: {e}")
    
    def incremental_training_job(self):
        """Задача инкрементального обучения"""
        logger.info("🔄 ЗАПУСК ИНКРЕМЕНТАЛЬНОГО ОБУЧЕНИЯ ПО РАСПИСАНИЮ")
        
        try:
            trainer = IncrementalTrainer()
            
            # Проверяем необходимость переобучения для каждого проекта
            projects = ['human_behavior', 'molecular', 'small_ml']
            
            for project in projects:
                if self.needs_retraining(project):
                    logger.info(f"Переобучение необходимо для {project}")
                    
                    if project == 'human_behavior':
                        trainer.incremental_behavior_training(
                            new_samples=self.config['new_samples'],
                            cycles=self.config['incremental_cycles']
                        )
                    elif project == 'molecular':
                        trainer.incremental_molecular_training(
                            new_samples=self.config['new_samples'],
                            cycles=self.config['incremental_cycles']
                        )
                    elif project == 'small_ml':
                        trainer.incremental_small_ml_training(
                            new_samples=self.config['new_samples'],
                            cycles=self.config['incremental_cycles']
                        )
                    
                    # Сохраняем историю производительности
                    performance = self.check_model_performance(project)
                    if performance:
                        self.save_performance_history(project, performance)
                else:
                    logger.info(f"Переобучение не требуется для {project}")
            
            logger.info("✅ Инкрементальное обучение завершено")
            
        except Exception as e:
            logger.error(f"❌ Критическая ошибка при инкрементальном обучении: {e}")
    
    def health_check_job(self):
        """Задача проверки здоровья системы"""
        logger.info("🔍 ПРОВЕРКА ЗДОРОВЬЯ СИСТЕМЫ")
        
        try:
            # Проверяем доступность моделей
            projects = ['human_behavior', 'molecular', 'small_ml']
            
            for project in projects:
                models_dir = Path(f'{project}/models')
                if not models_dir.exists() or not list(models_dir.glob('*')):
                    logger.warning(f"⚠️ Модели не найдены для {project}")
                else:
                    logger.info(f"✅ Модели доступны для {project}")
            
            # Проверяем свободное место на диске
            import shutil
            free_space = shutil.disk_usage('.').free / (1024**3)  # GB
            if free_space < 1:
                logger.warning(f"⚠️ Мало свободного места: {free_space:.2f} GB")
            else:
                logger.info(f"✅ Свободное место: {free_space:.2f} GB")
            
            # Проверяем логи на ошибки
            log_files = ['training.log', 'incremental_training.log', 'scheduled_training.log']
            for log_file in log_files:
                if os.path.exists(log_file):
                    with open(log_file, 'r') as f:
                        content = f.read()
                        if 'ERROR' in content or 'CRITICAL' in content:
                            logger.warning(f"⚠️ Обнаружены ошибки в {log_file}")
            
            logger.info("✅ Проверка здоровья завершена")
            
        except Exception as e:
            logger.error(f"❌ Ошибка при проверке здоровья: {e}")
    
    def setup_schedule(self):
        """Настройка расписания"""
        logger.info("Настройка расписания обучения...")
        
        # Полное обучение
        if self.config['full_training_interval'] == 'daily':
            schedule.every().day.at("02:00").do(self.full_training_job)
        elif self.config['full_training_interval'] == 'weekly':
            schedule.every().monday.at("02:00").do(self.full_training_job)
        elif self.config['full_training_interval'] == 'monthly':
            schedule.every().month.do(self.full_training_job)
        
        # Инкрементальное обучение
        if self.config['incremental_training_interval'] == 'hourly':
            schedule.every().hour.do(self.incremental_training_job)
        elif self.config['incremental_training_interval'] == 'daily':
            schedule.every().day.at("01:00").do(self.incremental_training_job)
        
        # Проверка здоровья
        schedule.every().day.at("00:00").do(self.health_check_job)
        
        logger.info("Расписание настроено:")
        logger.info(f"- Полное обучение: {self.config['full_training_interval']}")
        logger.info(f"- Инкрементальное обучение: {self.config['incremental_training_interval']}")
        logger.info("- Проверка здоровья: ежедневно")
    
    def run_scheduler(self):
        """Запуск планировщика"""
        logger.info("🚀 ЗАПУСК ПЛАНИРОВЩИКА ОБУЧЕНИЯ")
        logger.info("=" * 60)
        
        self.setup_schedule()
        
        while True:
            try:
                schedule.run_pending()
                time.sleep(60)  # Проверка каждую минуту
                
            except KeyboardInterrupt:
                logger.info("Остановка планировщика...")
                break
            except Exception as e:
                logger.error(f"Ошибка в планировщике: {e}")
                time.sleep(300)  # Ждем 5 минут перед повтором
    
    def run_once(self):
        """Запуск обучения один раз"""
        logger.info("🚀 ЗАПУСК ОДНОРАЗОВОГО ОБУЧЕНИЯ")
        
        # Проверяем необходимость переобучения
        projects = ['human_behavior', 'molecular', 'small_ml']
        
        for project in projects:
            if self.needs_retraining(project):
                logger.info(f"Запуск обучения для {project}")
                
                if project == 'human_behavior':
                    trainer = IncrementalTrainer()
                    trainer.incremental_behavior_training(
                        new_samples=self.config['new_samples'],
                        cycles=self.config['incremental_cycles']
                    )
                elif project == 'molecular':
                    trainer = IncrementalTrainer()
                    trainer.incremental_molecular_training(
                        new_samples=self.config['new_samples'],
                        cycles=self.config['incremental_cycles']
                    )
                elif project == 'small_ml':
                    trainer = IncrementalTrainer()
                    trainer.incremental_small_ml_training(
                        new_samples=self.config['new_samples'],
                        cycles=self.config['incremental_cycles']
                    )
                
                # Сохраняем историю производительности
                performance = self.check_model_performance(project)
                if performance:
                    self.save_performance_history(project, performance)
            else:
                logger.info(f"Обучение не требуется для {project}")

def main():
    """Основная функция"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Планировщик обучения моделей ML')
    parser.add_argument('--mode', choices=['schedule', 'once'], default='once',
                       help='Режим работы: schedule (по расписанию) или once (один раз)')
    parser.add_argument('--config', type=str, help='Путь к файлу конфигурации')
    
    args = parser.parse_args()
    
    trainer = ScheduledTrainer()
    
    if args.config:
        with open(args.config, 'r') as f:
            config = json.load(f)
            trainer.config.update(config)
            trainer.save_config()
    
    if args.mode == 'schedule':
        trainer.run_scheduler()
    else:
        trainer.run_once()

if __name__ == "__main__":
    main()
