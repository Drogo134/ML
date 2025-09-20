#!/usr/bin/env python3
"""
Главный скрипт для управления обучением моделей
"""

import os
import sys
import json
import argparse
import logging
from datetime import datetime

# Добавляем пути к проектам
sys.path.append('human_behavior_prediction')
sys.path.append('biochemistry_molecules')
sys.path.append('small_ml_project')

from train_models import ModelTrainer
from incremental_training import IncrementalTrainer
from test_models import ModelTester

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training_management.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class TrainingManager:
    def __init__(self):
        self.trainer = ModelTrainer()
        self.incremental_trainer = IncrementalTrainer()
        self.tester = ModelTester()
    
    def full_training(self, n_samples=10000, dataset_name='tox21'):
        """Полное обучение всех моделей"""
        logger.info("🚀 ЗАПУСК ПОЛНОГО ОБУЧЕНИЯ")
        return self.trainer.run_full_training(n_samples, dataset_name)
    
    def incremental_training(self, cycles=3):
        """Инкрементальное обучение"""
        logger.info("🔄 ЗАПУСК ИНКРЕМЕНТАЛЬНОГО ОБУЧЕНИЯ")
        return self.incremental_trainer.run_incremental_training(cycles)
    
    def test_models(self):
        """Тестирование моделей"""
        logger.info("🧪 ЗАПУСК ТЕСТИРОВАНИЯ МОДЕЛЕЙ")
        return self.tester.run_all_tests()
    
    def continue_training(self, project_name, additional_epochs=10):
        """Продолжение обучения"""
        logger.info(f"▶️ ПРОДОЛЖЕНИЕ ОБУЧЕНИЯ: {project_name}")
        return self.trainer.continue_training(project_name, additional_epochs)
    
    def check_status(self):
        """Проверка статуса обучения"""
        logger.info("📊 ПРОВЕРКА СТАТУСА ОБУЧЕНИЯ")
        
        # Загружаем статус
        self.trainer.load_training_status()
        
        print("\n" + "=" * 60)
        print("СТАТУС ОБУЧЕНИЯ МОДЕЛЕЙ")
        print("=" * 60)
        
        for project, status in self.trainer.training_status.items():
            print(f"\n{project.upper()}:")
            print(f"  Статус: {status['status']}")
            print(f"  Модели: {', '.join(status['models']) if status['models'] else 'Нет'}")
        
        # Проверяем файлы моделей
        print("\n" + "=" * 60)
        print("ФАЙЛЫ МОДЕЛЕЙ")
        print("=" * 60)
        
        projects = ['human_behavior_prediction', 'biochemistry_molecules', 'small_ml_project']
        
        for project in projects:
            models_dir = Path(project) / 'models'
            if models_dir.exists():
                model_files = list(models_dir.glob('*.pkl')) + list(models_dir.glob('*.h5'))
                print(f"\n{project}:")
                for model_file in model_files:
                    size = model_file.stat().st_size / 1024 / 1024  # MB
                    print(f"  {model_file.name}: {size:.2f} MB")
            else:
                print(f"\n{project}: Нет директории models")
    
    def clean_models(self):
        """Очистка обученных моделей"""
        logger.info("🧹 ОЧИСТКА ОБУЧЕННЫХ МОДЕЛЕЙ")
        
        projects = ['human_behavior_prediction', 'biochemistry_molecules', 'small_ml_project']
        
        for project in projects:
            models_dir = Path(project) / 'models'
            if models_dir.exists():
                for model_file in models_dir.glob('*'):
                    if model_file.is_file():
                        model_file.unlink()
                        logger.info(f"Удален: {model_file}")
        
        # Очищаем статус
        if os.path.exists('training_status.json'):
            os.remove('training_status.json')
        
        if os.path.exists('training_history.json'):
            os.remove('training_history.json')
        
        logger.info("✅ Очистка завершена")
    
    def backup_models(self):
        """Резервное копирование моделей"""
        logger.info("💾 РЕЗЕРВНОЕ КОПИРОВАНИЕ МОДЕЛЕЙ")
        
        backup_dir = Path('backup') / datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_dir.mkdir(parents=True, exist_ok=True)
        
        projects = ['human_behavior_prediction', 'biochemistry_molecules', 'small_ml_project']
        
        for project in projects:
            models_dir = Path(project) / 'models'
            if models_dir.exists():
                project_backup_dir = backup_dir / project
                project_backup_dir.mkdir(exist_ok=True)
                
                # Копируем файлы моделей
                import shutil
                for model_file in models_dir.glob('*'):
                    if model_file.is_file():
                        shutil.copy2(model_file, project_backup_dir)
                        logger.info(f"Скопирован: {model_file} -> {project_backup_dir}")
        
        logger.info(f"✅ Резервное копирование завершено: {backup_dir}")
    
    def restore_models(self, backup_path):
        """Восстановление моделей из резервной копии"""
        logger.info(f"🔄 ВОССТАНОВЛЕНИЕ МОДЕЛЕЙ ИЗ: {backup_path}")
        
        backup_dir = Path(backup_path)
        if not backup_dir.exists():
            logger.error(f"❌ Резервная копия не найдена: {backup_path}")
            return False
        
        projects = ['human_behavior_prediction', 'biochemistry_molecules', 'small_ml_project']
        
        for project in projects:
            project_backup_dir = backup_dir / project
            if project_backup_dir.exists():
                models_dir = Path(project) / 'models'
                models_dir.mkdir(parents=True, exist_ok=True)
                
                # Копируем файлы моделей
                import shutil
                for model_file in project_backup_dir.glob('*'):
                    if model_file.is_file():
                        shutil.copy2(model_file, models_dir)
                        logger.info(f"Восстановлен: {model_file} -> {models_dir}")
        
        logger.info("✅ Восстановление завершено")
        return True

def main():
    """Основная функция"""
    parser = argparse.ArgumentParser(description='Управление обучением моделей ML')
    parser.add_argument('command', choices=[
        'full', 'incremental', 'test', 'continue', 'status', 'clean', 'backup', 'restore'
    ], help='Команда для выполнения')
    parser.add_argument('--project', help='Название проекта (для continue)')
    parser.add_argument('--epochs', type=int, default=10, help='Количество эпох (для continue)')
    parser.add_argument('--cycles', type=int, default=3, help='Количество циклов (для incremental)')
    parser.add_argument('--samples', type=int, default=10000, help='Количество образцов (для full)')
    parser.add_argument('--dataset', default='tox21', help='Название датасета (для full)')
    parser.add_argument('--backup-path', help='Путь к резервной копии (для restore)')
    
    args = parser.parse_args()
    
    manager = TrainingManager()
    
    if args.command == 'full':
        manager.full_training(args.samples, args.dataset)
    elif args.command == 'incremental':
        manager.incremental_training(args.cycles)
    elif args.command == 'test':
        manager.test_models()
    elif args.command == 'continue':
        if not args.project:
            logger.error("❌ Необходимо указать проект для продолжения обучения")
            return
        manager.continue_training(args.project, args.epochs)
    elif args.command == 'status':
        manager.check_status()
    elif args.command == 'clean':
        manager.clean_models()
    elif args.command == 'backup':
        manager.backup_models()
    elif args.command == 'restore':
        if not args.backup_path:
            logger.error("❌ Необходимо указать путь к резервной копии")
            return
        manager.restore_models(args.backup_path)

if __name__ == "__main__":
    main()