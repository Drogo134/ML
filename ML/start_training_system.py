#!/usr/bin/env python3
"""
Главный скрипт для запуска системы обучения моделей ML
"""

import os
import sys
import time
import logging
import argparse
from datetime import datetime

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('start_training_system.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def check_dependencies():
    """Проверка зависимостей"""
    logger.info("Проверка зависимостей...")
    
    required_packages = [
        'numpy', 'pandas', 'scikit-learn', 'tensorflow', 'torch',
        'xgboost', 'lightgbm', 'matplotlib', 'seaborn', 'plotly'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        logger.error(f"Отсутствуют пакеты: {missing_packages}")
        logger.error("Установите их командой: pip install -r requirements.txt")
        return False
    
    logger.info("✅ Все зависимости установлены")
    return True

def create_directories():
    """Создание необходимых директорий"""
    logger.info("Создание директорий...")
    
    directories = [
        'human_behavior_prediction/data',
        'human_behavior_prediction/models',
        'human_behavior_prediction/results',
        'human_behavior_prediction/logs',
        'biochemistry_molecules/data',
        'biochemistry_molecules/models',
        'biochemistry_molecules/results',
        'biochemistry_molecules/logs',
        'small_ml_project/data',
        'small_ml_project/models',
        'small_ml_project/results',
        'small_ml_project/logs',
        'model_backups',
        'configs'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        logger.info(f"Создана директория: {directory}")
    
    logger.info("✅ Все директории созданы")

def run_quick_training():
    """Быстрое обучение для тестирования"""
    logger.info("🚀 ЗАПУСК БЫСТРОГО ОБУЧЕНИЯ")
    
    try:
        from manage_training import TrainingManager
        
        manager = TrainingManager()
        
        # Быстрое обучение с небольшими датасетами
        logger.info("Обучение моделей прогноза поведения...")
        manager.full_training(['human_behavior'], n_samples=1000)
        
        logger.info("Обучение моделей молекулярных свойств...")
        manager.full_training(['molecular'], n_samples=500)
        
        logger.info("Обучение моделей малого ML проекта...")
        manager.full_training(['small_ml'], n_samples=1000)
        
        logger.info("✅ Быстрое обучение завершено")
        return True
        
    except Exception as e:
        logger.error(f"❌ Ошибка при быстром обучении: {e}")
        return False

def run_full_training():
    """Полное обучение всех моделей"""
    logger.info("🚀 ЗАПУСК ПОЛНОГО ОБУЧЕНИЯ")
    
    try:
        from manage_training import TrainingManager
        
        manager = TrainingManager()
        
        # Полное обучение всех проектов
        success = manager.full_training()
        
        if success:
            logger.info("✅ Полное обучение завершено успешно")
        else:
            logger.error("❌ Ошибка при полном обучении")
        
        return success
        
    except Exception as e:
        logger.error(f"❌ Критическая ошибка при полном обучении: {e}")
        return False

def run_incremental_training():
    """Инкрементальное обучение"""
    logger.info("🔄 ЗАПУСК ИНКРЕМЕНТАЛЬНОГО ОБУЧЕНИЯ")
    
    try:
        from manage_training import TrainingManager
        
        manager = TrainingManager()
        
        # Инкрементальное обучение
        success = manager.incremental_training()
        
        if success:
            logger.info("✅ Инкрементальное обучение завершено успешно")
        else:
            logger.error("❌ Ошибка при инкрементальном обучении")
        
        return success
        
    except Exception as e:
        logger.error(f"❌ Критическая ошибка при инкрементальном обучении: {e}")
        return False

def start_monitoring():
    """Запуск мониторинга"""
    logger.info("📊 ЗАПУСК МОНИТОРИНГА")
    
    try:
        from manage_training import TrainingManager
        
        manager = TrainingManager()
        manager.start_monitoring()
        
    except Exception as e:
        logger.error(f"❌ Ошибка при запуске мониторинга: {e}")

def start_scheduler():
    """Запуск планировщика"""
    logger.info("⏰ ЗАПУСК ПЛАНИРОВЩИКА")
    
    try:
        from manage_training import TrainingManager
        
        manager = TrainingManager()
        manager.start_scheduled_training()
        
    except Exception as e:
        logger.error(f"❌ Ошибка при запуске планировщика: {e}")

def show_status():
    """Показ статуса системы"""
    logger.info("🔍 ПРОВЕРКА СТАТУСА СИСТЕМЫ")
    
    try:
        from manage_training import TrainingManager
        
        manager = TrainingManager()
        status = manager.check_status()
        
        return status
        
    except Exception as e:
        logger.error(f"❌ Ошибка при проверке статуса: {e}")
        return None

def generate_report():
    """Генерация отчета"""
    logger.info("📋 ГЕНЕРАЦИЯ ОТЧЕТА")
    
    try:
        from manage_training import TrainingManager
        
        manager = TrainingManager()
        manager.generate_report()
        
        logger.info("✅ Отчет сгенерирован")
        
    except Exception as e:
        logger.error(f"❌ Ошибка при генерации отчета: {e}")

def main():
    """Основная функция"""
    parser = argparse.ArgumentParser(description='Система обучения моделей ML')
    
    parser.add_argument('--mode', choices=[
        'quick', 'full', 'incremental', 'monitor', 'schedule', 'status', 'report'
    ], default='quick', help='Режим работы')
    
    parser.add_argument('--check-deps', action='store_true', help='Проверить зависимости')
    parser.add_argument('--create-dirs', action='store_true', help='Создать директории')
    parser.add_argument('--setup', action='store_true', help='Полная настройка системы')
    
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info("СИСТЕМА ОБУЧЕНИЯ МОДЕЛЕЙ ML")
    logger.info("=" * 60)
    
    # Проверка зависимостей
    if args.check_deps or args.setup:
        if not check_dependencies():
            return 1
    
    # Создание директорий
    if args.create_dirs or args.setup:
        create_directories()
    
    # Полная настройка
    if args.setup:
        logger.info("✅ Система настроена")
        return 0
    
    # Выполнение выбранного режима
    try:
        if args.mode == 'quick':
            success = run_quick_training()
        elif args.mode == 'full':
            success = run_full_training()
        elif args.mode == 'incremental':
            success = run_incremental_training()
        elif args.mode == 'monitor':
            start_monitoring()
            return 0
        elif args.mode == 'schedule':
            start_scheduler()
            return 0
        elif args.mode == 'status':
            show_status()
            return 0
        elif args.mode == 'report':
            generate_report()
            return 0
        else:
            logger.error(f"Неизвестный режим: {args.mode}")
            return 1
        
        if success:
            logger.info("✅ Операция выполнена успешно")
            return 0
        else:
            logger.error("❌ Операция завершена с ошибками")
            return 1
            
    except KeyboardInterrupt:
        logger.info("Операция прервана пользователем")
        return 0
    except Exception as e:
        logger.error(f"Критическая ошибка: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
