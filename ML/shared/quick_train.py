#!/usr/bin/env python3
"""
Быстрый скрипт для обучения моделей
"""

import os
import sys
import subprocess
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_command(command):
    """Выполнение команды"""
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        logger.info(f"✅ Команда выполнена: {command}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Ошибка выполнения команды: {command}")
        logger.error(f"Ошибка: {e.stderr}")
        return False

def main():
    """Основная функция"""
    print("🚀 БЫСТРОЕ ОБУЧЕНИЕ МОДЕЛЕЙ ML")
    print("=" * 50)
    
    # Проверяем, что мы в правильной директории
    if not os.path.exists('manage_training.py'):
        print("❌ Ошибка: Скрипт должен быть запущен из корневой директории проектов")
        return
    
    # Создаем директории если их нет
    print("📁 Создание директорий...")
    run_command("python -c \"from human_behavior_prediction.config import Config as C1; from biochemistry_molecules.config import Config as C2; from small_ml_project.config import Config as C3; C1.create_directories(); C2.create_directories(); C3.create_directories()\"")
    
    # Проверяем статус
    print("📊 Проверка статуса...")
    run_command("python manage_training.py status")
    
    # Спрашиваем пользователя
    print("\nВыберите действие:")
    print("1. Полное обучение всех моделей")
    print("2. Инкрементальное обучение")
    print("3. Тестирование моделей")
    print("4. Проверка статуса")
    print("5. Резервное копирование")
    print("6. Выход")
    
    choice = input("\nВведите номер (1-6): ").strip()
    
    if choice == '1':
        print("\n🚀 Запуск полного обучения...")
        samples = input("Количество образцов (по умолчанию 10000): ").strip()
        if not samples:
            samples = "10000"
        run_command(f"python manage_training.py full --samples {samples}")
        
    elif choice == '2':
        print("\n🔄 Запуск инкрементального обучения...")
        cycles = input("Количество циклов (по умолчанию 3): ").strip()
        if not cycles:
            cycles = "3"
        run_command(f"python manage_training.py incremental --cycles {cycles}")
        
    elif choice == '3':
        print("\n🧪 Запуск тестирования...")
        run_command("python manage_training.py test")
        
    elif choice == '4':
        print("\n📊 Проверка статуса...")
        run_command("python manage_training.py status")
        
    elif choice == '5':
        print("\n💾 Создание резервной копии...")
        run_command("python manage_training.py backup")
        
    elif choice == '6':
        print("\n👋 До свидания!")
        return
        
    else:
        print("\n❌ Неверный выбор!")
        return
    
    print("\n✅ Операция завершена!")
    print("\nДля повторного запуска выполните: python quick_train.py")

if __name__ == "__main__":
    main()
