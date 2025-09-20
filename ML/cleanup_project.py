#!/usr/bin/env python3
"""
Скрипт для очистки проекта и создания правильной структуры
"""

import os
import shutil
import json
from pathlib import Path

def cleanup_word_files():
    """Удаление Word и PDF файлов"""
    print("Очистка Word и PDF файлов...")
    
    # Список файлов для удаления
    files_to_remove = [
        "a-domain-specific-risk-taking-dospert-scale-for-adultpopulations.pdf",
        "a-domain.pdf", 
        "DOSPERT_40_2002.doc",
        "DOSPERT_40_2003.doc",
        "DOSPERT_40_coding.instructions.doc",
        "DOSPERT_40_coding_instructions.doc",
        "DOSPERT_40_English_2002.doc",
        "DOSPERT_40_English_2003.doc",
        "TPB Manual FINAL May2004 рус.pdf",
        "TPB Manual FINAL May2004.pdf",
        "tpb.intervention рус.pdf",
        "tpb.intervention.pdf",
        "tpb.measurement рус.pdf",
        "tpb.measurement.pdf",
        "tpb.questionnaire рус.pdf",
        "tpb.questionnaire.pdf",
        "tpb_manual рус.pdf",
        "tpb_manual.pdf",
        "tpb_with_background.png",
        "глаза.docx",
        "Диффузион-ая модель принятия решений.docx",
        "Домашний набор.docx",
        "Закон атракции.docx",
        "Закон приобретения-потерь.docx",
        "Модель готовности прототипа.docx",
        "Нерегулярное подкрепление.docx",
        "Определение лжи.docx",
        "Поведенческий анализ.docx",
        "Подталк-ее вмешательство.docx",
        "Прогноз по личности и прошлому поведению.docx",
        "Сила надежды. Обделенность.docx",
        "Ситуционизм и диспозиционизм.docx",
        "Смещенная активность.docx",
        "Теория активного вывода.docx",
        "Теория диссонанса.docx",
        "Теория запланированного поведения (одна из лучших).docx",
        "Теория коммулятивных перспектив.docx",
        "Теория контроля за действием.docx",
        "Теория разумного действия (старая версия ТЗП).docx",
        "Теория социального обмена.docx",
        "Теория социометра.docx",
        "товары для покупки.docx",
        "ТС, ТЗП, СИ.docx",
        "ТСИ.docx"
    ]
    
    removed_count = 0
    for file in files_to_remove:
        if os.path.exists(file):
            try:
                os.remove(file)
                print(f"Удален: {file}")
                removed_count += 1
            except Exception as e:
                print(f"Ошибка удаления {file}: {e}")
    
    print(f"Удалено файлов: {removed_count}")

def organize_shared_files():
    """Организация общих файлов в папку shared"""
    print("Организация общих файлов...")
    
    # Создаем папку shared если не существует
    shared_dir = Path("shared")
    shared_dir.mkdir(exist_ok=True)
    
    # Файлы для перемещения в shared
    files_to_move = [
        "ai_integration.py",
        "setup_ai_api.py", 
        "test_ai_integration.py",
        "train_models.py",
        "incremental_training.py",
        "test_models.py",
        "manage_training.py",
        "quick_train.py",
        "api_server.py",
        "test_api.py",
        "run_all_projects.py",
        "Dockerfile",
        "docker-compose.yml",
        "nginx.conf"
    ]
    
    moved_count = 0
    for file in files_to_move:
        if os.path.exists(file):
            try:
                shutil.move(file, shared_dir / file)
                print(f"Перемещен в shared: {file}")
                moved_count += 1
            except Exception as e:
                print(f"Ошибка перемещения {file}: {e}")
    
    print(f"Перемещено файлов: {moved_count}")

def organize_documentation():
    """Организация документации"""
    print("Организация документации...")
    
    # Создаем папку docs
    docs_dir = Path("docs")
    docs_dir.mkdir(exist_ok=True)
    
    # Файлы документации для перемещения
    doc_files = [
        "README.md",
        "START_HERE.md",
        "HOW_TO_USE.md",
        "TRAINING_GUIDE.md",
        "INSTALLATION.md",
        "USAGE_GUIDE.md",
        "DEPLOYMENT_GUIDE.md",
        "TESTING_GUIDE.md",
        "API_GUIDE.md",
        "PROJECT_STRUCTURE.md",
        "QUICK_START.md",
        "FILES_OVERVIEW.md",
        "FINAL_INSTRUCTIONS.md"
    ]
    
    moved_count = 0
    for file in doc_files:
        if os.path.exists(file):
            try:
                shutil.move(file, docs_dir / file)
                print(f"Перемещен в docs: {file}")
                moved_count += 1
            except Exception as e:
                print(f"Ошибка перемещения {file}: {e}")
    
    print(f"Перемещено документов: {moved_count}")

def organize_kubernetes():
    """Организация Kubernetes файлов"""
    print("Организация Kubernetes файлов...")
    
    # Перемещаем k8s в shared
    if os.path.exists("k8s"):
        try:
            shutil.move("k8s", "shared/k8s")
            print("Перемещена папка k8s в shared")
        except Exception as e:
            print(f"Ошибка перемещения k8s: {e}")

def create_project_structure():
    """Создание правильной структуры проекта"""
    print("Создание структуры проекта...")
    
    # Создаем основные папки
    directories = [
        "database",
        "shared",
        "docs", 
        "logs",
        "temp"
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"Создана папка: {directory}")

def create_main_readme():
    """Создание главного README"""
    print("Создание главного README...")
    
    main_readme = """# Machine Learning Projects Collection

## Описание

Коллекция из трех профессиональных проектов машинного обучения с AI интеграцией.

## Структура проекта

```
проекты ML/
├── database/                    # База данных с информацией
│   ├── documentation.json      # Метаданные проектов
│   ├── behavioral_theories.json # Теории поведения
│   ├── molecular_data.json     # Молекулярные данные
│   ├── training_data.json      # Данные для обучения
│   └── model_configurations.json # Конфигурации моделей
├── shared/                      # Общие файлы
│   ├── ai_integration.py       # AI интеграция
│   ├── train_models.py         # Обучение моделей
│   ├── manage_training.py      # Управление обучением
│   ├── api_server.py           # API сервер
│   ├── Dockerfile              # Docker контейнер
│   └── k8s/                    # Kubernetes манифесты
├── docs/                        # Документация
│   ├── README.md               # Основная документация
│   ├── START_HERE.md           # Начало работы
│   └── ...                     # Другие документы
├── human_behavior_prediction/   # Проект 1: Прогноз поведения
├── biochemistry_molecules/      # Проект 2: Молекулярные свойства
├── small_ml_project/           # Проект 3: Общий ML
└── requirements.txt            # Зависимости
```

## Быстрый старт

1. Установите зависимости:
```bash
pip install -r requirements.txt
```

2. Создайте директории:
```bash
python -c "from human_behavior_prediction.config import Config as C1; from biochemistry_molecules.config import Config as C2; from small_ml_project.config import Config as C3; C1.create_directories(); C2.create_directories(); C3.create_directories()"
```

3. Настройте AI API (опционально):
```bash
python shared/setup_ai_api.py
```

4. Запустите обучение:
```bash
python shared/quick_train.py
```

## Проекты

### 1. Human Behavior Prediction
- Прогнозирование поведения пользователей
- AI интеграция для анализа
- Расширяемая база данных

### 2. Molecular Property Prediction  
- Предсказание свойств молекул
- Графовые нейронные сети
- Химические базы данных

### 3. Small ML Project
- Классификация, регрессия, кластеризация
- Автоматическая генерация данных
- Сравнение моделей

## Документация

См. папку `docs/` для подробной документации.

## Лицензия

MIT License
"""
    
    with open("README.md", "w", encoding="utf-8") as f:
        f.write(main_readme)
    
    print("Создан главный README.md")

def update_imports():
    """Обновление импортов в файлах проектов"""
    print("Обновление импортов...")
    
    # Обновляем импорты в human_behavior_prediction/main.py
    main_file = "human_behavior_prediction/main.py"
    if os.path.exists(main_file):
        with open(main_file, "r", encoding="utf-8") as f:
            content = f.read()
        
        # Заменяем импорт AI интеграции
        content = content.replace(
            "sys.path.append('..')\nfrom ai_integration import AIEnhancer",
            "sys.path.append('../shared')\nfrom ai_integration import AIEnhancer"
        )
        
        with open(main_file, "w", encoding="utf-8") as f:
            f.write(content)
        
        print("Обновлены импорты в human_behavior_prediction/main.py")

def main():
    """Основная функция очистки"""
    print("=" * 60)
    print("ОЧИСТКА И ОРГАНИЗАЦИЯ ПРОЕКТА")
    print("=" * 60)
    
    # Очистка Word файлов
    cleanup_word_files()
    
    # Организация файлов
    organize_shared_files()
    organize_documentation()
    organize_kubernetes()
    
    # Создание структуры
    create_project_structure()
    create_main_readme()
    
    # Обновление импортов
    update_imports()
    
    print("\n" + "=" * 60)
    print("ОЧИСТКА ЗАВЕРШЕНА!")
    print("=" * 60)
    print("Структура проекта организована:")
    print("- Word/PDF файлы удалены")
    print("- Общие файлы перемещены в shared/")
    print("- Документация перемещена в docs/")
    print("- База данных создана в database/")
    print("- Импорты обновлены")

if __name__ == "__main__":
    main()
