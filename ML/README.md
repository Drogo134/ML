# Machine Learning Projects Collection

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
