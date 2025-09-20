# Руководство по обучению моделей ML

## Обзор

Система обучения моделей включает в себя несколько скриптов для полного, инкрементального обучения и тестирования моделей машинного обучения.

## Структура файлов

```
├── train_models.py              # Полное обучение всех моделей
├── incremental_training.py      # Инкрементальное обучение
├── test_models.py              # Тестирование моделей
├── manage_training.py          # Главный скрипт управления
├── training_status.json        # Статус обучения
├── training_history.json       # История обучения
└── TRAINING_GUIDE.md          # Данное руководство
```

## Быстрый старт

### 1. Полное обучение всех моделей

```bash
# Обучение всех моделей с настройками по умолчанию
python manage_training.py full

# Обучение с кастомными параметрами
python manage_training.py full --samples 20000 --dataset tox21
```

### 2. Инкрементальное обучение

```bash
# Инкрементальное обучение (3 цикла)
python manage_training.py incremental

# Инкрементальное обучение с кастомным количеством циклов
python manage_training.py incremental --cycles 5
```

### 3. Тестирование моделей

```bash
# Тестирование всех обученных моделей
python manage_training.py test
```

### 4. Проверка статуса

```bash
# Проверка статуса обучения и файлов моделей
python manage_training.py status
```

## Детальное использование

### Полное обучение (train_models.py)

Полное обучение всех моделей с нуля:

```bash
# Прямой запуск
python train_models.py

# Через manage_training.py
python manage_training.py full --samples 15000 --dataset bace
```

**Параметры:**
- `--samples`: Количество образцов для генерации данных (по умолчанию: 10000)
- `--dataset`: Название датасета для молекулярных моделей (по умолчанию: 'tox21')

**Что происходит:**
1. Генерация/загрузка данных
2. Подготовка признаков
3. Обучение всех моделей
4. Сохранение моделей
5. Оценка производительности
6. Создание визуализаций
7. Генерация отчета

### Инкрементальное обучение (incremental_training.py)

Постепенное улучшение моделей новыми данными:

```bash
# Прямой запуск
python incremental_training.py

# Через manage_training.py
python manage_training.py incremental --cycles 5
```

**Параметры:**
- `--cycles`: Количество циклов обучения (по умолчанию: 3)

**Что происходит:**
1. Загрузка существующих моделей
2. Генерация новых данных
3. Объединение с существующими данными
4. Дополнительное обучение моделей
5. Сохранение обновленных моделей
6. Обновление истории обучения

### Тестирование моделей (test_models.py)

Проверка качества обученных моделей:

```bash
# Прямой запуск
python test_models.py

# Через manage_training.py
python manage_training.py test
```

**Что происходит:**
1. Загрузка обученных моделей
2. Подготовка тестовых данных
3. Выполнение предсказаний
4. Расчет метрик качества
5. Генерация отчета о тестировании

## Управление моделями

### Проверка статуса

```bash
python manage_training.py status
```

Показывает:
- Статус обучения каждого проекта
- Список обученных моделей
- Размер файлов моделей
- Время последнего обучения

### Резервное копирование

```bash
# Создание резервной копии
python manage_training.py backup
```

Создает папку `backup/YYYYMMDD_HHMMSS/` с копиями всех моделей.

### Восстановление из резервной копии

```bash
# Восстановление из резервной копии
python manage_training.py restore --backup-path backup/20241201_143022
```

### Очистка моделей

```bash
# Удаление всех обученных моделей
python manage_training.py clean
```

**Внимание:** Эта команда удаляет все обученные модели и файлы статуса!

### Продолжение обучения

```bash
# Продолжение обучения конкретного проекта
python manage_training.py continue --project human_behavior --epochs 20
```

## Проекты и модели

### 1. Human Behavior Prediction

**Модели:**
- XGBoost
- LightGBM
- Neural Network

**Задачи:**
- Прогноз покупок
- Прогноз оттока клиентов
- Оценка вовлеченности

**Файлы:**
- `human_behavior_prediction/models/` - обученные модели
- `human_behavior_prediction/results/` - результаты и графики
- `human_behavior_prediction/logs/` - логи обучения

### 2. Molecular Property Prediction

**Модели:**
- GCN (Graph Convolutional Network)
- GAT (Graph Attention Network)
- Transformer
- XGBoost (для дескрипторов)
- LightGBM (для отпечатков)

**Задачи:**
- Предсказание токсичности
- Предсказание активности
- Предсказание ADMET свойств

**Файлы:**
- `biochemistry_molecules/models/` - обученные модели
- `biochemistry_molecules/results/` - результаты и графики
- `biochemistry_molecules/logs/` - логи обучения

### 3. Small ML Project

**Модели:**
- Random Forest
- XGBoost
- LightGBM
- Neural Network
- SVM
- Logistic Regression

**Задачи:**
- Классификация
- Регрессия
- Кластеризация

**Файлы:**
- `small_ml_project/models/` - обученные модели
- `small_ml_project/results/` - результаты и графики
- `small_ml_project/logs/` - логи обучения

## Мониторинг и логирование

### Логи обучения

```bash
# Просмотр логов
tail -f training.log
tail -f incremental_training.log
tail -f model_testing.log
```

### Файлы статуса

- `training_status.json` - текущий статус обучения
- `training_history.json` - история инкрементального обучения

### Отчеты

- `training_report.md` - отчет о полном обучении
- `incremental_training_report.md` - отчет об инкрементальном обучении
- `model_test_report.md` - отчет о тестировании

## Примеры использования

### Пример 1: Первоначальное обучение

```bash
# 1. Полное обучение всех моделей
python manage_training.py full --samples 20000

# 2. Проверка статуса
python manage_training.py status

# 3. Тестирование моделей
python manage_training.py test

# 4. Создание резервной копии
python manage_training.py backup
```

### Пример 2: Инкрементальное улучшение

```bash
# 1. Инкрементальное обучение
python manage_training.py incremental --cycles 5

# 2. Тестирование улучшенных моделей
python manage_training.py test

# 3. Проверка статуса
python manage_training.py status
```

### Пример 3: Продолжение обучения

```bash
# 1. Продолжение обучения нейронных сетей
python manage_training.py continue --project human_behavior --epochs 50

# 2. Тестирование обновленных моделей
python manage_training.py test
```

### Пример 4: Восстановление после сбоя

```bash
# 1. Восстановление из резервной копии
python manage_training.py restore --backup-path backup/20241201_143022

# 2. Проверка статуса
python manage_training.py status

# 3. Продолжение обучения
python manage_training.py incremental --cycles 2
```

## Устранение проблем

### Проблема: Модели не загружаются

**Решение:**
```bash
# Проверьте статус
python manage_training.py status

# Если модели отсутствуют, выполните полное обучение
python manage_training.py full
```

### Проблема: Ошибка памяти

**Решение:**
```bash
# Уменьшите количество образцов
python manage_training.py full --samples 5000

# Или используйте инкрементальное обучение
python manage_training.py incremental --cycles 2
```

### Проблема: Модели не сохраняются

**Решение:**
```bash
# Проверьте права доступа к папкам
ls -la */models/

# Создайте папки вручную
mkdir -p human_behavior_prediction/models
mkdir -p biochemistry_molecules/models
mkdir -p small_ml_project/models
```

### Проблема: Ошибка при тестировании

**Решение:**
```bash
# Убедитесь, что модели обучены
python manage_training.py status

# Если модели отсутствуют, обучите их
python manage_training.py full
```

## Рекомендации

### 1. Регулярное обучение

```bash
# Еженедельное инкрементальное обучение
python manage_training.py incremental --cycles 3
```

### 2. Резервное копирование

```bash
# Ежедневное резервное копирование
python manage_training.py backup
```

### 3. Мониторинг качества

```bash
# Еженедельное тестирование
python manage_training.py test
```

### 4. Очистка старых моделей

```bash
# Ежемесячная очистка (осторожно!)
python manage_training.py clean
python manage_training.py full
```

## Автоматизация

### Cron задача для инкрементального обучения

```bash
# Добавьте в crontab
0 2 * * * cd /path/to/projects && python manage_training.py incremental --cycles 2
```

### Cron задача для резервного копирования

```bash
# Добавьте в crontab
0 3 * * * cd /path/to/projects && python manage_training.py backup
```

### Cron задача для тестирования

```bash
# Добавьте в crontab
0 4 * * * cd /path/to/projects && python manage_training.py test
```

## Заключение

Система обучения моделей предоставляет:

1. **Полное обучение** - обучение всех моделей с нуля
2. **Инкрементальное обучение** - постепенное улучшение моделей
3. **Тестирование** - проверка качества моделей
4. **Управление** - резервное копирование, восстановление, очистка
5. **Мониторинг** - отслеживание статуса и производительности

Используйте эти инструменты для эффективного обучения и управления моделями машинного обучения.