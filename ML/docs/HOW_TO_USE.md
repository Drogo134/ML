# Как пользоваться системой обучения моделей

## 🚀 Быстрый старт

### 1. Установка зависимостей

```bash
# Установка всех зависимостей
pip install -r requirements.txt

# Создание директорий
python -c "
from human_behavior_prediction.config import Config as C1
from biochemistry_molecules.config import Config as C2
from small_ml_project.config import Config as C3
C1.create_directories(); C2.create_directories(); C3.create_directories()
"
```

### 2. Быстрое обучение (рекомендуется)

```bash
# Интерактивный скрипт
python quick_train.py
```

### 3. Полное обучение всех моделей

```bash
# Обучение с настройками по умолчанию
python manage_training.py full

# Обучение с кастомными параметрами
python manage_training.py full --samples 20000 --dataset tox21
```

## 📋 Основные команды

### Обучение моделей

```bash
# Полное обучение
python manage_training.py full

# Инкрементальное обучение
python manage_training.py incremental --cycles 5

# Продолжение обучения
python manage_training.py continue --project human_behavior --epochs 20
```

### Тестирование и проверка

```bash
# Тестирование моделей
python manage_training.py test

# Проверка статуса
python manage_training.py status
```

### Управление моделями

```bash
# Резервное копирование
python manage_training.py backup

# Восстановление
python manage_training.py restore --backup-path backup/20241201_143022

# Очистка (осторожно!)
python manage_training.py clean
```

## 🔄 Циклы обучения

### 1. Первоначальное обучение

```bash
# Шаг 1: Полное обучение
python manage_training.py full --samples 15000

# Шаг 2: Проверка статуса
python manage_training.py status

# Шаг 3: Тестирование
python manage_training.py test

# Шаг 4: Резервное копирование
python manage_training.py backup
```

### 2. Инкрементальное улучшение

```bash
# Шаг 1: Инкрементальное обучение
python manage_training.py incremental --cycles 3

# Шаг 2: Тестирование улучшенных моделей
python manage_training.py test

# Шаг 3: Проверка статуса
python manage_training.py status
```

### 3. Продолжение обучения

```bash
# Шаг 1: Продолжение обучения нейронных сетей
python manage_training.py continue --project human_behavior --epochs 50

# Шаг 2: Тестирование обновленных моделей
python manage_training.py test
```

## 📊 Мониторинг

### Проверка статуса

```bash
python manage_training.py status
```

Показывает:
- ✅ Статус обучения каждого проекта
- 📁 Список обученных моделей
- 💾 Размер файлов моделей
- ⏰ Время последнего обучения

### Просмотр логов

```bash
# Логи обучения
tail -f training.log

# Логи инкрементального обучения
tail -f incremental_training.log

# Логи тестирования
tail -f model_testing.log
```

### Отчеты

```bash
# Отчет о полном обучении
cat training_report.md

# Отчет об инкрементальном обучении
cat incremental_training_report.md

# Отчет о тестировании
cat model_test_report.md
```

## 🛠️ Устранение проблем

### Проблема: Модели не загружаются

```bash
# Проверьте статус
python manage_training.py status

# Если модели отсутствуют, обучите их
python manage_training.py full
```

### Проблема: Ошибка памяти

```bash
# Уменьшите количество образцов
python manage_training.py full --samples 5000

# Или используйте инкрементальное обучение
python manage_training.py incremental --cycles 2
```

### Проблема: Ошибка при тестировании

```bash
# Убедитесь, что модели обучены
python manage_training.py status

# Если модели отсутствуют, обучите их
python manage_training.py full
```

## 📁 Структура файлов

```
проекты ML/
├── manage_training.py          # Главный скрипт управления
├── train_models.py             # Полное обучение
├── incremental_training.py     # Инкрементальное обучение
├── test_models.py              # Тестирование моделей
├── quick_train.py              # Быстрый запуск
├── training_status.json        # Статус обучения
├── training_history.json       # История обучения
├── training_report.md          # Отчет о полном обучении
├── incremental_training_report.md # Отчет об инкрементальном обучении
├── model_test_report.md        # Отчет о тестировании
├── training.log                # Логи полного обучения
├── incremental_training.log    # Логи инкрементального обучения
├── model_testing.log           # Логи тестирования
└── backup/                     # Резервные копии
    └── YYYYMMDD_HHMMSS/
        ├── human_behavior_prediction/
        ├── biochemistry_molecules/
        └── small_ml_project/
```

## 🎯 Рекомендации по использованию

### 1. Первый запуск

```bash
# 1. Установите зависимости
pip install -r requirements.txt

# 2. Создайте директории
python -c "from human_behavior_prediction.config import Config as C1; from biochemistry_molecules.config import Config as C2; from small_ml_project.config import Config as C3; C1.create_directories(); C2.create_directories(); C3.create_directories()"

# 3. Запустите полное обучение
python manage_training.py full

# 4. Проверьте статус
python manage_training.py status

# 5. Протестируйте модели
python manage_training.py test
```

### 2. Регулярное использование

```bash
# Еженедельно: инкрементальное обучение
python manage_training.py incremental --cycles 3

# Еженедельно: тестирование
python manage_training.py test

# Ежедневно: резервное копирование
python manage_training.py backup
```

### 3. Автоматизация

```bash
# Добавьте в crontab для автоматического выполнения
# Еженедельное инкрементальное обучение (каждое воскресенье в 2:00)
0 2 * * 0 cd /path/to/projects && python manage_training.py incremental --cycles 3

# Ежедневное резервное копирование (каждый день в 3:00)
0 3 * * * cd /path/to/projects && python manage_training.py backup

# Еженедельное тестирование (каждое воскресенье в 4:00)
0 4 * * 0 cd /path/to/projects && python manage_training.py test
```

## 🔧 Настройка параметров

### Параметры полного обучения

```bash
python manage_training.py full --samples 20000 --dataset bace
```

- `--samples`: Количество образцов для генерации данных
- `--dataset`: Название датасета для молекулярных моделей

### Параметры инкрементального обучения

```bash
python manage_training.py incremental --cycles 5
```

- `--cycles`: Количество циклов обучения

### Параметры продолжения обучения

```bash
python manage_training.py continue --project human_behavior --epochs 50
```

- `--project`: Название проекта
- `--epochs`: Количество дополнительных эпох

## 📈 Мониторинг производительности

### Проверка качества моделей

```bash
# Тестирование всех моделей
python manage_training.py test

# Просмотр отчета
cat model_test_report.md
```

### Анализ логов

```bash
# Поиск ошибок
grep -i error training.log

# Поиск предупреждений
grep -i warning training.log

# Анализ производительности
grep -i "accuracy\|mse\|auc" training.log
```

## 🚨 Важные замечания

### 1. Резервное копирование

**Всегда создавайте резервные копии перед:**
- Очисткой моделей
- Полным переобучением
- Изменением параметров

```bash
python manage_training.py backup
```

### 2. Мониторинг ресурсов

**Следите за:**
- Использованием памяти
- Размером файлов моделей
- Временем обучения

```bash
# Проверка размера моделей
du -sh */models/

# Проверка использования памяти
top -p $(pgrep -f python)
```

### 3. Очистка

**Очищайте старые файлы:**
- Логи старше 30 дней
- Резервные копии старше 90 дней
- Временные файлы

```bash
# Очистка старых логов
find . -name "*.log" -mtime +30 -delete

# Очистка старых резервных копий
find backup/ -type d -mtime +90 -exec rm -rf {} +
```

## 🎉 Заключение

Система обучения моделей предоставляет:

1. **Простое использование** - интерактивные скрипты
2. **Гибкость** - множество параметров настройки
3. **Надежность** - резервное копирование и восстановление
4. **Мониторинг** - детальные логи и отчеты
5. **Автоматизацию** - возможность настройки cron задач

Используйте эти инструменты для эффективного обучения и управления моделями машинного обучения!
