# Запуск системы обучения моделей ML

## Быстрый старт

### 1. Настройка системы

```bash
# Полная настройка системы
python start_training_system.py --setup

# Или пошагово
python start_training_system.py --check-deps
python start_training_system.py --create-dirs
```

### 2. Быстрое обучение (для тестирования)

```bash
# Обучение с небольшими датасетами
python start_training_system.py --mode quick
```

### 3. Полное обучение

```bash
# Обучение всех моделей
python start_training_system.py --mode full
```

### 4. Инкрементальное обучение

```bash
# Дообучение существующих моделей
python start_training_system.py --mode incremental
```

## Детальное использование

### Управление обучением

```bash
# Полное обучение всех проектов
python manage_training.py full

# Обучение конкретных проектов
python manage_training.py full --projects human_behavior molecular

# Обучение с кастомными параметрами
python manage_training.py full --samples 15000
```

### Инкрементальное обучение

```bash
# Инкрементальное обучение
python manage_training.py incremental

# С кастомными параметрами
python manage_training.py incremental --cycles 5 --samples 2000
```

### Продолжение обучения

```bash
# Продолжение обучения нейронных сетей
python manage_training.py continue --project human_behavior --epochs 20
```

### Мониторинг

```bash
# Запуск мониторинга
python manage_training.py monitor

# Проверка статуса
python manage_training.py status

# Генерация отчета
python manage_training.py report --project human_behavior --days 7

# Построение графиков
python manage_training.py plot --project molecular --days 14
```

### Автоматическое обучение

```bash
# Запуск планировщика
python manage_training.py schedule
```

### Резервное копирование

```bash
# Создание резервной копии
python manage_training.py backup

# Восстановление
python manage_training.py restore --dir /path/to/backup
```

## Режимы работы

### 1. Быстрое обучение (quick)
- Небольшие датасеты для тестирования
- Быстрое обучение всех моделей
- Подходит для проверки работоспособности

### 2. Полное обучение (full)
- Обучение всех моделей с нуля
- Большие датасеты
- Полная функциональность

### 3. Инкрементальное обучение (incremental)
- Дообучение существующих моделей
- Добавление новых данных
- Адаптивное обучение

### 4. Мониторинг (monitor)
- Отслеживание процесса обучения
- Системные метрики
- Алерты и уведомления

### 5. Планировщик (schedule)
- Автоматическое обучение по расписанию
- Адаптивное переобучение
- Мониторинг здоровья системы

## Конфигурация

### Основные параметры

```json
{
  "default_samples": 10000,
  "incremental_samples": 1000,
  "incremental_cycles": 3,
  "auto_retrain_threshold": 0.05,
  "monitoring_interval": 60
}
```

### Параметры моделей

```json
{
  "human_behavior": {
    "xgboost": {
      "n_estimators": 1000,
      "max_depth": 6,
      "learning_rate": 0.1
    },
    "neural_network": {
      "hidden_layers": [128, 64, 32],
      "dropout_rate": 0.3,
      "epochs": 100
    }
  }
}
```

## Мониторинг

### Системные метрики
- **CPU Usage**: Загрузка процессора
- **Memory Usage**: Использование памяти
- **Disk Usage**: Использование диска
- **Training Duration**: Длительность обучения

### Алерты
- **Warning**: Высокая загрузка системы
- **Critical**: Критические ошибки
- **Info**: Информационные сообщения

### Логи
```bash
# Просмотр логов
tail -f start_training_system.log
tail -f manage_training.log
tail -f training.log
tail -f incremental_training.log
tail -f scheduled_training.log
tail -f training_monitor.log
```

## Устранение проблем

### Частые проблемы

1. **Отсутствуют зависимости**
   ```bash
   pip install -r requirements.txt
   ```

2. **Нехватка памяти**
   ```bash
   # Уменьшение размера датасета
   python start_training_system.py --mode quick
   ```

3. **Ошибки CUDA**
   ```bash
   # Установка CPU версии
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
   ```

4. **Проблемы с директориями**
   ```bash
   python start_training_system.py --create-dirs
   ```

### Отладка

```bash
# Включение детального логирования
export LOG_LEVEL=DEBUG
python start_training_system.py --mode quick

# Проверка статуса
python start_training_system.py --mode status

# Генерация отчета
python start_training_system.py --mode report
```

## Производительность

### Рекомендации

- **CPU**: 8+ ядер для параллельного обучения
- **RAM**: 16+ GB для больших датасетов
- **GPU**: NVIDIA GTX 1060+ для нейронных сетей
- **Диск**: SSD для быстрого доступа к данным

### Оптимизация

1. **Параллельное обучение**: Использование всех доступных ядер
2. **Батчевая обработка**: Обучение по батчам
3. **Кэширование**: Сохранение промежуточных результатов
4. **GPU ускорение**: Использование GPU для нейронных сетей

## Примеры использования

### Пример 1: Первый запуск

```bash
# 1. Настройка системы
python start_training_system.py --setup

# 2. Быстрое обучение для тестирования
python start_training_system.py --mode quick

# 3. Проверка статуса
python start_training_system.py --mode status

# 4. Генерация отчета
python start_training_system.py --mode report
```

### Пример 2: Полное обучение

```bash
# 1. Полное обучение всех моделей
python start_training_system.py --mode full

# 2. Мониторинг процесса
python manage_training.py monitor

# 3. Проверка результатов
python manage_training.py status
```

### Пример 3: Инкрементальное обучение

```bash
# 1. Инкрементальное обучение
python start_training_system.py --mode incremental

# 2. Продолжение обучения
python manage_training.py continue --project human_behavior --epochs 20

# 3. Генерация отчета
python manage_training.py report --project human_behavior --days 7
```

### Пример 4: Автоматическое обучение

```bash
# 1. Запуск планировщика
python start_training_system.py --mode schedule

# 2. Мониторинг в фоновом режиме
python manage_training.py monitor
```

## Заключение

Система обучения моделей ML предоставляет:

1. **Простота**: Легкий запуск одной командой
2. **Гибкость**: Различные режимы обучения
3. **Автоматизация**: Обучение по расписанию
4. **Мониторинг**: Отслеживание процесса
5. **Надежность**: Резервное копирование
6. **Масштабируемость**: Поддержка больших данных

Используйте систему для эффективного обучения и управления моделями машинного обучения.
