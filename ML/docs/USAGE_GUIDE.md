# Руководство по использованию ML проектов

## Обзор проектов

Созданы три профессиональных проекта машинного обучения, каждый из которых демонстрирует современные подходы к ИИ, нейронным сетям и анализу данных:

### 1. 🧠 Human Behavior Prediction System
**Папка**: `human_behavior_prediction/`

**Назначение**: Прогнозирование человеческого поведения с использованием нейронных сетей и машинного обучения.

**Ключевые возможности**:
- Генерация синтетических поведенческих данных
- Расширяемая база данных с API интеграцией
- Множественные модели: XGBoost, LightGBM, нейронные сети
- Аугментация данных (SMOTE, ADASYN)
- Мониторинг производительности и дрифта данных
- Интерпретация моделей (SHAP, LIME)

### 2. 🧪 Molecular Property Prediction System
**Папка**: `biochemistry_molecules/`

**Назначение**: Предсказание свойств молекул с использованием графовых нейронных сетей.

**Ключевые возможности**:
- Графовые нейронные сети (GCN, GAT, Transformer)
- Молекулярные дескрипторы и отпечатки
- Интеграция с химическими базами данных (Tox21, BACE, BBBP)
- Визуализация молекулярных структур
- Оптимизация гиперпараметров

### 3. 🔬 Small ML Project - Comprehensive Pipeline
**Папка**: `small_ml_project/`

**Назначение**: Комплексная система для различных задач ML.

**Ключевые возможности**:
- Классификация, регрессия, кластеризация
- Автоматическая генерация данных
- Сравнение множественных алгоритмов
- Интерактивная визуализация

## Быстрый старт

### 1. Установка и настройка

```bash
# Клонирование репозитория
git clone <repository-url>
cd "проекты ML"

# Создание виртуального окружения
python -m venv venv
source venv/bin/activate  # Linux/Mac
# или
venv\Scripts\activate  # Windows

# Установка зависимостей
pip install -r requirements.txt

# Создание директорий
python -c "
from human_behavior_prediction.config import Config as Config1
from biochemistry_molecules.config import Config as Config2
from small_ml_project.config import Config as Config3
Config1.create_directories()
Config2.create_directories()
Config3.create_directories()
"
```

### 2. Запуск всех проектов

```bash
python run_all_projects.py
```

### 3. Запуск отдельных проектов

```bash
# Human Behavior Prediction
cd human_behavior_prediction
python main.py

# Molecular Property Prediction
cd biochemistry_molecules
python main.py

# Small ML Project
cd small_ml_project
python main.py
```

## Детальное использование

### Human Behavior Prediction System

#### Базовое использование
```python
from human_behavior_prediction.main import HumanBehaviorPredictionPipeline

# Создание пайплайна
pipeline = HumanBehaviorPredictionPipeline()

# Запуск с настройками по умолчанию
results = pipeline.run_full_pipeline(
    n_samples=10000,
    target_column='will_purchase'
)
```

#### Расширенное использование
```python
# Генерация собственных данных
from human_behavior_prediction.data_generator import HumanBehaviorDataGenerator

generator = HumanBehaviorDataGenerator()
custom_data = generator.generate_dataset(n_samples=50000)

# Аугментация данных
from human_behavior_prediction.data_augmentation import DataAugmentation

augmenter = DataAugmentation()
X_aug, y_aug = augmenter.augment_dataset(X, y, method='smote')

# API интеграция
from human_behavior_prediction.api_integration import APIIntegration

api = APIIntegration(config)
enriched_data = api.enrich_behavior_data(df, ['demographic', 'economic'])

# Мониторинг
from human_behavior_prediction.monitoring import ModelMonitoring

monitor = ModelMonitoring(config)
monitor.log_model_performance('xgboost', metrics)
```

#### Настройка параметров
```python
# В config.py
MODEL_PARAMS = {
    'xgboost': {
        'n_estimators': 1000,
        'max_depth': 6,
        'learning_rate': 0.1
    },
    'neural_network': {
        'hidden_layers': [128, 64, 32],
        'dropout_rate': 0.3,
        'epochs': 100
    }
}
```

### Molecular Property Prediction System

#### Базовое использование
```python
from biochemistry_molecules.main import MolecularPropertyPredictionPipeline

# Создание пайплайна
pipeline = MolecularPropertyPredictionPipeline()

# Запуск на датасете Tox21
results = pipeline.run_full_pipeline(
    dataset_name='tox21',
    target_column='NR-AR'
)
```

#### Работа с молекулярными данными
```python
# Загрузка данных
from biochemistry_molecules.molecular_data_loader import MolecularDataLoader

loader = MolecularDataLoader(config)
df = loader.load_dataset('bace')
processed_df = loader.process_dataset(df)

# Обучение графовых моделей
from biochemistry_molecules.graph_models import MolecularGraphTrainer

trainer = MolecularGraphTrainer(config)
model = trainer.train_model('gcn', train_loader, val_loader)

# Визуализация
from biochemistry_molecules.visualization import MolecularVisualization

viz = MolecularVisualization(config)
viz.plot_molecular_structures(smiles_list)
```

#### Доступные датасеты
- **Tox21**: Токсикологические данные (12 задач)
- **BACE**: Ингибиторы β-секретазы
- **BBBP**: Проницаемость гематоэнцефалического барьера
- **ClinTox**: Клиническая токсичность

### Small ML Project

#### Базовое использование
```python
from small_ml_project.main import MLPipeline

# Создание пайплайна
pipeline = MLPipeline()

# Классификация
results_clf = pipeline.run_pipeline(
    task_type='classification',
    n_samples=2000,
    n_features=15
)

# Регрессия
results_reg = pipeline.run_pipeline(
    task_type='regression',
    n_samples=2000,
    n_features=15
)
```

#### Генерация данных
```python
from small_ml_project.data_generator import DataGenerator

generator = DataGenerator(config)

# Классификация
clf_data = generator.create_dataset(
    task_type='classification',
    n_samples=5000,
    n_features=30,
    n_classes=3
)

# Регрессия
reg_data = generator.create_dataset(
    task_type='regression',
    n_samples=3000,
    n_features=25,
    noise=0.1
)

# Кластеризация
cluster_data = generator.create_dataset(
    task_type='clustering',
    n_samples=2000,
    n_features=2,
    centers=4
)
```

## Настройка и конфигурация

### Общие параметры
```python
# В config.py каждого проекта
RANDOM_STATE = 42              # Случайное состояние
TEST_SIZE = 0.2               # Размер тестовой выборки
VALIDATION_SIZE = 0.2         # Размер валидационной выборки
```

### Параметры моделей
```python
MODEL_PARAMS = {
    'xgboost': {
        'n_estimators': 1000,
        'max_depth': 6,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8
    },
    'neural_network': {
        'hidden_layers': [128, 64, 32],
        'dropout_rate': 0.3,
        'learning_rate': 0.001,
        'epochs': 100,
        'batch_size': 32
    }
}
```

## Мониторинг и логирование

### Просмотр логов
```bash
# Логи каждого проекта
tail -f human_behavior_prediction/logs/*.log
tail -f biochemistry_molecules/logs/*.log
tail -f small_ml_project/logs/*.log
```

### Мониторинг производительности
```python
# Human Behavior Prediction
from human_behavior_prediction.monitoring import ModelMonitoring

monitor = ModelMonitoring(config)
report = monitor.generate_monitoring_report(days=7)
```

## Результаты и визуализация

### Автоматически создаваемые файлы
- **Модели**: `models/` - сохраненные обученные модели
- **Результаты**: `results/` - метрики и графики
- **Логи**: `logs/` - логи обучения и выполнения

### Типы графиков
- Матрицы ошибок
- ROC-кривые
- Графики остатков
- Сравнение моделей
- Важность признаков
- Молекулярные структуры (для биохимии)

## Расширение и кастомизация

### Добавление новых моделей
```python
# В models.py
def train_custom_model(self, X_train, y_train, **kwargs):
    # Реализация новой модели
    model = CustomModel(**kwargs)
    model.fit(X_train, y_train)
    return model
```

### Добавление новых метрик
```python
# В evaluation.py
def custom_metric(self, y_true, y_pred):
    # Реализация новой метрики
    return custom_score
```

### Интеграция с внешними API
```python
# В api_integration.py
def fetch_custom_data(self, endpoint, params):
    # Реализация загрузки данных
    response = self.session.get(endpoint, params=params)
    return response.json()
```

## Производительность и оптимизация

### Оптимизация для CPU
```python
import os
os.environ['OMP_NUM_THREADS'] = '4'
os.environ['MKL_NUM_THREADS'] = '4'
```

### Оптимизация для GPU
```python
import torch
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
```

### Уменьшение использования памяти
```python
# В main.py уменьшите размеры датасетов
results = pipeline.run_full_pipeline(n_samples=1000)  # вместо 10000
```

## Устранение проблем

### Частые проблемы

1. **Ошибка установки RDKit**
   ```bash
   conda install -c conda-forge rdkit
   ```

2. **Ошибка с памятью**
   - Уменьшите `n_samples` в main.py
   - Увеличьте `batch_size` для нейронных сетей

3. **Ошибка с CUDA**
   - Установите CPU версию PyTorch
   - Или установите совместимую CUDA версию

4. **Ошибка с зависимостями**
   ```bash
   pip install --upgrade -r requirements.txt
   ```

### Проверка установки
```bash
python -c "
import numpy, pandas, sklearn, tensorflow, torch, xgboost, lightgbm
print('✅ Все основные библиотеки установлены')
"
```

## Поддержка

### Документация
- `README.md` - общая документация
- `INSTALLATION.md` - руководство по установке
- `USAGE_GUIDE.md` - данное руководство
- `*/README.md` - документация каждого проекта

### Логи и отладка
- Проверьте логи в папке `logs/`
- Убедитесь, что все зависимости установлены
- Проверьте версии Python и библиотек

### Сообщество
- Создайте issue в репозитории
- Обратитесь к документации библиотек
- Проверьте примеры использования

## Примеры использования

### Пример 1: Анализ поведения пользователей
```python
# Создание пайплайна
pipeline = HumanBehaviorPredictionPipeline()

# Генерация данных
pipeline.generate_data(n_samples=50000)

# Обучение моделей
results = pipeline.run_full_pipeline(target_column='will_purchase')

# Анализ результатов
print(f"Лучшая модель: {max(results, key=results.get)}")
```

### Пример 2: Предсказание токсичности молекул
```python
# Создание пайплайна
pipeline = MolecularPropertyPredictionPipeline()

# Загрузка данных Tox21
pipeline.load_and_process_data('tox21')

# Обучение графовых моделей
results = pipeline.run_full_pipeline(target_column='NR-AR')

# Визуализация результатов
pipeline.create_visualizations(X_test, y_test)
```

### Пример 3: Классификация данных
```python
# Создание пайплайна
pipeline = MLPipeline()

# Генерация данных
pipeline.generate_data(task_type='classification', n_samples=3000)

# Обучение моделей
results = pipeline.run_pipeline(task_type='classification')

# Сравнение моделей
pipeline.plot_model_comparison(task_type)
```

## Заключение

Эти проекты демонстрируют современные подходы к машинному обучению и нейронным сетям. Каждый проект решает уникальные задачи и может быть использован как основа для более сложных систем.

Для получения дополнительной информации обратитесь к документации каждого проекта или создайте issue в репозитории.
