# Руководство по тестированию ML проектов

## Обзор тестирования

Данное руководство описывает процесс тестирования всех трех проектов машинного обучения для обеспечения их корректной работы и качества.

## Предварительные требования

### 1. Установка зависимостей
```bash
pip install -r requirements.txt
pip install pytest pytest-cov
```

### 2. Создание тестовых директорий
```bash
mkdir -p tests/{unit,integration,e2e}
```

## Типы тестов

### 1. Unit тесты
Тестирование отдельных компонентов и функций.

### 2. Integration тесты
Тестирование взаимодействия между компонентами.

### 3. End-to-End тесты
Тестирование полного пайплайна.

## Тестирование Human Behavior Prediction

### Unit тесты

#### Тест генерации данных
```python
# tests/unit/test_behavior_data_generator.py
import pytest
import numpy as np
from human_behavior_prediction.data_generator import HumanBehaviorDataGenerator

def test_data_generation():
    generator = HumanBehaviorDataGenerator()
    data = generator.generate_dataset(n_samples=1000)
    
    assert len(data) == 1000
    assert 'age' in data.columns
    assert 'gender' in data.columns
    assert 'will_purchase' in data.columns
    assert data['age'].min() >= 0
    assert data['age'].max() <= 100

def test_demographic_features():
    generator = HumanBehaviorDataGenerator()
    features = generator.generate_demographic_features(100)
    
    assert len(features['age']) == 100
    assert all(age >= 0 for age in features['age'])
    assert all(gender in ['Male', 'Female', 'Other'] for gender in features['gender'])
```

#### Тест инженерии признаков
```python
# tests/unit/test_behavior_feature_engineering.py
import pytest
import pandas as pd
from human_behavior_prediction.feature_engineering import FeatureEngineer
from human_behavior_prediction.config import Config

def test_feature_engineering():
    config = Config()
    engineer = FeatureEngineer(config)
    
    # Создаем тестовые данные
    data = pd.DataFrame({
        'age': [25, 30, 35],
        'income': [50000, 60000, 70000],
        'session_duration': [100, 200, 300],
        'will_purchase': [0, 1, 0]
    })
    
    processed_data = engineer.engineer_features(data, target_col='will_purchase')
    
    assert len(processed_data) == 3
    assert 'age_income_interaction' in processed_data.columns
    assert 'session_click_interaction' in processed_data.columns
```

#### Тест моделей
```python
# tests/unit/test_behavior_models.py
import pytest
import numpy as np
from human_behavior_prediction.models import ModelTrainer
from human_behavior_prediction.config import Config

def test_xgboost_training():
    config = Config()
    trainer = ModelTrainer(config)
    
    # Создаем тестовые данные
    X = np.random.rand(100, 10)
    y = np.random.randint(0, 2, 100)
    
    X_train, X_val, X_test, y_train, y_val, y_test = trainer.prepare_data(X, y)
    
    model = trainer.train_xgboost(X_train, y_train, X_val, y_val)
    
    assert model is not None
    assert hasattr(model, 'predict')
    assert hasattr(model, 'predict_proba')
```

### Integration тесты

#### Тест полного пайплайна
```python
# tests/integration/test_behavior_pipeline.py
import pytest
from human_behavior_prediction.main import HumanBehaviorPredictionPipeline

def test_full_pipeline():
    pipeline = HumanBehaviorPredictionPipeline()
    
    # Генерируем небольшой датасет для тестирования
    data = pipeline.generate_data(n_samples=1000, save_data=False)
    
    assert len(data) > 0
    assert 'will_purchase' in data.columns
    
    # Тестируем подготовку признаков
    X, y = pipeline.prepare_features('will_purchase')
    
    assert len(X) == len(y)
    assert len(X.columns) > 0

def test_model_training():
    pipeline = HumanBehaviorPredictionPipeline()
    
    # Генерируем данные
    pipeline.generate_data(n_samples=1000, save_data=False)
    X, y = pipeline.prepare_features('will_purchase')
    
    # Тестируем обучение моделей
    X_test, y_test = pipeline.train_models(X, y)
    
    assert len(pipeline.model_trainer.models) > 0
    assert len(pipeline.results) > 0
```

## Тестирование Molecular Property Prediction

### Unit тесты

#### Тест загрузки данных
```python
# tests/unit/test_molecular_data_loader.py
import pytest
from biochemistry_molecules.molecular_data_loader import MolecularDataLoader
from biochemistry_molecules.config import Config

def test_molecular_descriptors():
    config = Config()
    loader = MolecularDataLoader(config)
    
    # Тестируем вычисление дескрипторов
    smiles = "CCO"  # Этанол
    descriptors = loader.calculate_molecular_descriptors(smiles)
    
    assert 'MW' in descriptors
    assert 'LogP' in descriptors
    assert descriptors['MW'] > 0
    assert isinstance(descriptors['LogP'], float)

def test_fingerprints():
    config = Config()
    loader = MolecularDataLoader(config)
    
    smiles = "CCO"
    fingerprints = loader.calculate_fingerprints(smiles)
    
    assert 'Morgan' in fingerprints
    assert 'MACCS' in fingerprints
    assert len(fingerprints['Morgan']) == 2048
    assert len(fingerprints['MACCS']) == 167
```

#### Тест графовых моделей
```python
# tests/unit/test_molecular_graph_models.py
import pytest
import torch
from biochemistry_molecules.graph_models import GCNModel, GATModel
from biochemistry_molecules.config import Config

def test_gcn_model():
    config = Config()
    
    model = GCNModel(
        input_dim=7,
        hidden_dim=64,
        output_dim=1,
        num_layers=3
    )
    
    # Тестовые данные
    x = torch.randn(10, 7)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)
    batch = torch.zeros(10, dtype=torch.long)
    
    output = model(x, edge_index, batch)
    
    assert output.shape == (1, 1)  # Один граф, один выход

def test_gat_model():
    config = Config()
    
    model = GATModel(
        input_dim=7,
        hidden_dim=64,
        output_dim=1,
        num_heads=8,
        num_layers=3
    )
    
    # Тестовые данные
    x = torch.randn(10, 7)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)
    batch = torch.zeros(10, dtype=torch.long)
    
    output = model(x, edge_index, batch)
    
    assert output.shape == (1, 1)
```

### Integration тесты

#### Тест молекулярного пайплайна
```python
# tests/integration/test_molecular_pipeline.py
import pytest
from biochemistry_molecules.main import MolecularPropertyPredictionPipeline

def test_molecular_pipeline():
    pipeline = MolecularPropertyPredictionPipeline()
    
    # Тестируем загрузку данных
    data = pipeline.load_and_process_data('tox21')
    
    assert len(data) > 0
    assert 'smiles' in data.columns or 'smiles_clean' in data.columns
    
    # Тестируем подготовку признаков
    X_desc, y_desc = pipeline.prepare_traditional_features('NR-AR')
    
    assert len(X_desc) == len(y_desc)
    assert X_desc.shape[1] > 0
```

## Тестирование Small ML Project

### Unit тесты

#### Тест генерации данных
```python
# tests/unit/test_small_ml_data_generator.py
import pytest
import numpy as np
from small_ml_project.data_generator import DataGenerator
from small_ml_project.config import Config

def test_classification_data():
    config = Config()
    generator = DataGenerator(config)
    
    data = generator.create_dataset(
        task_type='classification',
        n_samples=1000,
        n_features=10,
        n_classes=2
    )
    
    assert len(data) == 1000
    assert data['target'].nunique() == 2
    assert data['target'].min() >= 0
    assert data['target'].max() <= 1

def test_regression_data():
    config = Config()
    generator = DataGenerator(config)
    
    data = generator.create_dataset(
        task_type='regression',
        n_samples=1000,
        n_features=10
    )
    
    assert len(data) == 1000
    assert data['target'].dtype in [np.float64, np.float32]
    assert not data['target'].isna().any()

def test_clustering_data():
    config = Config()
    generator = DataGenerator(config)
    
    data = generator.create_dataset(
        task_type='clustering',
        n_samples=1000,
        n_features=2,
        centers=3
    )
    
    assert len(data) == 1000
    assert data['target'].nunique() == 3
```

#### Тест моделей
```python
# tests/unit/test_small_ml_models.py
import pytest
import numpy as np
from small_ml_project.models import ModelTrainer
from small_ml_project.config import Config

def test_random_forest():
    config = Config()
    trainer = ModelTrainer(config)
    
    X = np.random.rand(100, 10)
    y = np.random.randint(0, 2, 100)
    
    X_train, X_val, X_test, y_train, y_val, y_test = trainer.prepare_data(X, y)
    
    model = trainer.train_random_forest(X_train, y_train)
    
    assert model is not None
    assert hasattr(model, 'predict')
    assert hasattr(model, 'predict_proba')

def test_neural_network():
    config = Config()
    trainer = ModelTrainer(config)
    
    X = np.random.rand(100, 10)
    y = np.random.randint(0, 2, 100)
    
    X_train, X_val, X_test, y_train, y_val, y_test = trainer.prepare_data(X, y)
    
    model = trainer.train_neural_network(X_train, y_train, X_val, y_val)
    
    assert model is not None
    assert hasattr(model, 'predict')
```

### Integration тесты

#### Тест ML пайплайна
```python
# tests/integration/test_small_ml_pipeline.py
import pytest
from small_ml_project.main import MLPipeline

def test_classification_pipeline():
    pipeline = MLPipeline()
    
    # Тестируем генерацию данных
    data = pipeline.generate_data(
        task_type='classification',
        n_samples=1000,
        n_features=10
    )
    
    assert len(data) > 0
    assert 'target' in data.columns
    
    # Тестируем предобработку
    X, y = pipeline.preprocess_data()
    
    assert len(X) == len(y)
    assert X.shape[1] > 0

def test_regression_pipeline():
    pipeline = MLPipeline()
    
    data = pipeline.generate_data(
        task_type='regression',
        n_samples=1000,
        n_features=10
    )
    
    assert len(data) > 0
    
    X, y = pipeline.preprocess_data()
    
    assert len(X) == len(y)
    assert y.dtype in [np.float64, np.float32]
```

## End-to-End тесты

### Тест всех проектов
```python
# tests/e2e/test_all_projects.py
import pytest
import subprocess
import sys

def test_run_all_projects():
    """Тест запуска всех проектов"""
    result = subprocess.run(
        [sys.executable, 'run_all_projects.py'],
        capture_output=True,
        text=True,
        timeout=300  # 5 минут
    )
    
    assert result.returncode == 0
    assert 'Pipeline completed successfully' in result.stdout

def test_individual_projects():
    """Тест запуска отдельных проектов"""
    
    # Human Behavior Prediction
    result = subprocess.run(
        [sys.executable, 'human_behavior_prediction/main.py'],
        capture_output=True,
        text=True,
        timeout=120
    )
    
    assert result.returncode == 0
    
    # Molecular Property Prediction
    result = subprocess.run(
        [sys.executable, 'biochemistry_molecules/main.py'],
        capture_output=True,
        text=True,
        timeout=120
    )
    
    assert result.returncode == 0
    
    # Small ML Project
    result = subprocess.run(
        [sys.executable, 'small_ml_project/main.py'],
        capture_output=True,
        text=True,
        timeout=120
    )
    
    assert result.returncode == 0
```

## Тестирование производительности

### Тест времени выполнения
```python
# tests/performance/test_performance.py
import pytest
import time
from human_behavior_prediction.main import HumanBehaviorPredictionPipeline

def test_behavior_prediction_performance():
    pipeline = HumanBehaviorPredictionPipeline()
    
    start_time = time.time()
    
    # Генерируем данные
    pipeline.generate_data(n_samples=1000, save_data=False)
    
    # Обучаем модели
    X, y = pipeline.prepare_features('will_purchase')
    pipeline.train_models(X, y)
    
    end_time = time.time()
    execution_time = end_time - start_time
    
    # Проверяем, что выполнение заняло не более 5 минут
    assert execution_time < 300
    
    # Проверяем, что модели обучены
    assert len(pipeline.model_trainer.models) > 0
```

### Тест использования памяти
```python
# tests/performance/test_memory.py
import pytest
import psutil
import os
from human_behavior_prediction.main import HumanBehaviorPredictionPipeline

def test_memory_usage():
    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    pipeline = HumanBehaviorPredictionPipeline()
    pipeline.generate_data(n_samples=5000, save_data=False)
    
    final_memory = process.memory_info().rss / 1024 / 1024  # MB
    memory_increase = final_memory - initial_memory
    
    # Проверяем, что использование памяти не превышает 2 GB
    assert memory_increase < 2048
```

## Тестирование качества данных

### Тест качества сгенерированных данных
```python
# tests/quality/test_data_quality.py
import pytest
import pandas as pd
from human_behavior_prediction.data_generator import HumanBehaviorDataGenerator

def test_data_quality():
    generator = HumanBehaviorDataGenerator()
    data = generator.generate_dataset(n_samples=1000)
    
    # Проверяем отсутствие пропусков
    assert data.isnull().sum().sum() == 0
    
    # Проверяем корректность типов данных
    assert data['age'].dtype in ['int64', 'int32']
    assert data['gender'].dtype == 'object'
    assert data['will_purchase'].dtype in ['int64', 'int32']
    
    # Проверяем диапазоны значений
    assert data['age'].min() >= 0
    assert data['age'].max() <= 100
    assert data['will_purchase'].min() >= 0
    assert data['will_purchase'].max() <= 1
    
    # Проверяем уникальность значений
    assert data['gender'].nunique() == 3
    assert data['will_purchase'].nunique() == 2
```

## Запуск тестов

### Установка pytest
```bash
pip install pytest pytest-cov pytest-xdist
```

### Запуск всех тестов
```bash
pytest tests/ -v
```

### Запуск с покрытием
```bash
pytest tests/ --cov=. --cov-report=html
```

### Запуск параллельно
```bash
pytest tests/ -n auto
```

### Запуск конкретных тестов
```bash
# Unit тесты
pytest tests/unit/ -v

# Integration тесты
pytest tests/integration/ -v

# E2E тесты
pytest tests/e2e/ -v

# Тесты конкретного проекта
pytest tests/unit/test_behavior_*.py -v
```

## Непрерывная интеграция

### GitHub Actions
```yaml
# .github/workflows/test.yml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.8, 3.9, 3.10]
    
    steps:
    - uses: actions/checkout@v2
    
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v2
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install pytest pytest-cov
    
    - name: Run tests
      run: |
        pytest tests/ --cov=. --cov-report=xml
    
    - name: Upload coverage
      uses: codecov/codecov-action@v1
```

## Мониторинг тестов

### Логирование
```python
# tests/conftest.py
import logging
import pytest

@pytest.fixture(autouse=True)
def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
```

### Отчеты
```bash
# Генерация HTML отчета
pytest tests/ --html=report.html --self-contained-html

# Генерация JUnit XML
pytest tests/ --junitxml=report.xml
```

## Заключение

Данное руководство по тестированию обеспечивает:

1. **Полное покрытие**: Unit, integration и E2E тесты
2. **Качество кода**: Проверка корректности и производительности
3. **Надежность**: Автоматическое обнаружение проблем
4. **Документацию**: Примеры использования компонентов
5. **CI/CD**: Интеграция с системами непрерывной интеграции

Регулярное выполнение тестов гарантирует стабильность и качество всех проектов машинного обучения.
