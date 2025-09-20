# Installation Guide

## Системные требования

### Минимальные требования
- **OS**: Windows 10, macOS 10.14+, Ubuntu 18.04+
- **Python**: 3.8 или выше
- **RAM**: 8 GB
- **CPU**: 4 ядра
- **Диск**: 10 GB свободного места

### Рекомендуемые требования
- **OS**: Windows 11, macOS 12+, Ubuntu 20.04+
- **Python**: 3.9 или выше
- **RAM**: 16 GB
- **CPU**: 8 ядер
- **GPU**: NVIDIA GTX 1060+ (для ускорения нейронных сетей)
- **Диск**: 20 GB свободного места

## Установка

### 1. Клонирование репозитория

```bash
git clone <repository-url>
cd "проекты ML"
```

### 2. Создание виртуального окружения

#### Windows
```bash
python -m venv venv
venv\Scripts\activate
```

#### macOS/Linux
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Обновление pip

```bash
python -m pip install --upgrade pip
```

### 4. Установка зависимостей

#### Установка всех зависимостей
```bash
pip install -r requirements.txt
```

#### Установка по проектам
```bash
# Human Behavior Prediction
cd human_behavior_prediction
pip install -r requirements.txt
cd ..

# Molecular Property Prediction
cd biochemistry_molecules
pip install -r requirements.txt
cd ..

# Small ML Project
cd small_ml_project
pip install -r requirements.txt
cd ..
```

### 5. Установка дополнительных зависимостей

#### Для Windows
```bash
# Установка Visual C++ Build Tools (если нужно)
# Скачайте с https://visualstudio.microsoft.com/visual-cpp-build-tools/

# Установка RDKit (если возникают проблемы)
conda install -c conda-forge rdkit
```

#### Для macOS
```bash
# Установка Xcode Command Line Tools
xcode-select --install

# Установка RDKit через conda
conda install -c conda-forge rdkit
```

#### Для Ubuntu/Debian
```bash
# Установка системных зависимостей
sudo apt-get update
sudo apt-get install python3-dev python3-pip
sudo apt-get install libxrender1 libxext6 libsm6 libglib2.0-0

# Установка RDKit
sudo apt-get install python3-rdkit
```

### 6. Создание необходимых директорий

```bash
python -c "
from human_behavior_prediction.config import Config as Config1
from biochemistry_molecules.config import Config as Config2
from small_ml_project.config import Config as Config3

Config1.create_directories()
Config2.create_directories()
Config3.create_directories()
print('Directories created successfully')
"
```

## Проверка установки

### 1. Проверка Python и основных библиотек

```bash
python -c "
import sys
print(f'Python version: {sys.version}')

import numpy as np
print(f'NumPy version: {np.__version__}')

import pandas as pd
print(f'Pandas version: {pd.__version__}')

import sklearn
print(f'Scikit-learn version: {sklearn.__version__}')

print('✅ Core libraries installed successfully')
"
```

### 2. Проверка TensorFlow

```bash
python -c "
import tensorflow as tf
print(f'TensorFlow version: {tf.__version__}')
print(f'GPU available: {tf.config.list_physical_devices("GPU")}')
print('✅ TensorFlow installed successfully')
"
```

### 3. Проверка PyTorch

```bash
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print('✅ PyTorch installed successfully')
"
```

### 4. Проверка специализированных библиотек

```bash
python -c "
try:
    import xgboost as xgb
    print(f'XGBoost version: {xgb.__version__}')
except ImportError:
    print('❌ XGBoost not installed')

try:
    import lightgbm as lgb
    print(f'LightGBM version: {lgb.__version__}')
except ImportError:
    print('❌ LightGBM not installed')

try:
    import rdkit
    print(f'RDKit version: {rdkit.__version__}')
except ImportError:
    print('❌ RDKit not installed')

print('✅ Specialized libraries check completed')
"
```

## Запуск проектов

### 1. Запуск всех проектов

```bash
python run_all_projects.py
```

### 2. Запуск отдельных проектов

#### Human Behavior Prediction
```bash
cd human_behavior_prediction
python main.py
```

#### Molecular Property Prediction
```bash
cd biochemistry_molecules
python main.py
```

#### Small ML Project
```bash
cd small_ml_project
python main.py
```

## Устранение проблем

### Проблема: Ошибка установки RDKit

**Решение:**
```bash
# Используйте conda вместо pip
conda install -c conda-forge rdkit

# Или установите через conda-forge
conda create -n ml_env python=3.9
conda activate ml_env
conda install -c conda-forge rdkit
```

### Проблема: Ошибка установки PyTorch Geometric

**Решение:**
```bash
# Установите PyTorch сначала
pip install torch torchvision torchaudio

# Затем установите PyTorch Geometric
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-2.0.0+cpu.html
```

### Проблема: Ошибка с CUDA

**Решение:**
```bash
# Установите CPU версию
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Или установите CUDA версию (если есть GPU)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Проблема: Ошибка с TensorFlow

**Решение:**
```bash
# Установите совместимую версию
pip install tensorflow==2.13.0

# Или для GPU
pip install tensorflow[and-cuda]==2.13.0
```

### Проблема: Ошибка с памятью

**Решение:**
```python
# В config.py уменьшите размеры датасетов
RANDOM_STATE = 42
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.2

# Уменьшите n_samples в main.py
results = pipeline.run_full_pipeline(n_samples=1000)  # вместо 10000
```

## Настройка для разработки

### 1. Установка дополнительных инструментов разработки

```bash
pip install jupyter notebook ipykernel
pip install black flake8 pytest
pip install pre-commit
```

### 2. Настройка Jupyter

```bash
python -m ipykernel install --user --name=ml_projects
```

### 3. Настройка pre-commit

```bash
pre-commit install
```

## Производительность

### Оптимизация для CPU

```python
# В config.py
import os
os.environ['OMP_NUM_THREADS'] = '4'
os.environ['MKL_NUM_THREADS'] = '4'
```

### Оптимизация для GPU

```python
# В config.py
import torch
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
```

## Мониторинг

### Проверка использования ресурсов

```bash
# CPU и память
htop  # Linux/macOS
taskmgr  # Windows

# GPU (если используется)
nvidia-smi
```

### Логи

```bash
# Просмотр логов
tail -f human_behavior_prediction/logs/*.log
tail -f biochemistry_molecules/logs/*.log
tail -f small_ml_project/logs/*.log
```

## Поддержка

Если у вас возникли проблемы:

1. Проверьте логи в папке `logs/`
2. Убедитесь, что все зависимости установлены
3. Проверьте версии Python и библиотек
4. Создайте issue в репозитории с описанием проблемы

## Обновление

### Обновление зависимостей

```bash
pip install --upgrade -r requirements.txt
```

### Обновление проекта

```bash
git pull origin main
pip install --upgrade -r requirements.txt
```
