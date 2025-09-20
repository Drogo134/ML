# Быстрый старт

## Установка

```bash
# 1. Клонирование
git clone <repository-url>
cd "проекты ML"

# 2. Виртуальное окружение
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate  # Windows

# 3. Установка зависимостей
pip install -r requirements.txt

# 4. Создание директорий
python -c "
from human_behavior_prediction.config import Config as C1
from biochemistry_molecules.config import Config as C2
from small_ml_project.config import Config as C3
C1.create_directories(); C2.create_directories(); C3.create_directories()
"
```

## Запуск

### Все проекты
```bash
python run_all_projects.py
```

### Отдельные проекты
```bash
# Human Behavior Prediction
cd human_behavior_prediction && python main.py

# Molecular Property Prediction  
cd biochemistry_molecules && python main.py

# Small ML Project
cd small_ml_project && python main.py
```

## Результаты

- **Модели**: `*/models/`
- **Результаты**: `*/results/`
- **Логи**: `*/logs/`

## Проблемы

### RDKit не устанавливается
```bash
conda install -c conda-forge rdkit
```

### Нехватка памяти
Уменьшите `n_samples` в main.py

### Ошибки CUDA
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

## Поддержка

- Документация: README.md
- Установка: INSTALLATION.md
- Использование: USAGE_GUIDE.md
