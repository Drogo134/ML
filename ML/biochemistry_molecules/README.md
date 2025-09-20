# Molecular Property Prediction System

## Описание проекта

Система предсказания свойств молекул на основе графовых нейронных сетей и методов машинного обучения. Проект использует современные достижения в области химической информатики, глубокого обучения и графовых нейронных сетей для анализа молекулярных структур и предсказания их биологических и физико-химических свойств.

## Функции проекта

### Основные возможности
- Предсказание токсикологических свойств молекул
- Анализ биологической активности соединений
- Оценка ADMET свойств (Absorption, Distribution, Metabolism, Excretion, Toxicity)
- Классификация молекул по функциональным группам
- Регрессионный анализ молекулярных дескрипторов

### Дополнительные функции
- Визуализация молекулярных структур и графов
- Интеграция с химическими базами данных (ChEMBL, PubChem)
- Автоматическая генерация молекулярных дескрипторов
- Сравнение различных архитектур нейронных сетей
- AI-анализ результатов и создание отчетов

## Технологический стек

### Графовые нейронные сети
- **PyTorch Geometric 2.3** - графовые нейронные сети
- **DGL 1.1.1** - глубокие графовые библиотеки
- **Graph Convolutional Networks (GCN)** - сверточные сети на графах
- **Graph Attention Networks (GAT)** - сети с механизмом внимания
- **Graph Transformer** - трансформеры для графов

### Химическая информатика
- **RDKit 2023.3** - химическая информатика
- **DeepChem 2.7** - глубокое обучение для химии
- **Mordred 1.2.0** - молекулярные дескрипторы
- **MolSets 0.1.0** - наборы молекул
- **ChEMBL Webresource Client 0.10.8** - доступ к ChEMBL
- **PubChemPy 1.0.4** - доступ к PubChem
- **MoleculeKit 1.6.6** - работа с молекулами

### Машинное обучение
- **PyTorch 2.0.1** - глубокое обучение
- **TensorFlow 2.13** - альтернативный фреймворк
- **XGBoost 1.7.6** - градиентный бустинг
- **LightGBM 4.0.0** - быстрый градиентный бустинг
- **Scikit-learn 1.3.0** - традиционные алгоритмы

### Обработка данных
- **Pandas 2.0.3** - анализ данных
- **NumPy 1.24.3** - численные вычисления
- **Feature-engine 1.6.2** - инженерия признаков
- **Imbalanced-learn 0.11.0** - работа с несбалансированными данными

### Визуализация
- **Matplotlib 3.7.2** - базовые графики
- **Seaborn 0.12.2** - статистическая визуализация
- **Plotly 5.15.0** - интерактивные графики
- **RDKit Visualization** - молекулярные структуры

### AI интеграция
- **OpenAI API** - GPT-3.5, GPT-4 для анализа
- **Anthropic API** - Claude для генерации отчетов
- **Google Gemini API** - альтернативный AI сервис

## Архитектура проекта

```
biochemistry_molecules/
├── config.py                 # Конфигурация проекта
├── molecular_data_loader.py  # Загрузка молекулярных данных
├── graph_models.py           # Графовые нейронные сети
├── main.py                   # Основной пайплайн
├── visualization.py          # Визуализация результатов
├── optimization.py           # Оптимизация моделей
├── requirements.txt          # Зависимости проекта
├── data/                     # Молекулярные данные
├── models/                   # Сохраненные модели
├── results/                  # Результаты и графики
└── logs/                     # Логи системы
```

## Установка и настройка

### Системные требования
- Python 3.8 или выше
- 16+ GB RAM (рекомендуется 32+ GB)
- 8+ CPU ядер (рекомендуется 16+ ядер)
- NVIDIA GPU с 8+ GB VRAM (рекомендуется)
- Windows 10+, macOS 10.14+, Ubuntu 18.04+

### Установка зависимостей
```bash
# Создание виртуального окружения
python -m venv venv
source venv/bin/activate  # Linux/Mac
# или
venv\Scripts\activate  # Windows

# Установка зависимостей
pip install -r requirements.txt

# Установка PyTorch с поддержкой CUDA (опционально)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Настройка AI интеграции (опционально)
```bash
# Настройка API ключей
python ../setup_ai_api.py

# Тестирование подключения
python ../test_ai_integration.py
```

## Использование

### Быстрый старт
```bash
# Запуск полного пайплайна
python main.py

# Или через общий скрипт
python ../quick_train.py
```

### Программное использование

#### Создание экземпляра пайплайна
```python
from main import MolecularPropertyPredictionPipeline

pipeline = MolecularPropertyPredictionPipeline()
```

#### Загрузка молекулярных данных
```python
# Загрузка данных Tox21
pipeline.load_and_process_data('tox21')

# Загрузка данных BACE
pipeline.load_and_process_data('bace')

# Загрузка данных BBBP
pipeline.load_and_process_data('bbbp')
```

#### Подготовка признаков
```python
# Традиционные молекулярные дескрипторы
X_desc, y_desc = pipeline.prepare_traditional_features('NR-AR')

# Молекулярные отпечатки
X_fp, y_fp = pipeline.prepare_fingerprint_features('NR-AR')

# Графовые данные
graphs, y_graph = pipeline.prepare_graph_data('NR-AR')
```

#### Обучение моделей
```python
# Обучение традиционных моделей
pipeline.train_traditional_models(X_desc, y_desc, 'classification')

# Обучение графовых моделей
pipeline.train_graph_models(graphs, y_graph, 'classification')
```

#### Визуализация результатов
```python
# Визуализация молекулярных структур
pipeline.visualize_molecules(smiles_list, properties)

# Визуализация графов
pipeline.visualize_graphs(graphs, labels)

# Визуализация результатов
pipeline.visualize_results(results)
```

### Параметры конфигурации

#### Основные параметры
- `dataset_name` - название датасета ('tox21', 'bace', 'bbbp', 'clintox')
- `task_type` - тип задачи ('classification', 'regression')
- `test_size` - доля тестовых данных (по умолчанию: 0.2)
- `random_state` - случайное состояние для воспроизводимости

#### Параметры графовых моделей
- `gcn_hidden_dim` - размер скрытого слоя GCN
- `gcn_num_layers` - количество слоев GCN
- `gat_hidden_dim` - размер скрытого слоя GAT
- `gat_num_heads` - количество голов внимания
- `transformer_hidden_dim` - размер скрытого слоя Transformer

## Модели машинного обучения

### Графовые нейронные сети

#### Graph Convolutional Network (GCN)
- **Архитектура**: Входной слой → GCN слои → Выходной слой
- **Активация**: ReLU, Dropout
- **Оптимизатор**: Adam
- **Функция потерь**: CrossEntropyLoss / MSELoss

#### Graph Attention Network (GAT)
- **Архитектура**: Входной слой → GAT слои → Выходной слой
- **Механизм внимания**: Multi-head attention
- **Активация**: LeakyReLU, Dropout
- **Оптимизатор**: Adam

#### Graph Transformer
- **Архитектура**: Входной слой → Transformer блоки → Выходной слой
- **Позиционное кодирование**: Sinusoidal encoding
- **Нормализация**: Layer Normalization
- **Оптимизатор**: AdamW

### Традиционные модели
- **Random Forest** - случайный лес
- **XGBoost** - градиентный бустинг
- **LightGBM** - быстрый градиентный бустинг
- **SVM** - машина опорных векторов
- **Neural Network** - многослойный перцептрон

## Молекулярные дескрипторы

### Физико-химические свойства
- **Molecular Weight** - молекулярный вес
- **LogP** - липофильность
- **HBD/HBA** - доноры/акцепторы водорода
- **TPSA** - топологическая полярная поверхность
- **Rotatable Bonds** - вращающиеся связи

### Молекулярные отпечатки
- **Morgan Fingerprints** - отпечатки Моргана
- **MACCS Keys** - MACCS ключи
- **RDKit Fingerprints** - отпечатки RDKit
- **ECFP** - Extended Connectivity Fingerprints

### Графовые представления
- **Node Features** - признаки узлов (атомы)
- **Edge Features** - признаки ребер (связи)
- **Graph Features** - признаки графа (молекула)

## Химические базы данных

### Tox21
- **Описание**: Токсикологические данные
- **Соединений**: ~8,000
- **Задач**: 12 задач классификации
- **Источник**: NIH

### BACE
- **Описание**: Ингибиторы β-секретазы
- **Соединений**: ~1,500
- **Задач**: 1 задача регрессии
- **Источник**: ChEMBL

### BBBP
- **Описание**: Проницаемость гематоэнцефалического барьера
- **Соединений**: ~2,000
- **Задач**: 1 задача классификации
- **Источник**: ChEMBL

### ClinTox
- **Описание**: Клиническая токсичность
- **Соединений**: ~1,500
- **Задач**: 2 задачи классификации
- **Источник**: FDA

## Оценка производительности

### Метрики классификации
- **Accuracy** - общая точность
- **Precision** - точность положительных предсказаний
- **Recall** - полнота положительных случаев
- **F1-Score** - гармоническое среднее
- **AUC-ROC** - площадь под ROC кривой
- **AUC-PR** - площадь под Precision-Recall кривой

### Метрики регрессии
- **MSE** - среднеквадратичная ошибка
- **MAE** - средняя абсолютная ошибка
- **RMSE** - корень из среднеквадратичной ошибки
- **R²** - коэффициент детерминации
- **Pearson R** - корреляция Пирсона

### Специализированные метрики
- **Enrichment Factor** - фактор обогащения
- **Hit Rate** - частота попаданий
- **Specificity** - специфичность
- **Sensitivity** - чувствительность

## Визуализация

### Молекулярные структуры
- **2D структуры** - плоские представления
- **3D структуры** - объемные модели
- **Графы** - узлы и ребра
- **Дескрипторы** - цветовое кодирование

### Результаты анализа
- **Confusion Matrix** - матрица ошибок
- **ROC Curves** - ROC кривые
- **Feature Importance** - важность признаков
- **Learning Curves** - кривые обучения

### Интерактивные дашборды
- **Plotly Dash** - веб-интерфейс
- **Jupyter Widgets** - интерактивные виджеты
- **Bokeh** - интерактивная визуализация

## Оптимизация моделей

### Гиперпараметры
- **Grid Search** - сеточный поиск
- **Random Search** - случайный поиск
- **Bayesian Optimization** - байесовская оптимизация
- **Optuna** - автоматическая оптимизация

### Архитектурный поиск
- **Neural Architecture Search (NAS)** - поиск архитектуры
- **AutoML** - автоматическое ML
- **Transfer Learning** - трансферное обучение
- **Ensemble Methods** - ансамблевые методы

## Мониторинг и обслуживание

### Отслеживание производительности
- **Model Performance** - производительность моделей
- **Data Drift** - дрифт данных
- **Concept Drift** - дрифт концепций
- **Model Degradation** - деградация моделей

### Логирование
- **Training Logs** - логи обучения
- **Prediction Logs** - логи предсказаний
- **Error Logs** - логи ошибок
- **Performance Logs** - логи производительности

## API интеграция

### Внешние API
- **ChEMBL API** - химические данные
- **PubChem API** - молекулярные данные
- **NIH API** - токсикологические данные
- **AI сервисы** - OpenAI, Anthropic, Google

### Внутренний API
- **REST API** - HTTP интерфейс
- **GraphQL** - гибкий API
- **WebSocket** - реальное время
- **Batch Processing** - пакетная обработка

## Развертывание

### Локальное развертывание
```bash
# Запуск локального сервера
python api_server.py

# Тестирование API
python test_api.py
```

### Docker развертывание
```bash
# Сборка образа
docker build -t molecular-prediction .

# Запуск контейнера
docker run -p 8000:8000 molecular-prediction
```

### Kubernetes развертывание
```bash
# Применение манифестов
kubectl apply -f ../k8s/

# Проверка статуса
kubectl get pods
```

## Примеры использования

### Базовый пример
```python
from main import MolecularPropertyPredictionPipeline

# Создание пайплайна
pipeline = MolecularPropertyPredictionPipeline()

# Загрузка данных
pipeline.load_and_process_data('tox21')

# Подготовка признаков
X_desc, y_desc = pipeline.prepare_traditional_features('NR-AR')
graphs, y_graph = pipeline.prepare_graph_data('NR-AR')

# Обучение моделей
pipeline.train_traditional_models(X_desc, y_desc, 'classification')
pipeline.train_graph_models(graphs, y_graph, 'classification')
```

### Продвинутый пример с визуализацией
```python
# Визуализация молекулярных структур
smiles_list = ['CCO', 'CCN', 'CCC']
properties = [0.1, 0.5, 0.8]
pipeline.visualize_molecules(smiles_list, properties)

# Визуализация графов
pipeline.visualize_graphs(graphs[:5], y_graph[:5])

# Создание интерактивного дашборда
pipeline.create_dashboard(results)
```

### Пример с AI интеграцией
```python
# AI анализ молекулярных данных
ai_analysis = pipeline.enhance_molecular_prediction(molecular_data, predictions)

# Генерация синтетических молекулярных данных
synthetic_data = pipeline.ai_enhancer.generate_synthetic_training_data(
    'molecular_data', n_samples=1000
)

# Создание AI отчета
report = pipeline.create_ai_report(results)
```

## Устранение неполадок

### Частые проблемы
1. **Ошибка памяти GPU** - уменьшите batch_size или используйте CPU
2. **Медленное обучение** - используйте более мощную GPU или уменьшите размер модели
3. **Низкая точность** - проверьте качество данных и настройте гиперпараметры
4. **Ошибки RDKit** - проверьте корректность SMILES строк

### Логи и отладка
```bash
# Просмотр логов
tail -f logs/training.log

# Отладка моделей
python -c "from main import MolecularPropertyPredictionPipeline; pipeline = MolecularPropertyPredictionPipeline(); pipeline.debug_mode = True"
```

## Лицензия

Проект распространяется под лицензией MIT. См. файл LICENSE для подробностей.

## Контакты

Для вопросов и предложений обращайтесь к разработчикам проекта.

## Changelog

### Версия 1.0.0
- Первоначальный релиз
- Поддержка графовых нейронных сетей
- Интеграция с химическими базами данных
- AI интеграция для анализа
- Система визуализации
- Docker и Kubernetes поддержка