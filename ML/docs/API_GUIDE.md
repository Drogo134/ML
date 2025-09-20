# API Guide для ML проектов

## Обзор API

Каждый проект предоставляет REST API для взаимодействия с моделями машинного обучения.

## Human Behavior Prediction API

### Endpoints

#### POST /predict
Предсказание поведения пользователя

**Request:**
```json
{
  "age": 25,
  "gender": "Male",
  "income": 50000,
  "session_duration": 300,
  "page_views": 10
}
```

**Response:**
```json
{
  "prediction": 0.85,
  "probability": 0.85,
  "model": "xgboost",
  "confidence": "high"
}
```

#### GET /models
Список доступных моделей

**Response:**
```json
{
  "models": ["xgboost", "lightgbm", "neural_network"],
  "active_model": "xgboost"
}
```

## Molecular Property Prediction API

### Endpoints

#### POST /predict
Предсказание свойств молекулы

**Request:**
```json
{
  "smiles": "CCO",
  "model": "gcn"
}
```

**Response:**
```json
{
  "prediction": 0.75,
  "property": "toxicity",
  "model": "gcn",
  "confidence": 0.82
}
```

## Small ML Project API

### Endpoints

#### POST /classify
Классификация данных

**Request:**
```json
{
  "features": [1.2, 3.4, 5.6, 7.8],
  "task_type": "classification"
}
```

**Response:**
```json
{
  "prediction": 1,
  "probabilities": [0.2, 0.8],
  "model": "random_forest"
}
```

## Использование

### Установка
```bash
pip install flask flask-cors
```

### Запуск API
```bash
python api_server.py
```

### Тестирование
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"age": 25, "gender": "Male"}'
```
