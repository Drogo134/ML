FROM python:3.9-slim

# Установка системных зависимостей
RUN apt-get update && apt-get install -y \
    build-essential \
    libxrender1 \
    libxext6 \
    libsm6 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Установка рабочей директории
WORKDIR /app

# Копирование файлов
COPY requirements.txt .
COPY . .

# Установка Python зависимостей
RUN pip install --no-cache-dir -r requirements.txt

# Создание директорий
RUN python -c "
from human_behavior_prediction.config import Config as Config1
from biochemistry_molecules.config import Config as Config2
from small_ml_project.config import Config as Config3
Config1.create_directories()
Config2.create_directories()
Config3.create_directories()
"

# Установка переменных окружения
ENV PYTHONPATH=/app
ENV ML_PROJECTS_HOME=/app

# Открытие портов
EXPOSE 5000

# Команда по умолчанию
CMD ["python", "api_server.py"]
