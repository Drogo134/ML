# Руководство по развертыванию ML проектов

## Обзор развертывания

Данное руководство описывает процесс развертывания трех проектов машинного обучения в различных окружениях: локальном, облачном и контейнеризованном.

## Предварительные требования

### Системные требования
- **OS**: Windows 10+, macOS 10.14+, Ubuntu 18.04+
- **Python**: 3.8 или выше
- **RAM**: 8+ GB
- **CPU**: 4+ ядер
- **Диск**: 20+ GB свободного места

### Зависимости
- Docker (для контейнеризации)
- Kubernetes (для оркестрации)
- Cloud provider account (AWS, GCP, Azure)

## Локальное развертывание

### 1. Установка зависимостей

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
```

### 2. Настройка окружения

```bash
# Создание директорий
python -c "
from human_behavior_prediction.config import Config as Config1
from biochemistry_molecules.config import Config as Config2
from small_ml_project.config import Config as Config3
Config1.create_directories()
Config2.create_directories()
Config3.create_directories()
"

# Настройка переменных окружения
export ML_PROJECTS_HOME=$(pwd)
export PYTHONPATH=$ML_PROJECTS_HOME:$PYTHONPATH
```

### 3. Запуск проектов

```bash
# Запуск всех проектов
python run_all_projects.py

# Или запуск отдельных проектов
cd human_behavior_prediction && python main.py
cd ../biochemistry_molecules && python main.py
cd ../small_ml_project && python main.py
```

### 4. Проверка развертывания

```bash
# Проверка логов
tail -f */logs/*.log

# Проверка результатов
ls -la */results/

# Проверка моделей
ls -la */models/
```

## Контейнеризация с Docker

### 1. Создание Dockerfile

```dockerfile
# Dockerfile
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
EXPOSE 8000 8001 8002

# Команда по умолчанию
CMD ["python", "run_all_projects.py"]
```

### 2. Создание docker-compose.yml

```yaml
# docker-compose.yml
version: '3.8'

services:
  human-behavior:
    build: .
    container_name: human-behavior-prediction
    ports:
      - "8000:8000"
    volumes:
      - ./human_behavior_prediction/data:/app/human_behavior_prediction/data
      - ./human_behavior_prediction/models:/app/human_behavior_prediction/models
      - ./human_behavior_prediction/results:/app/human_behavior_prediction/results
    environment:
      - PYTHONPATH=/app
      - ML_PROJECTS_HOME=/app
    command: ["python", "human_behavior_prediction/main.py"]

  molecular-property:
    build: .
    container_name: molecular-property-prediction
    ports:
      - "8001:8001"
    volumes:
      - ./biochemistry_molecules/data:/app/biochemistry_molecules/data
      - ./biochemistry_molecules/models:/app/biochemistry_molecules/models
      - ./biochemistry_molecules/results:/app/biochemistry_molecules/results
    environment:
      - PYTHONPATH=/app
      - ML_PROJECTS_HOME=/app
    command: ["python", "biochemistry_molecules/main.py"]

  small-ml:
    build: .
    container_name: small-ml-project
    ports:
      - "8002:8002"
    volumes:
      - ./small_ml_project/data:/app/small_ml_project/data
      - ./small_ml_project/models:/app/small_ml_project/models
      - ./small_ml_project/results:/app/small_ml_project/results
    environment:
      - PYTHONPATH=/app
      - ML_PROJECTS_HOME=/app
    command: ["python", "small_ml_project/main.py"]

  nginx:
    image: nginx:alpine
    container_name: ml-nginx
    ports:
      - "80:80"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
    depends_on:
      - human-behavior
      - molecular-property
      - small-ml
```

### 3. Создание nginx.conf

```nginx
# nginx.conf
events {
    worker_connections 1024;
}

http {
    upstream human_behavior {
        server human-behavior:8000;
    }
    
    upstream molecular_property {
        server molecular-property:8001;
    }
    
    upstream small_ml {
        server small-ml:8002;
    }
    
    server {
        listen 80;
        
        location /human-behavior/ {
            proxy_pass http://human_behavior/;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
        }
        
        location /molecular-property/ {
            proxy_pass http://molecular_property/;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
        }
        
        location /small-ml/ {
            proxy_pass http://small_ml/;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
        }
    }
}
```

### 4. Запуск контейнеров

```bash
# Сборка и запуск
docker-compose up --build

# Запуск в фоновом режиме
docker-compose up -d

# Просмотр логов
docker-compose logs -f

# Остановка
docker-compose down
```

## Развертывание в облаке

### AWS Deployment

#### 1. Создание EC2 инстанса

```bash
# Создание ключа
aws ec2 create-key-pair --key-name ml-projects-key --query 'KeyMaterial' --output text > ml-projects-key.pem
chmod 400 ml-projects-key.pem

# Запуск инстанса
aws ec2 run-instances \
    --image-id ami-0c02fb55956c7d316 \
    --instance-type t3.large \
    --key-name ml-projects-key \
    --security-group-ids sg-12345678 \
    --subnet-id subnet-12345678 \
    --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=ml-projects}]'
```

#### 2. Настройка инстанса

```bash
# Подключение к инстансу
ssh -i ml-projects-key.pem ec2-user@<instance-ip>

# Установка Docker
sudo yum update -y
sudo yum install -y docker
sudo service docker start
sudo usermod -a -G docker ec2-user

# Установка Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/download/1.29.2/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose
```

#### 3. Развертывание приложения

```bash
# Клонирование репозитория
git clone <repository-url>
cd "проекты ML"

# Запуск контейнеров
docker-compose up -d
```

### Google Cloud Platform

#### 1. Создание кластера Kubernetes

```bash
# Создание кластера
gcloud container clusters create ml-projects-cluster \
    --zone us-central1-a \
    --num-nodes 3 \
    --machine-type n1-standard-2

# Получение учетных данных
gcloud container clusters get-credentials ml-projects-cluster --zone us-central1-a
```

#### 2. Создание Kubernetes манифестов

```yaml
# k8s/namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: ml-projects
```

```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: human-behavior-prediction
  namespace: ml-projects
spec:
  replicas: 2
  selector:
    matchLabels:
      app: human-behavior-prediction
  template:
    metadata:
      labels:
        app: human-behavior-prediction
    spec:
      containers:
      - name: human-behavior
        image: ml-projects:latest
        ports:
        - containerPort: 8000
        env:
        - name: PYTHONPATH
          value: "/app"
        - name: ML_PROJECTS_HOME
          value: "/app"
        command: ["python", "human_behavior_prediction/main.py"]
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
```

```yaml
# k8s/service.yaml
apiVersion: v1
kind: Service
metadata:
  name: human-behavior-service
  namespace: ml-projects
spec:
  selector:
    app: human-behavior-prediction
  ports:
  - port: 80
    targetPort: 8000
  type: LoadBalancer
```

#### 3. Развертывание в Kubernetes

```bash
# Применение манифестов
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml

# Проверка статуса
kubectl get pods -n ml-projects
kubectl get services -n ml-projects
```

### Azure Deployment

#### 1. Создание Azure Container Instances

```bash
# Создание группы ресурсов
az group create --name ml-projects-rg --location eastus

# Создание Container Registry
az acr create --resource-group ml-projects-rg --name mlprojectsacr --sku Basic

# Сборка и отправка образа
az acr build --registry mlprojectsacr --image ml-projects:latest .

# Создание Container Instance
az container create \
    --resource-group ml-projects-rg \
    --name human-behavior-container \
    --image mlprojectsacr.azurecr.io/ml-projects:latest \
    --cpu 2 \
    --memory 4 \
    --registry-login-server mlprojectsacr.azurecr.io \
    --registry-username <username> \
    --registry-password <password> \
    --command-line "python human_behavior_prediction/main.py"
```

## Мониторинг и логирование

### 1. Настройка логирования

```python
# logging_config.py
import logging
import logging.handlers
import os

def setup_logging():
    # Создание директории для логов
    os.makedirs('logs', exist_ok=True)
    
    # Настройка root logger
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.handlers.RotatingFileHandler(
                'logs/ml_projects.log',
                maxBytes=10*1024*1024,  # 10MB
                backupCount=5
            ),
            logging.StreamHandler()
        ]
    )
```

### 2. Мониторинг с Prometheus

```python
# monitoring.py
from prometheus_client import Counter, Histogram, start_http_server
import time

# Метрики
REQUEST_COUNT = Counter('requests_total', 'Total requests', ['method', 'endpoint'])
REQUEST_DURATION = Histogram('request_duration_seconds', 'Request duration')

def monitor_requests(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            REQUEST_COUNT.labels(method='GET', endpoint=func.__name__).inc()
            return result
        finally:
            REQUEST_DURATION.observe(time.time() - start_time)
    return wrapper

# Запуск Prometheus сервера
start_http_server(8000)
```

### 3. Health checks

```python
# health_check.py
from flask import Flask, jsonify
import psutil
import os

app = Flask(__name__)

@app.route('/health')
def health_check():
    return jsonify({
        'status': 'healthy',
        'cpu_percent': psutil.cpu_percent(),
        'memory_percent': psutil.virtual_memory().percent,
        'disk_percent': psutil.disk_usage('/').percent
    })

@app.route('/ready')
def readiness_check():
    # Проверка доступности моделей
    model_files = [
        'human_behavior_prediction/models/',
        'biochemistry_molecules/models/',
        'small_ml_project/models/'
    ]
    
    for model_dir in model_files:
        if not os.path.exists(model_dir):
            return jsonify({'status': 'not ready'}), 503
    
    return jsonify({'status': 'ready'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
```

## Автоматизация развертывания

### 1. CI/CD с GitHub Actions

```yaml
# .github/workflows/deploy.yml
name: Deploy ML Projects

on:
  push:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: 3.9
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install pytest
    - name: Run tests
      run: pytest tests/

  build:
    needs: test
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    - name: Build Docker image
      run: docker build -t ml-projects:${{ github.sha }} .
    - name: Push to registry
      run: |
        echo ${{ secrets.DOCKER_PASSWORD }} | docker login -u ${{ secrets.DOCKER_USERNAME }} --password-stdin
        docker push ml-projects:${{ github.sha }}

  deploy:
    needs: build
    runs-on: ubuntu-latest
    steps:
    - name: Deploy to production
      run: |
        # Развертывание в production
        kubectl set image deployment/ml-projects ml-projects=ml-projects:${{ github.sha }}
```

### 2. Terraform для инфраструктуры

```hcl
# terraform/main.tf
provider "aws" {
  region = "us-west-2"
}

resource "aws_instance" "ml_projects" {
  ami           = "ami-0c02fb55956c7d316"
  instance_type = "t3.large"
  
  tags = {
    Name = "ml-projects"
  }
}

resource "aws_security_group" "ml_projects" {
  name_prefix = "ml-projects-"
  
  ingress {
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }
  
  ingress {
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }
}
```

## Безопасность

### 1. Настройка SSL/TLS

```nginx
# nginx-ssl.conf
server {
    listen 443 ssl;
    server_name ml-projects.example.com;
    
    ssl_certificate /etc/ssl/certs/ml-projects.crt;
    ssl_certificate_key /etc/ssl/private/ml-projects.key;
    
    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

### 2. Аутентификация и авторизация

```python
# auth.py
from functools import wraps
from flask import request, jsonify
import jwt

def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.headers.get('Authorization')
        
        if not token:
            return jsonify({'message': 'Token is missing'}), 401
        
        try:
            data = jwt.decode(token, app.config['SECRET_KEY'])
            current_user = data['user_id']
        except:
            return jsonify({'message': 'Token is invalid'}), 401
        
        return f(current_user, *args, **kwargs)
    
    return decorated
```

### 3. Ограничение скорости

```python
# rate_limiting.py
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

limiter = Limiter(
    app,
    key_func=get_remote_address,
    default_limits=["200 per day", "50 per hour"]
)

@app.route('/api/predict')
@limiter.limit("10 per minute")
def predict():
    # Логика предсказания
    pass
```

## Масштабирование

### 1. Горизонтальное масштабирование

```yaml
# k8s/hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: ml-projects-hpa
  namespace: ml-projects
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: human-behavior-prediction
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
```

### 2. Вертикальное масштабирование

```yaml
# k8s/vpa.yaml
apiVersion: autoscaling.k8s.io/v1
kind: VerticalPodAutoscaler
metadata:
  name: ml-projects-vpa
  namespace: ml-projects
spec:
  targetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: human-behavior-prediction
  updatePolicy:
    updateMode: "Auto"
```

## Резервное копирование

### 1. Backup данных

```bash
#!/bin/bash
# backup.sh

# Создание backup директории
mkdir -p backups/$(date +%Y%m%d)

# Backup данных
tar -czf backups/$(date +%Y%m%d)/data.tar.gz */data/

# Backup моделей
tar -czf backups/$(date +%Y%m%d)/models.tar.gz */models/

# Backup результатов
tar -czf backups/$(date +%Y%m%d)/results.tar.gz */results/

# Загрузка в S3
aws s3 cp backups/$(date +%Y%m%d)/ s3://ml-projects-backup/$(date +%Y%m%d)/ --recursive
```

### 2. Восстановление

```bash
#!/bin/bash
# restore.sh

# Загрузка из S3
aws s3 cp s3://ml-projects-backup/$1/ ./backups/$1/ --recursive

# Восстановление данных
tar -xzf backups/$1/data.tar.gz
tar -xzf backups/$1/models.tar.gz
tar -xzf backups/$1/results.tar.gz
```

## Заключение

Данное руководство по развертыванию обеспечивает:

1. **Локальное развертывание**: Быстрый старт для разработки
2. **Контейнеризация**: Изолированное и переносимое развертывание
3. **Облачное развертывание**: Масштабируемые решения
4. **Мониторинг**: Отслеживание производительности и здоровья
5. **Безопасность**: Защита данных и API
6. **Автоматизация**: CI/CD и инфраструктура как код
7. **Масштабирование**: Горизонтальное и вертикальное масштабирование
8. **Резервное копирование**: Защита данных и моделей

Выберите подходящий метод развертывания в зависимости от ваших требований и ресурсов.
