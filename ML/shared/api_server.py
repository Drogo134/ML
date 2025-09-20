#!/usr/bin/env python3
"""
API Server for ML Projects
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import sys
import logging
from datetime import datetime

# Добавляем пути к проектам
sys.path.append('human_behavior_prediction')
sys.path.append('biochemistry_molecules')
sys.path.append('small_ml_project')

app = Flask(__name__)
CORS(app)

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Глобальные переменные для моделей
models = {}

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'models_loaded': len(models)
    })

@app.route('/human-behavior/predict', methods=['POST'])
def predict_behavior():
    """Predict human behavior"""
    try:
        data = request.get_json()
        
        # Простая логика предсказания (заглушка)
        prediction = 0.75 if data.get('age', 0) > 30 else 0.25
        
        return jsonify({
            'prediction': prediction,
            'probability': prediction,
            'model': 'xgboost',
            'confidence': 'high' if prediction > 0.7 else 'low'
        })
    
    except Exception as e:
        logger.error(f"Error in behavior prediction: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/molecular/predict', methods=['POST'])
def predict_molecular():
    """Predict molecular properties"""
    try:
        data = request.get_json()
        smiles = data.get('smiles', '')
        
        # Простая логика предсказания (заглушка)
        prediction = 0.8 if len(smiles) > 5 else 0.3
        
        return jsonify({
            'prediction': prediction,
            'property': 'toxicity',
            'model': 'gcn',
            'confidence': 0.82
        })
    
    except Exception as e:
        logger.error(f"Error in molecular prediction: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/small-ml/classify', methods=['POST'])
def classify_data():
    """Classify data using small ML project"""
    try:
        data = request.get_json()
        features = data.get('features', [])
        
        # Простая логика классификации (заглушка)
        prediction = 1 if sum(features) > 10 else 0
        
        return jsonify({
            'prediction': prediction,
            'probabilities': [0.3, 0.7] if prediction == 1 else [0.7, 0.3],
            'model': 'random_forest'
        })
    
    except Exception as e:
        logger.error(f"Error in classification: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/models', methods=['GET'])
def get_models():
    """Get available models"""
    return jsonify({
        'human_behavior': ['xgboost', 'lightgbm', 'neural_network'],
        'molecular': ['gcn', 'gat', 'transformer'],
        'small_ml': ['random_forest', 'xgboost', 'neural_network']
    })

@app.route('/status', methods=['GET'])
def get_status():
    """Get system status"""
    return jsonify({
        'status': 'running',
        'timestamp': datetime.now().isoformat(),
        'projects': {
            'human_behavior_prediction': 'active',
            'biochemistry_molecules': 'active',
            'small_ml_project': 'active'
        }
    })

if __name__ == '__main__':
    logger.info("Starting ML Projects API Server...")
    app.run(host='0.0.0.0', port=5000, debug=True)
