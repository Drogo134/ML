#!/usr/bin/env python3
"""
API Testing Script
"""

import requests
import json
import time

API_BASE_URL = "http://localhost:5000"

def test_health():
    """Test health endpoint"""
    print("Testing health endpoint...")
    response = requests.get(f"{API_BASE_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    print()

def test_behavior_prediction():
    """Test human behavior prediction"""
    print("Testing behavior prediction...")
    data = {
        "age": 25,
        "gender": "Male",
        "income": 50000,
        "session_duration": 300,
        "page_views": 10
    }
    response = requests.post(f"{API_BASE_URL}/human-behavior/predict", json=data)
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    print()

def test_molecular_prediction():
    """Test molecular property prediction"""
    print("Testing molecular prediction...")
    data = {
        "smiles": "CCO",
        "model": "gcn"
    }
    response = requests.post(f"{API_BASE_URL}/molecular/predict", json=data)
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    print()

def test_classification():
    """Test data classification"""
    print("Testing classification...")
    data = {
        "features": [1.2, 3.4, 5.6, 7.8],
        "task_type": "classification"
    }
    response = requests.post(f"{API_BASE_URL}/small-ml/classify", json=data)
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    print()

def test_models():
    """Test models endpoint"""
    print("Testing models endpoint...")
    response = requests.get(f"{API_BASE_URL}/models")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    print()

def test_status():
    """Test status endpoint"""
    print("Testing status endpoint...")
    response = requests.get(f"{API_BASE_URL}/status")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    print()

def main():
    """Run all tests"""
    print("=" * 50)
    print("ML Projects API Testing")
    print("=" * 50)
    
    try:
        test_health()
        test_behavior_prediction()
        test_molecular_prediction()
        test_classification()
        test_models()
        test_status()
        
        print("=" * 50)
        print("All tests completed successfully!")
        print("=" * 50)
        
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to API server.")
        print("Make sure the server is running on http://localhost:5000")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
