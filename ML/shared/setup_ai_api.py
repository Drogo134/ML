#!/usr/bin/env python3
"""
Скрипт для настройки AI API ключей
"""

import os
import json
import getpass
from ai_integration import AIAPIManager

def setup_api_keys():
    """Интерактивная настройка API ключей"""
    print("🤖 НАСТРОЙКА AI API КЛЮЧЕЙ")
    print("=" * 50)
    
    api_manager = AIAPIManager()
    
    # Список доступных сервисов
    services = {
        'openai': 'OpenAI (GPT-3.5, GPT-4)',
        'anthropic': 'Anthropic (Claude)',
        'google': 'Google (Gemini)',
        'huggingface': 'Hugging Face',
        'cohere': 'Cohere',
        'replicate': 'Replicate',
        'stability': 'Stability AI',
        'midjourney': 'Midjourney',
        'custom': 'Кастомный API'
    }
    
    print("Доступные AI сервисы:")
    for i, (key, name) in enumerate(services.items(), 1):
        print(f"{i}. {name} ({key})")
    
    print("\nВыберите сервисы для настройки (через запятую, например: 1,2,3):")
    choices = input("Введите номера: ").strip()
    
    if not choices:
        print("❌ Не выбрано ни одного сервиса")
        return
    
    try:
        selected_indices = [int(x.strip()) - 1 for x in choices.split(',')]
        selected_services = [list(services.keys())[i] for i in selected_indices if 0 <= i < len(services)]
    except ValueError:
        print("❌ Неверный формат ввода")
        return
    
    print(f"\nВыбраны сервисы: {[services[s] for s in selected_services]}")
    
    # Настройка каждого сервиса
    for service in selected_services:
        print(f"\n--- Настройка {services[service]} ---")
        
        if service == 'custom':
            url = input("Введите URL API: ").strip()
            if url:
                api_manager.update_api_key('custom_api_url', url)
            
            key = getpass.getpass("Введите API ключ: ").strip()
            if key:
                api_manager.update_api_key('custom_api_key', key)
        else:
            key = getpass.getpass(f"Введите API ключ для {services[service]}: ").strip()
            if key:
                api_manager.update_api_key(f'{service}_api_key', key)
    
    # Тестирование подключений
    print("\n🧪 ТЕСТИРОВАНИЕ ПОДКЛЮЧЕНИЙ")
    print("=" * 30)
    
    for service in selected_services:
        if service == 'custom':
            continue  # Пропускаем кастомный API
        
        print(f"Тестирование {services[service]}...")
        if api_manager.test_api_connection(service):
            print(f"✅ {services[service]}: подключение успешно")
        else:
            print(f"❌ {services[service]}: ошибка подключения")
    
    # Сохранение конфигурации
    api_manager.save_api_keys()
    print(f"\n✅ Конфигурация сохранена в ai_api_config.json")
    
    # Показываем доступные сервисы
    available_services = api_manager.get_available_services()
    if available_services:
        print(f"\n🎉 Доступные AI сервисы: {', '.join(available_services)}")
    else:
        print("\n⚠️ Нет доступных AI сервисов")

def test_ai_integration():
    """Тестирование AI интеграции"""
    print("\n🧪 ТЕСТИРОВАНИЕ AI ИНТЕГРАЦИИ")
    print("=" * 40)
    
    api_manager = AIAPIManager()
    available_services = api_manager.get_available_services()
    
    if not available_services:
        print("❌ Нет доступных AI сервисов")
        return
    
    # Тестируем генерацию текста
    for service in available_services:
        print(f"\nТестирование {service}...")
        
        response = api_manager.generate_text(
            "Привет! Это тест AI интеграции. Ответь кратко.",
            service,
            max_tokens=50
        )
        
        if response:
            print(f"✅ {service}: {response[:100]}...")
        else:
            print(f"❌ {service}: ошибка генерации")
    
    # Тестируем анализ данных
    test_data = {
        "accuracy": 0.85,
        "precision": 0.82,
        "recall": 0.88,
        "f1_score": 0.85
    }
    
    print(f"\nТестирование анализа данных...")
    analysis = api_manager.analyze_data_with_ai(
        test_data, 'ml_results', available_services[0]
    )
    
    if analysis:
        print(f"✅ Анализ данных: {analysis[:200]}...")
    else:
        print("❌ Ошибка анализа данных")

def show_api_status():
    """Показ статуса API"""
    print("\n📊 СТАТУС AI API")
    print("=" * 30)
    
    api_manager = AIAPIManager()
    available_services = api_manager.get_available_services()
    
    if available_services:
        print(f"✅ Доступные сервисы: {', '.join(available_services)}")
        
        for service in available_services:
            if api_manager.test_api_connection(service):
                print(f"  ✅ {service}: подключение активно")
            else:
                print(f"  ❌ {service}: ошибка подключения")
    else:
        print("❌ Нет доступных AI сервисов")
        print("Настройте API ключи с помощью: python setup_ai_api.py")

def main():
    """Основная функция"""
    print("🤖 AI API SETUP TOOL")
    print("=" * 50)
    
    while True:
        print("\nВыберите действие:")
        print("1. Настроить API ключи")
        print("2. Тестировать AI интеграцию")
        print("3. Показать статус API")
        print("4. Выход")
        
        choice = input("\nВведите номер (1-4): ").strip()
        
        if choice == '1':
            setup_api_keys()
        elif choice == '2':
            test_ai_integration()
        elif choice == '3':
            show_api_status()
        elif choice == '4':
            print("\n👋 До свидания!")
            break
        else:
            print("❌ Неверный выбор!")

if __name__ == "__main__":
    main()
