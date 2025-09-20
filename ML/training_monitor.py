#!/usr/bin/env python3
"""
Скрипт для мониторинга процесса обучения моделей
"""

import os
import sys
import json
import time
import logging
from datetime import datetime, timedelta
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training_monitor.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class TrainingMonitor:
    def __init__(self):
        self.monitoring_data = {
            'training_sessions': [],
            'model_performance': {},
            'system_metrics': [],
            'alerts': []
        }
        self.load_monitoring_data()
    
    def load_monitoring_data(self):
        """Загрузка данных мониторинга"""
        if os.path.exists('monitoring_data.json'):
            with open('monitoring_data.json', 'r') as f:
                self.monitoring_data = json.load(f)
    
    def save_monitoring_data(self):
        """Сохранение данных мониторинга"""
        with open('monitoring_data.json', 'w') as f:
            json.dump(self.monitoring_data, f, indent=2)
    
    def start_training_session(self, project_name, session_type='full'):
        """Начало сессии обучения"""
        session = {
            'id': f"{project_name}_{session_type}_{int(time.time())}",
            'project': project_name,
            'type': session_type,
            'start_time': datetime.now().isoformat(),
            'status': 'running',
            'metrics': []
        }
        
        self.monitoring_data['training_sessions'].append(session)
        self.save_monitoring_data()
        
        logger.info(f"Начата сессия обучения: {session['id']}")
        return session['id']
    
    def end_training_session(self, session_id, status='completed', metrics=None):
        """Завершение сессии обучения"""
        for session in self.monitoring_data['training_sessions']:
            if session['id'] == session_id:
                session['end_time'] = datetime.now().isoformat()
                session['status'] = status
                if metrics:
                    session['metrics'] = metrics
                
                # Вычисляем длительность
                start_time = datetime.fromisoformat(session['start_time'])
                end_time = datetime.fromisoformat(session['end_time'])
                duration = (end_time - start_time).total_seconds()
                session['duration'] = duration
                
                logger.info(f"Завершена сессия обучения: {session_id} (статус: {status}, длительность: {duration:.2f}s)")
                break
        
        self.save_monitoring_data()
    
    def update_training_metrics(self, session_id, metrics):
        """Обновление метрик обучения"""
        for session in self.monitoring_data['training_sessions']:
            if session['id'] == session_id:
                session['metrics'].append({
                    'timestamp': datetime.now().isoformat(),
                    'metrics': metrics
                })
                break
        
        self.save_monitoring_data()
    
    def get_system_metrics(self):
        """Получение системных метрик"""
        try:
            import psutil
            
            metrics = {
                'timestamp': datetime.now().isoformat(),
                'cpu_percent': psutil.cpu_percent(),
                'memory_percent': psutil.virtual_memory().percent,
                'disk_percent': psutil.disk_usage('.').percent,
                'load_average': os.getloadavg() if hasattr(os, 'getloadavg') else [0, 0, 0]
            }
            
            self.monitoring_data['system_metrics'].append(metrics)
            
            # Ограничиваем историю последними 1000 записями
            if len(self.monitoring_data['system_metrics']) > 1000:
                self.monitoring_data['system_metrics'] = self.monitoring_data['system_metrics'][-1000:]
            
            self.save_monitoring_data()
            return metrics
            
        except Exception as e:
            logger.error(f"Ошибка при получении системных метрик: {e}")
            return None
    
    def check_training_health(self):
        """Проверка здоровья обучения"""
        alerts = []
        
        # Проверяем активные сессии
        active_sessions = [s for s in self.monitoring_data['training_sessions'] 
                          if s['status'] == 'running']
        
        for session in active_sessions:
            start_time = datetime.fromisoformat(session['start_time'])
            duration = (datetime.now() - start_time).total_seconds()
            
            # Предупреждение о долгой сессии
            if duration > 3600:  # 1 час
                alert = {
                    'type': 'warning',
                    'message': f"Сессия {session['id']} выполняется более 1 часа",
                    'timestamp': datetime.now().isoformat()
                }
                alerts.append(alert)
            
            # Критическое предупреждение о очень долгой сессии
            if duration > 7200:  # 2 часа
                alert = {
                    'type': 'critical',
                    'message': f"Сессия {session['id']} выполняется более 2 часов",
                    'timestamp': datetime.now().isoformat()
                }
                alerts.append(alert)
        
        # Проверяем системные метрики
        if self.monitoring_data['system_metrics']:
            latest_metrics = self.monitoring_data['system_metrics'][-1]
            
            if latest_metrics['cpu_percent'] > 90:
                alert = {
                    'type': 'warning',
                    'message': f"Высокая загрузка CPU: {latest_metrics['cpu_percent']:.1f}%",
                    'timestamp': datetime.now().isoformat()
                }
                alerts.append(alert)
            
            if latest_metrics['memory_percent'] > 90:
                alert = {
                    'type': 'warning',
                    'message': f"Высокое использование памяти: {latest_metrics['memory_percent']:.1f}%",
                    'timestamp': datetime.now().isoformat()
                }
                alerts.append(alert)
            
            if latest_metrics['disk_percent'] > 90:
                alert = {
                    'type': 'critical',
                    'message': f"Мало свободного места на диске: {latest_metrics['disk_percent']:.1f}%",
                    'timestamp': datetime.now().isoformat()
                }
                alerts.append(alert)
        
        # Добавляем новые алерты
        for alert in alerts:
            if alert not in self.monitoring_data['alerts']:
                self.monitoring_data['alerts'].append(alert)
        
        # Ограничиваем историю алертов
        if len(self.monitoring_data['alerts']) > 100:
            self.monitoring_data['alerts'] = self.monitoring_data['alerts'][-100:]
        
        self.save_monitoring_data()
        return alerts
    
    def generate_performance_report(self, project_name=None, days=7):
        """Генерация отчета о производительности"""
        logger.info("Генерация отчета о производительности...")
        
        # Фильтруем сессии по проекту и времени
        cutoff_date = datetime.now() - timedelta(days=days)
        
        if project_name:
            sessions = [s for s in self.monitoring_data['training_sessions'] 
                       if s['project'] == project_name and 
                       datetime.fromisoformat(s['start_time']) >= cutoff_date]
        else:
            sessions = [s for s in self.monitoring_data['training_sessions'] 
                       if datetime.fromisoformat(s['start_time']) >= cutoff_date]
        
        if not sessions:
            logger.warning("Нет данных для отчета")
            return
        
        # Анализируем сессии
        total_sessions = len(sessions)
        completed_sessions = len([s for s in sessions if s['status'] == 'completed'])
        failed_sessions = len([s for s in sessions if s['status'] == 'failed'])
        
        # Средняя длительность
        durations = [s.get('duration', 0) for s in sessions if s.get('duration')]
        avg_duration = np.mean(durations) if durations else 0
        
        # Группировка по проектам
        projects = {}
        for session in sessions:
            project = session['project']
            if project not in projects:
                projects[project] = {'total': 0, 'completed': 0, 'failed': 0}
            projects[project]['total'] += 1
            if session['status'] == 'completed':
                projects[project]['completed'] += 1
            elif session['status'] == 'failed':
                projects[project]['failed'] += 1
        
        # Генерируем отчет
        report = f"""
# Отчет о производительности обучения
Период: {days} дней
Сгенерирован: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Общая статистика
- Всего сессий: {total_sessions}
- Завершено успешно: {completed_sessions}
- Завершено с ошибками: {failed_sessions}
- Средняя длительность: {avg_duration:.2f} секунд

## Статистика по проектам
"""
        
        for project, stats in projects.items():
            success_rate = (stats['completed'] / stats['total'] * 100) if stats['total'] > 0 else 0
            report += f"""
### {project}
- Всего сессий: {stats['total']}
- Успешно: {stats['completed']}
- С ошибками: {stats['failed']}
- Успешность: {success_rate:.1f}%
"""
        
        # Сохраняем отчет
        report_file = f'performance_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.md'
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        logger.info(f"Отчет сохранен в {report_file}")
    
    def plot_training_metrics(self, project_name=None, days=7):
        """Построение графиков метрик обучения"""
        logger.info("Построение графиков метрик обучения...")
        
        # Фильтруем сессии
        cutoff_date = datetime.now() - timedelta(days=days)
        
        if project_name:
            sessions = [s for s in self.monitoring_data['training_sessions'] 
                       if s['project'] == project_name and 
                       datetime.fromisoformat(s['start_time']) >= cutoff_date]
        else:
            sessions = [s for s in self.monitoring_data['training_sessions'] 
                       if datetime.fromisoformat(s['start_time']) >= cutoff_date]
        
        if not sessions:
            logger.warning("Нет данных для построения графиков")
            return
        
        # График длительности сессий
        plt.figure(figsize=(15, 10))
        
        # Подграфик 1: Длительность сессий
        plt.subplot(2, 2, 1)
        durations = [s.get('duration', 0) for s in sessions if s.get('duration')]
        if durations:
            plt.hist(durations, bins=20, alpha=0.7)
            plt.title('Распределение длительности сессий')
            plt.xlabel('Длительность (секунды)')
            plt.ylabel('Количество')
        
        # Подграфик 2: Статус сессий
        plt.subplot(2, 2, 2)
        status_counts = {}
        for session in sessions:
            status = session['status']
            status_counts[status] = status_counts.get(status, 0) + 1
        
        if status_counts:
            plt.pie(status_counts.values(), labels=status_counts.keys(), autopct='%1.1f%%')
            plt.title('Статус сессий')
        
        # Подграфик 3: Временная линия
        plt.subplot(2, 2, 3)
        if self.monitoring_data['system_metrics']:
            df = pd.DataFrame(self.monitoring_data['system_metrics'])
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.set_index('timestamp')
            
            plt.plot(df.index, df['cpu_percent'], label='CPU %')
            plt.plot(df.index, df['memory_percent'], label='Memory %')
            plt.title('Системные метрики')
            plt.xlabel('Время')
            plt.ylabel('Процент')
            plt.legend()
        
        # Подграфик 4: Сессии по проектам
        plt.subplot(2, 2, 4)
        project_counts = {}
        for session in sessions:
            project = session['project']
            project_counts[project] = project_counts.get(project, 0) + 1
        
        if project_counts:
            plt.bar(project_counts.keys(), project_counts.values())
            plt.title('Количество сессий по проектам')
            plt.xlabel('Проект')
            plt.ylabel('Количество')
        
        plt.tight_layout()
        
        # Сохраняем график
        plot_file = f'training_metrics_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.show()
        
        logger.info(f"График сохранен в {plot_file}")
    
    def monitor_training(self, interval=60):
        """Мониторинг обучения в реальном времени"""
        logger.info("Запуск мониторинга обучения...")
        
        while True:
            try:
                # Получаем системные метрики
                self.get_system_metrics()
                
                # Проверяем здоровье
                alerts = self.check_training_health()
                
                if alerts:
                    for alert in alerts:
                        if alert['type'] == 'critical':
                            logger.critical(alert['message'])
                        elif alert['type'] == 'warning':
                            logger.warning(alert['message'])
                
                # Ждем следующую проверку
                time.sleep(interval)
                
            except KeyboardInterrupt:
                logger.info("Остановка мониторинга...")
                break
            except Exception as e:
                logger.error(f"Ошибка в мониторинге: {e}")
                time.sleep(interval)
    
    def get_training_status(self):
        """Получение текущего статуса обучения"""
        status = {
            'active_sessions': len([s for s in self.monitoring_data['training_sessions'] 
                                  if s['status'] == 'running']),
            'total_sessions': len(self.monitoring_data['training_sessions']),
            'recent_alerts': len([a for a in self.monitoring_data['alerts'] 
                                if datetime.fromisoformat(a['timestamp']) > 
                                datetime.now() - timedelta(hours=24)]),
            'system_health': 'healthy'
        }
        
        # Проверяем системные метрики
        if self.monitoring_data['system_metrics']:
            latest_metrics = self.monitoring_data['system_metrics'][-1]
            if (latest_metrics['cpu_percent'] > 90 or 
                latest_metrics['memory_percent'] > 90 or 
                latest_metrics['disk_percent'] > 90):
                status['system_health'] = 'warning'
        
        return status

def main():
    """Основная функция"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Мониторинг обучения моделей ML')
    parser.add_argument('--mode', choices=['monitor', 'report', 'plot'], default='monitor',
                       help='Режим работы: monitor (мониторинг), report (отчет), plot (графики)')
    parser.add_argument('--project', type=str, help='Название проекта для фильтрации')
    parser.add_argument('--days', type=int, default=7, help='Количество дней для анализа')
    parser.add_argument('--interval', type=int, default=60, help='Интервал мониторинга в секундах')
    
    args = parser.parse_args()
    
    monitor = TrainingMonitor()
    
    if args.mode == 'monitor':
        monitor.monitor_training(interval=args.interval)
    elif args.mode == 'report':
        monitor.generate_performance_report(project_name=args.project, days=args.days)
    elif args.mode == 'plot':
        monitor.plot_training_metrics(project_name=args.project, days=args.days)

if __name__ == "__main__":
    main()
