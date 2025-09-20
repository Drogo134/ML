import logging
import json
import time
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings('ignore')

class ModelMonitoring:
    def __init__(self, config):
        self.config = config
        self.logs_dir = config.LOGS_DIR
        self.results_dir = config.RESULTS_DIR
        
        self.setup_logging()
        self.metrics_history = []
        self.performance_alerts = []
        
    def setup_logging(self):
        log_file = self.logs_dir / f"model_monitoring_{datetime.now().strftime('%Y%m%d')}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger('ModelMonitoring')
    
    def log_model_performance(self, model_name: str, metrics: Dict[str, float], 
                            data_info: Dict[str, Any] = None):
        performance_record = {
            'timestamp': datetime.now().isoformat(),
            'model_name': model_name,
            'metrics': metrics,
            'data_info': data_info or {}
        }
        
        self.metrics_history.append(performance_record)
        
        self.logger.info(f"Model {model_name} performance: {metrics}")
        
        self.save_performance_log(performance_record)
        self.check_performance_thresholds(model_name, metrics)
    
    def save_performance_log(self, performance_record: Dict):
        log_file = self.logs_dir / "performance_log.jsonl"
        
        with open(log_file, 'a') as f:
            f.write(json.dumps(performance_record) + '\n')
    
    def check_performance_thresholds(self, model_name: str, metrics: Dict[str, float]):
        thresholds = {
            'accuracy': 0.8,
            'precision': 0.75,
            'recall': 0.75,
            'f1': 0.75,
            'auc': 0.8
        }
        
        alerts = []
        
        for metric, threshold in thresholds.items():
            if metric in metrics and metrics[metric] < threshold:
                alert = {
                    'timestamp': datetime.now().isoformat(),
                    'model_name': model_name,
                    'metric': metric,
                    'value': metrics[metric],
                    'threshold': threshold,
                    'severity': 'warning' if metrics[metric] > threshold * 0.8 else 'critical'
                }
                alerts.append(alert)
                self.performance_alerts.append(alert)
                
                self.logger.warning(
                    f"Performance alert: {model_name} {metric} = {metrics[metric]:.4f} "
                    f"(threshold: {threshold})"
                )
        
        if alerts:
            self.save_alerts(alerts)
    
    def save_alerts(self, alerts: List[Dict]):
        alert_file = self.logs_dir / "performance_alerts.jsonl"
        
        with open(alert_file, 'a') as f:
            for alert in alerts:
                f.write(json.dumps(alert) + '\n')
    
    def get_model_performance_history(self, model_name: str = None, 
                                    days: int = 30) -> pd.DataFrame:
        cutoff_date = datetime.now() - timedelta(days=days)
        
        if model_name:
            filtered_history = [
                record for record in self.metrics_history
                if (record['model_name'] == model_name and 
                    datetime.fromisoformat(record['timestamp']) >= cutoff_date)
            ]
        else:
            filtered_history = [
                record for record in self.metrics_history
                if datetime.fromisoformat(record['timestamp']) >= cutoff_date
            ]
        
        if not filtered_history:
            return pd.DataFrame()
        
        df_data = []
        for record in filtered_history:
            row = {
                'timestamp': record['timestamp'],
                'model_name': record['model_name']
            }
            row.update(record['metrics'])
            df_data.append(row)
        
        return pd.DataFrame(df_data)
    
    def plot_performance_trends(self, model_name: str = None, days: int = 30):
        df = self.get_model_performance_history(model_name, days)
        
        if df.empty:
            self.logger.warning("No performance data available for plotting")
            return
        
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1', 'auc']
        available_metrics = [m for m in metrics_to_plot if m in df.columns]
        
        if not available_metrics:
            self.logger.warning("No metrics available for plotting")
            return
        
        fig, axes = plt.subplots(len(available_metrics), 1, figsize=(12, 3 * len(available_metrics)))
        if len(available_metrics) == 1:
            axes = [axes]
        
        for i, metric in enumerate(available_metrics):
            if model_name:
                metric_data = df[df['model_name'] == model_name]
                title = f'{model_name} - {metric.title()} Over Time'
            else:
                metric_data = df
                title = f'{metric.title()} Over Time (All Models)'
            
            if not metric_data.empty:
                axes[i].plot(metric_data['timestamp'], metric_data[metric], marker='o')
                axes[i].set_title(title)
                axes[i].set_ylabel(metric.title())
                axes[i].grid(True, alpha=0.3)
                
                if model_name:
                    axes[i].legend([model_name])
                else:
                    for model in metric_data['model_name'].unique():
                        model_data = metric_data[metric_data['model_name'] == model]
                        axes[i].plot(model_data['timestamp'], model_data[metric], 
                                   marker='o', label=model)
                    axes[i].legend()
        
        plt.tight_layout()
        
        plot_file = self.results_dir / f"performance_trends_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.show()
        
        self.logger.info(f"Performance trends plot saved to {plot_file}")
    
    def detect_data_drift(self, current_data: pd.DataFrame, 
                         reference_data: pd.DataFrame = None,
                         threshold: float = 0.1) -> Dict[str, Any]:
        if reference_data is None:
            reference_data = current_data
        
        drift_report = {
            'timestamp': datetime.now().isoformat(),
            'drift_detected': False,
            'drift_metrics': {},
            'alerts': []
        }
        
        numeric_columns = current_data.select_dtypes(include=[np.number]).columns
        
        for column in numeric_columns:
            if column in reference_data.columns:
                current_mean = current_data[column].mean()
                reference_mean = reference_data[column].mean()
                
                current_std = current_data[column].std()
                reference_std = reference_data[column].std()
                
                mean_drift = abs(current_mean - reference_mean) / reference_mean if reference_mean != 0 else 0
                std_drift = abs(current_std - reference_std) / reference_std if reference_std != 0 else 0
                
                drift_report['drift_metrics'][column] = {
                    'mean_drift': mean_drift,
                    'std_drift': std_drift,
                    'current_mean': current_mean,
                    'reference_mean': reference_mean,
                    'current_std': current_std,
                    'reference_std': reference_std
                }
                
                if mean_drift > threshold or std_drift > threshold:
                    drift_report['drift_detected'] = True
                    alert = {
                        'column': column,
                        'mean_drift': mean_drift,
                        'std_drift': std_drift,
                        'severity': 'high' if max(mean_drift, std_drift) > threshold * 2 else 'medium'
                    }
                    drift_report['alerts'].append(alert)
                    
                    self.logger.warning(
                        f"Data drift detected in {column}: mean_drift={mean_drift:.4f}, "
                        f"std_drift={std_drift:.4f}"
                    )
        
        self.save_drift_report(drift_report)
        return drift_report
    
    def save_drift_report(self, drift_report: Dict):
        drift_file = self.logs_dir / f"drift_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(drift_file, 'w') as f:
            json.dump(drift_report, f, indent=2)
    
    def monitor_model_predictions(self, model, X_test: pd.DataFrame, 
                                y_test: pd.Series, model_name: str):
        predictions = model.predict(X_test)
        
        if hasattr(model, 'predict_proba'):
            prediction_probas = model.predict_proba(X_test)
        else:
            prediction_probas = None
        
        prediction_stats = {
            'timestamp': datetime.now().isoformat(),
            'model_name': model_name,
            'total_predictions': len(predictions),
            'prediction_distribution': np.bincount(predictions).tolist(),
            'prediction_mean': float(np.mean(predictions)),
            'prediction_std': float(np.std(predictions))
        }
        
        if prediction_probas is not None:
            prediction_stats['probability_mean'] = float(np.mean(prediction_probas))
            prediction_stats['probability_std'] = float(np.std(prediction_probas))
        
        self.save_prediction_stats(prediction_stats)
        
        self.logger.info(f"Prediction monitoring completed for {model_name}")
        return prediction_stats
    
    def save_prediction_stats(self, prediction_stats: Dict):
        stats_file = self.logs_dir / "prediction_stats.jsonl"
        
        with open(stats_file, 'a') as f:
            f.write(json.dumps(prediction_stats) + '\n')
    
    def generate_monitoring_report(self, days: int = 7) -> Dict[str, Any]:
        report = {
            'generated_at': datetime.now().isoformat(),
            'period_days': days,
            'summary': {},
            'performance_summary': {},
            'drift_summary': {},
            'alerts_summary': {}
        }
        
        df = self.get_model_performance_history(days=days)
        
        if not df.empty:
            report['performance_summary'] = {
                'total_evaluations': len(df),
                'models_evaluated': df['model_name'].nunique(),
                'average_metrics': df.groupby('model_name').mean().to_dict()
            }
        
        recent_alerts = [
            alert for alert in self.performance_alerts
            if datetime.fromisoformat(alert['timestamp']) >= datetime.now() - timedelta(days=days)
        ]
        
        report['alerts_summary'] = {
            'total_alerts': len(recent_alerts),
            'critical_alerts': len([a for a in recent_alerts if a['severity'] == 'critical']),
            'warning_alerts': len([a for a in recent_alerts if a['severity'] == 'warning']),
            'alerts_by_model': {}
        }
        
        for alert in recent_alerts:
            model = alert['model_name']
            if model not in report['alerts_summary']['alerts_by_model']:
                report['alerts_summary']['alerts_by_model'][model] = 0
            report['alerts_summary']['alerts_by_model'][model] += 1
        
        report_file = self.results_dir / f"monitoring_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        self.logger.info(f"Monitoring report generated: {report_file}")
        return report
    
    def setup_automated_monitoring(self, check_interval_hours: int = 24):
        self.logger.info(f"Setting up automated monitoring (check every {check_interval_hours} hours)")
        
        while True:
            try:
                self.generate_monitoring_report()
                self.plot_performance_trends()
                
                time.sleep(check_interval_hours * 3600)
                
            except KeyboardInterrupt:
                self.logger.info("Automated monitoring stopped")
                break
            except Exception as e:
                self.logger.error(f"Error in automated monitoring: {e}")
                time.sleep(300)  # Wait 5 minutes before retrying
