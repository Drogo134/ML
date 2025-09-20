#!/usr/bin/env python3
"""
Master script to run all ML projects
"""

import os
import sys
import time
import logging
from pathlib import Path
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('master_run.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def run_project(project_name, project_path):
    """Run a specific project"""
    logger.info(f"Starting {project_name}...")
    
    try:
        # Change to project directory
        original_cwd = os.getcwd()
        os.chdir(project_path)
        
        # Import and run the project
        if project_name == "Human Behavior Prediction":
            from main import HumanBehaviorPredictionPipeline
            pipeline = HumanBehaviorPredictionPipeline()
            results = pipeline.run_full_pipeline(n_samples=5000)
            
        elif project_name == "Molecular Property Prediction":
            from main import MolecularPropertyPredictionPipeline
            pipeline = MolecularPropertyPredictionPipeline()
            results = pipeline.run_full_pipeline(dataset_name='tox21')
            
        elif project_name == "Small ML Project":
            from main import MLPipeline
            pipeline = MLPipeline()
            
            # Run multiple task types
            results_clf = pipeline.run_pipeline(
                task_type='classification',
                n_samples=2000,
                n_features=15
            )
            
            results_reg = pipeline.run_pipeline(
                task_type='regression',
                n_samples=2000,
                n_features=15
            )
            
            results = {'classification': results_clf, 'regression': results_reg}
        
        else:
            raise ValueError(f"Unknown project: {project_name}")
        
        logger.info(f"{project_name} completed successfully")
        return results
        
    except Exception as e:
        logger.error(f"Error running {project_name}: {e}")
        return None
        
    finally:
        # Return to original directory
        os.chdir(original_cwd)

def check_dependencies():
    """Check if all required dependencies are installed"""
    logger.info("Checking dependencies...")
    
    required_packages = [
        'numpy', 'pandas', 'scikit-learn', 'tensorflow', 'torch',
        'xgboost', 'lightgbm', 'matplotlib', 'seaborn', 'plotly'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        logger.error(f"Missing packages: {missing_packages}")
        logger.error("Please install missing packages using: pip install -r requirements.txt")
        return False
    
    logger.info("All dependencies are installed")
    return True

def create_directories():
    """Create necessary directories for all projects"""
    logger.info("Creating directories...")
    
    projects = [
        'human_behavior_prediction',
        'biochemistry_molecules', 
        'small_ml_project'
    ]
    
    for project in projects:
        project_path = Path(project)
        if project_path.exists():
            # Create subdirectories
            for subdir in ['data', 'models', 'results', 'logs']:
                (project_path / subdir).mkdir(exist_ok=True)
            logger.info(f"Directories created for {project}")

def generate_summary_report(results):
    """Generate a summary report of all projects"""
    logger.info("Generating summary report...")
    
    report = f"""
# ML Projects Execution Summary
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Project Results

"""
    
    for project_name, result in results.items():
        if result is not None:
            report += f"### {project_name}\n"
            report += f"Status: ✅ Completed Successfully\n"
            
            if isinstance(result, dict):
                if 'classification' in result and 'regression' in result:
                    # Small ML Project results
                    report += f"Classification Results: {len(result['classification'])} models evaluated\n"
                    report += f"Regression Results: {len(result['regression'])} models evaluated\n"
                else:
                    # Single project results
                    report += f"Models evaluated: {len(result)}\n"
                    for model_name, metrics in result.items():
                        if isinstance(metrics, dict):
                            report += f"- {model_name}: "
                            if 'accuracy' in metrics:
                                report += f"Accuracy={metrics['accuracy']:.4f}"
                            elif 'mse' in metrics:
                                report += f"MSE={metrics['mse']:.4f}"
                            report += "\n"
            else:
                report += f"Results: {type(result).__name__}\n"
        else:
            report += f"### {project_name}\n"
            report += f"Status: ❌ Failed\n"
        
        report += "\n"
    
    report += f"""
## Summary
- Total projects: {len(results)}
- Successful: {sum(1 for r in results.values() if r is not None)}
- Failed: {sum(1 for r in results.values() if r is None)}

## Next Steps
1. Check individual project results in their respective results/ directories
2. Review logs for any warnings or errors
3. Customize parameters in config.py files as needed
4. Run individual projects for more detailed analysis

## Support
- Check project-specific README files for detailed documentation
- Review logs in logs/ directories for troubleshooting
- Create issues in the repository for bugs or questions
"""
    
    # Save report
    with open('execution_summary.md', 'w', encoding='utf-8') as f:
        f.write(report)
    
    logger.info("Summary report saved to execution_summary.md")
    print(report)

def main():
    """Main execution function"""
    start_time = time.time()
    
    logger.info("=" * 60)
    logger.info("Starting ML Projects Execution")
    logger.info("=" * 60)
    
    # Check dependencies
    if not check_dependencies():
        logger.error("Dependency check failed. Exiting.")
        sys.exit(1)
    
    # Create directories
    create_directories()
    
    # Define projects to run
    projects = {
        "Human Behavior Prediction": "human_behavior_prediction",
        "Molecular Property Prediction": "biochemistry_molecules",
        "Small ML Project": "small_ml_project"
    }
    
    # Run projects
    results = {}
    
    for project_name, project_path in projects.items():
        if Path(project_path).exists():
            project_start = time.time()
            result = run_project(project_name, project_path)
            project_time = time.time() - project_start
            
            results[project_name] = result
            
            if result is not None:
                logger.info(f"{project_name} completed in {project_time:.2f} seconds")
            else:
                logger.error(f"{project_name} failed after {project_time:.2f} seconds")
        else:
            logger.error(f"Project directory not found: {project_path}")
            results[project_name] = None
    
    # Generate summary
    total_time = time.time() - start_time
    logger.info(f"Total execution time: {total_time:.2f} seconds")
    
    generate_summary_report(results)
    
    logger.info("=" * 60)
    logger.info("ML Projects Execution Completed")
    logger.info("=" * 60)

if __name__ == "__main__":
    main()
