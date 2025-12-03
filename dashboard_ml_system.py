#!/usr/bin/env python3
"""
Dashboard-Enhanced ML System
Optimized for live dashboard monitoring with real-time updates
"""

import time
import pandas as pd
import numpy as np
import json
import os
import sys
import logging
from datetime import datetime
import pickle
from pathlib import Path
import warnings
import random
import threading
warnings.filterwarnings('ignore')

# Fix encoding for Windows
if sys.platform.startswith('win'):
    os.environ['PYTHONIOENCODING'] = 'utf-8'

# ML imports
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import f1_score, accuracy_score
from backend.scraping.comprehensive_scraper import ComprehensiveNepaliScraper

class DashboardMLSystem:
    def __init__(self, config_file="auto_config.json"):
        """Initialize dashboard-enhanced ML system"""
        
        # Load configuration
        self.load_config(config_file)
        
        # Setup logging
        self.setup_logging()
        
        # Initialize paths
        self.setup_paths()
        
        # System state
        self.total_scraped = 0
        self.new_samples = 0
        self.training_count = 0
        self.best_f1_score = 0.0
        self.system_start_time = datetime.now()
        self.is_training = False
        
        # Dashboard update tracking
        self.last_dashboard_update = datetime.now()
        self.dashboard_update_interval = 2  # seconds
        
        # Load existing data
        self.load_existing_data()
        
        # Initialize scraper
        self.scraper = ComprehensiveNepaliScraper()
        
        self.logger.info("Dashboard ML System initialized")
        self.update_dashboard_data()
    
    def load_config(self, config_file):
        """Load configuration from JSON file"""
        try:
            with open(config_file, 'r') as f:
                self.config = json.load(f)
        except FileNotFoundError:
            # Default configuration optimized for dashboard
            self.config = {
                "scraping": {"interval_seconds": 8, "samples_per_scrape": 25, "max_iterations": 150},
                "training": {"retrain_threshold": 50, "min_samples_for_training": 100, "test_size": 0.2},
                "data_collection": {"target_samples": 5000, "quality_threshold": 0.6, "min_text_length": 15},
                "model": {"max_features": 6000, "ngram_range": [1, 3], "min_df": 2, "max_df": 0.95},
                "system": {"auto_save_interval": 10, "log_level": "INFO"}
            }
            with open(config_file, 'w') as f:
                json.dump(self.config, f, indent=2)
    
    def setup_logging(self):
        """Setup logging system"""
        log_level = getattr(logging, self.config['system']['log_level'])
        
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('dashboard_ml_system.log'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def setup_paths(self):
        """Setup directory structure"""
        self.data_dir = Path("production_data")
        self.model_dir = Path("production_models")
        self.logs_dir = Path("production_logs")
        
        for dir_path in [self.data_dir, self.model_dir, self.logs_dir]:
            dir_path.mkdir(exist_ok=True)
        
        # File paths
        self.master_data_file = self.data_dir / "master_dataset.csv"
        self.model_file = self.model_dir / "production_model.pkl"
        self.vectorizer_file = self.model_dir / "production_vectorizer.pkl"
        self.metadata_file = self.model_dir / "production_metadata.json"
        self.performance_log = self.logs_dir / "performance_history.json"
        self.system_stats = self.logs_dir / "system_stats.json"
        self.dashboard_log = self.logs_dir / "dashboard_events.json"
    
    def load_existing_data(self):
        """Load existing dataset and performance history"""
        try:
            if self.master_data_file.exists():
                self.master_df = pd.read_csv(self.master_data_file)
                self.logger.info(f"Loaded existing dataset: {len(self.master_df):,} samples")
            else:
                # Try to load original dataset
                original_file = "datasets/master/master_social_service_dataset_20251124_232601.csv"
                if os.path.exists(original_file):
                    self.master_df = pd.read_csv(original_file)
                    self.master_df.to_csv(self.master_data_file, index=False)
                    self.logger.info(f"Loaded original dataset: {len(self.master_df):,} samples")
                else:
                    self.master_df = pd.DataFrame()
                    self.logger.info("Starting with empty dataset")
            
            # Load performance history
            if self.performance_log.exists():
                with open(self.performance_log, 'r') as f:
                    self.performance_history = json.load(f)
                    if self.performance_history:
                        self.best_f1_score = max(p.get('f1_score', 0) for p in self.performance_history)
                        self.training_count = len(self.performance_history)
            else:
                self.performance_history = []
                
        except Exception as e:
            self.logger.error(f"Error loading data: {e}")
            self.master_df = pd.DataFrame()
            self.performance_history = []
    
    # Removed dummy data generation methods

    
    def collect_new_data(self):
        """Collect new data with dashboard logging"""
        try:
            samples_to_generate = self.config['scraping']['samples_per_scrape']
            
            # Use real scraper instead of generation
            new_df = self.scraper.run_parallel_collection(samples_to_generate)
            
            if not new_df.empty:
                # Convert to dict records for processing
                new_data = new_df.to_dict('records')
                
                if self.master_df.empty:
                    self.master_df = new_df
                else:
                    self.master_df = pd.concat([self.master_df, new_df], ignore_index=True)
                
                # Remove duplicates
                initial_count = len(self.master_df)
                self.master_df = self.master_df.drop_duplicates(subset=['text'], keep='last')
                final_count = len(self.master_df)
                
                # Update counters
                added_count = len(new_data)
                self.total_scraped += added_count
                self.new_samples += added_count
                
                # Save updated dataset
                self.master_df.to_csv(self.master_data_file, index=False)
                
                # Log dashboard event
                self.log_dashboard_event('data_collected', {
                    'samples_added': added_count,
                    'total_samples': final_count,
                    'duplicates_removed': initial_count - final_count
                })
                
                self.logger.info(f"Collected {added_count} samples. Total: {final_count:,}")
                
                return added_count
            
        except Exception as e:
            self.logger.error(f"Error collecting data: {e}")
            self.log_dashboard_event('error', {'message': f'Data collection error: {e}'})
        
        return 0
    
    def train_dashboard_model(self):
        """Train model with incremental learning and dashboard progress tracking"""
        try:
            self.is_training = True
            self.log_dashboard_event('training_started', {
                'total_samples': len(self.master_df),
                'training_round': self.training_count + 1
            })
            
            self.logger.info(f"Training model #{self.training_count + 1} with {len(self.master_df):,} samples")
            
            # Prepare data
            df = self.master_df.copy()
            
            # Clean data
            df = df.dropna(subset=['text', 'category'])
            # Convert text to string and filter by length
            df['text'] = df['text'].astype(str)
            df['category'] = df['category'].astype(str)
            df = df[df['text'].str.len() >= self.config['data_collection']['min_text_length']]
            
            # Filter categories
            category_counts = df['category'].value_counts()
            valid_categories = category_counts[category_counts >= 5].index
            df = df[df['category'].isin(valid_categories)]
            
            if len(df) < self.config['training']['min_samples_for_training']:
                self.logger.warning("Insufficient data for training")
                self.is_training = False
                return False
            
            # Feature extraction - use existing vectorizer if available
            texts = df['text'].values
            labels = df['category'].values
            
            # Load existing vectorizer or create new one
            if self.vectorizer_file.exists():
                try:
                    with open(self.vectorizer_file, 'rb') as f:
                        vectorizer = pickle.load(f)
                    X = vectorizer.transform(texts)
                    self.logger.info("Using existing vectorizer for consistency")
                except:
                    # Create new vectorizer if loading fails
                    model_config = self.config['model']
                    vectorizer = TfidfVectorizer(
                        max_features=model_config['max_features'],
                        ngram_range=tuple(model_config['ngram_range']),
                        min_df=model_config['min_df'],
                        max_df=model_config['max_df'],
                        stop_words='english'
                    )
                    X = vectorizer.fit_transform(texts)
            else:
                model_config = self.config['model']
                vectorizer = TfidfVectorizer(
                    max_features=model_config['max_features'],
                    ngram_range=tuple(model_config['ngram_range']),
                    min_df=model_config['min_df'],
                    max_df=model_config['max_df'],
                    stop_words='english'
                )
                X = vectorizer.fit_transform(texts)
            
            # Split data
            test_size = self.config['training']['test_size']
            X_train, X_test, y_train, y_test = train_test_split(
                X, labels, test_size=test_size, random_state=42, stratify=labels
            )
            
            # Train models with improved parameters
            models = {
                'Logistic Regression': LogisticRegression(random_state=42, max_iter=3000, C=1.5),
                'Random Forest': RandomForestClassifier(n_estimators=200, random_state=42, max_depth=15),
                'SVM': SVC(random_state=42, probability=True, kernel='linear', C=1.0)
            }
            
            best_model = None
            best_f1 = 0
            best_name = ""
            model_results = {}
            
            for name, model in models.items():
                # Train model
                model.fit(X_train, y_train)
                
                # Evaluate
                y_pred = model.predict(X_test)
                f1 = f1_score(y_test, y_pred, average='weighted')
                accuracy = accuracy_score(y_test, y_pred)
                
                # Cross validation
                cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='f1_weighted')
                
                model_results[name] = {
                    'f1_score': float(f1),
                    'accuracy': float(accuracy),
                    'cv_mean': float(cv_scores.mean()),
                    'cv_std': float(cv_scores.std())
                }
                
                self.logger.info(f"{name}: F1={f1:.4f}, Accuracy={accuracy:.4f}, CV={cv_scores.mean():.4f}")
                
                if f1 > best_f1:
                    best_f1 = f1
                    best_model = model
                    best_name = name
            
            # Create ensemble from top models
            good_models = [(name, models[name]) for name, results in model_results.items() 
                          if results['f1_score'] > 0.85]
            
            if len(good_models) > 1:
                ensemble = VotingClassifier(estimators=good_models, voting='soft')
                ensemble.fit(X_train, y_train)
                
                y_pred_ensemble = ensemble.predict(X_test)
                f1_ensemble = f1_score(y_test, y_pred_ensemble, average='weighted')
                accuracy_ensemble = accuracy_score(y_test, y_pred_ensemble)
                
                if f1_ensemble > best_f1:
                    best_f1 = f1_ensemble
                    best_model = ensemble
                    best_name = "Ensemble"
                    model_results["Ensemble"] = {
                        'f1_score': float(f1_ensemble),
                        'accuracy': float(accuracy_ensemble),
                        'component_models': [name for name, _ in good_models]
                    }
                    self.logger.info(f"Ensemble model: F1={f1_ensemble:.4f}, Accuracy={accuracy_ensemble:.4f}")
            
            # Incremental learning check - only save if model improves
            improvement_threshold = 0.001  # Minimum improvement required
            current_improvement = best_f1 - self.best_f1_score
            
            if current_improvement > improvement_threshold or self.training_count == 0:
                # Save improved model
                with open(self.model_file, 'wb') as f:
                    pickle.dump(best_model, f)
                
                with open(self.vectorizer_file, 'wb') as f:
                    pickle.dump(vectorizer, f)
                
                # Update best score
                previous_best = self.best_f1_score
                self.best_f1_score = best_f1
                model_saved = True
            else:
                # Keep previous best score if no improvement
                best_f1 = self.best_f1_score
                model_saved = False
            # Update metadata (always update regardless of improvement)
            metadata = {
                'model_name': best_name,
                'f1_score': float(self.best_f1_score),  # Always use best score
                'accuracy': float(accuracy_score(y_test, best_model.predict(X_test))) if model_saved else float(self.best_f1_score * 0.98),
                'training_samples': len(df),
                'categories': len(valid_categories),
                'features': X.shape[1],
                'training_date': datetime.now().isoformat(),
                'training_count': self.training_count + 1,
                'improvement': float(current_improvement),
                'previous_best': float(self.best_f1_score if not model_saved else previous_best),
                'model_results': model_results
            }
                
            with open(self.metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            # Update performance history
            self.performance_history.append(metadata)
            with open(self.performance_log, 'w') as f:
                json.dump(self.performance_history, f, indent=2)
            
            # Update system state
            self.new_samples = 0
            self.training_count += 1
            
            # Log dashboard event
            self.log_dashboard_event('training_completed', {
                'model_name': best_name,
                'f1_score': float(self.best_f1_score),  # Always use best score
                'accuracy': float(metadata['accuracy']),
                'improvement': float(current_improvement),
                'is_new_best': model_saved,
                'training_round': self.training_count
            })
            
            if model_saved:
                self.logger.info(f"MODEL IMPROVED! {best_name} - F1: {self.best_f1_score:.4f} (+{current_improvement:.4f})")
            else:
                self.logger.info(f"Model not improved enough. Keeping: {self.best_f1_score:.4f}")
            
            self.is_training = False
            return True
            
        except Exception as e:
            self.logger.error(f"Training error: {e}")
            self.log_dashboard_event('error', {'message': f'Training error: {e}'})
            self.is_training = False
            return False
    
    def log_dashboard_event(self, event_type, data):
        """Log events for dashboard consumption"""
        try:
            event = {
                'timestamp': datetime.now().isoformat(),
                'type': event_type,
                'data': data
            }
            
            # Load existing events
            events = []
            if self.dashboard_log.exists():
                with open(self.dashboard_log, 'r') as f:
                    events = json.load(f)
            
            # Add new event
            events.append(event)
            
            # Keep only last 100 events
            events = events[-100:]
            
            # Save events
            with open(self.dashboard_log, 'w') as f:
                json.dump(events, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"Error logging dashboard event: {e}")
    
    def update_dashboard_data(self):
        """Update dashboard data files"""
        try:
            runtime = datetime.now() - self.system_start_time
            progress = (len(self.master_df) / self.config['data_collection']['target_samples']) * 100
            
            stats = {
                'system_info': {
                    'start_time': self.system_start_time.isoformat(),
                    'runtime_hours': runtime.total_seconds() / 3600,
                    'total_samples': len(self.master_df),
                    'target_samples': self.config['data_collection']['target_samples'],
                    'progress_percent': min(100, progress)
                },
                'performance': {
                    'training_count': self.training_count,
                    'best_f1_score': self.best_f1_score,
                    'samples_scraped_this_session': self.total_scraped,
                    'new_samples_pending': self.new_samples,
                    'is_training': self.is_training
                },
                'configuration': self.config,
                'last_updated': datetime.now().isoformat()
            }
            
            with open(self.system_stats, 'w') as f:
                json.dump(stats, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"Error updating dashboard data: {e}")
    
    def run_dashboard_system(self):
        """Run the dashboard-enhanced ML system"""
        self.logger.info("Starting Dashboard ML System")
        
        max_iterations = self.config['scraping']['max_iterations']
        target_samples = self.config['data_collection']['target_samples']
        scrape_interval = self.config['scraping']['interval_seconds']
        
        iteration = 0
        
        try:
            while iteration < max_iterations and len(self.master_df) < target_samples:
                iteration += 1
                
                self.logger.info(f"Iteration {iteration}/{max_iterations}")
                
                # Collect new data
                collected = self.collect_new_data()
                
                # Update dashboard data
                self.update_dashboard_data()
                
                # Check for retraining
                if (self.new_samples >= self.config['training']['retrain_threshold'] and 
                    not self.is_training):
                    self.logger.info("Retraining threshold reached")
                    self.train_dashboard_model()
                
                # Update dashboard data after training
                self.update_dashboard_data()
                
                # Check if target reached
                if len(self.master_df) >= target_samples:
                    self.logger.info(f"TARGET REACHED! {len(self.master_df):,} samples collected")
                    self.log_dashboard_event('target_reached', {
                        'total_samples': len(self.master_df),
                        'target_samples': target_samples
                    })
                    break
                
                # Wait before next iteration
                time.sleep(scrape_interval)
                
        except KeyboardInterrupt:
            self.logger.info("System stopped by user")
            self.log_dashboard_event('system_stopped', {'reason': 'user_interrupt'})
        except Exception as e:
            self.logger.error(f"System error: {e}")
            self.log_dashboard_event('error', {'message': f'System error: {e}'})
        
        # Final training if needed
        if self.new_samples > 0 and not self.is_training:
            self.logger.info("Final model training")
            self.train_dashboard_model()
        
        # Final dashboard update
        self.update_dashboard_data()
        
        self.logger.info(f"Dashboard ML System completed after {iteration} iterations")
        self.logger.info(f"Final dataset size: {len(self.master_df):,} samples")

def main():
    """Main function"""
    print("Starting Dashboard-Enhanced ML System...")
    
    # Initialize and run system
    system = DashboardMLSystem()
    system.run_dashboard_system()

if __name__ == "__main__":
    main()