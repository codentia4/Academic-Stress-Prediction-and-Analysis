"""
Configuration file for Academic Stress Prediction System
"""

import os

# Base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Data paths
DATA_DIR = os.path.join(BASE_DIR, 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
DATASET_PATH = os.path.join(RAW_DATA_DIR, 'academic_stress_dataset.csv')

# Model paths
MODEL_DIR = os.path.join(BASE_DIR, 'models')
MODEL_PATH = os.path.join(MODEL_DIR, 'stress_predictor.pkl')
SCALER_PATH = os.path.join(MODEL_DIR, 'feature_scaler.pkl')
METRICS_PATH = os.path.join(MODEL_DIR, 'model_metrics.json')

# Visualization paths
VIZ_DIR = os.path.join(BASE_DIR, 'visualizations')

# Dataset configuration
RANDOM_SEED = 42
TEST_SIZE = 0.2

# Stress level thresholds
STRESS_THRESHOLDS = {
    'low': (0, 40),
    'medium': (40, 70),
    'high': (70, 100)
}

# Color schemes for visualizations
COLORS = {
    'low_stress': '#2ecc71',
    'medium_stress': '#f39c12',
    'high_stress': '#e74c3c',
    'primary': '#667eea',
    'secondary': '#764ba2'
}