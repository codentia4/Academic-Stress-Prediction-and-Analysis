"""
Academic Stress Prediction System - Model Training
Chaithanya D | 24SUPMCAGL015
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import xgboost as xgb
import joblib
import json
import os
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("  ACADEMIC STRESS PREDICTION - MODEL TRAINING")
print("="*70)

# Load Dataset
print("\n[1/6] Loading Dataset...")
df = pd.read_csv('data/raw/academic_stress_dataset.csv')
print(f"✓ Loaded {len(df)} records with {df.shape[1]} columns")
print(f"\nStress Distribution:\n{df['stress_level'].value_counts()}")

# Create visualizations folder
os.makedirs('visualizations', exist_ok=True)

# EDA - Visualization 1: Stress Distribution
print("\n[2/6] Creating Visualizations...")
plt.figure(figsize=(10, 6))
colors = ['#10b981', '#f59e0b', '#ef4444']
df['stress_level'].value_counts().plot(kind='pie', autopct='%1.1f%%', 
                                        colors=colors, startangle=90)
plt.title('Stress Level Distribution', fontsize=14, weight='bold')
plt.ylabel('')
plt.savefig('visualizations/stress_distribution.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ stress_distribution.png")

# Visualization 2: Correlation Heatmap
plt.figure(figsize=(12, 8))
numeric_cols = df.select_dtypes(include=[np.number]).columns
corr = df[numeric_cols].corr()
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0, 
            square=True, linewidths=0.5)
plt.title('Correlation Heatmap', fontsize=14, weight='bold')
plt.tight_layout()
plt.savefig('visualizations/correlation_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ correlation_heatmap.png")

# Visualization 3: Feature Analysis
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
scatter_features = [
    ('cgpa', 'CGPA vs Stress'),
    ('study_hours_per_day', 'Study Hours vs Stress'),
    ('sleep_hours', 'Sleep Hours vs Stress'),
    ('screen_time_hours', 'Screen Time vs Stress')
]
for idx, (feature, title) in enumerate(scatter_features):
    ax = axes[idx // 2, idx % 2]
    ax.scatter(df[feature], df['stress_score'], c=df['stress_score'], 
               cmap='RdYlGn_r', alpha=0.6, edgecolors='black', linewidth=0.5)
    ax.set_xlabel(feature.replace('_', ' ').title(), fontweight='bold')
    ax.set_ylabel('Stress Score', fontweight='bold')
    ax.set_title(title, fontweight='bold')
    ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('visualizations/feature_analysis.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ feature_analysis.png")

# Data Preprocessing
print("\n[3/6] Preprocessing Data...")
df_model = df.drop(['student_id', 'student_feedback', 'stress_score'], axis=1)

# Encode categorical variables
cat_cols = df_model.select_dtypes(include=['object']).columns.tolist()
cat_cols.remove('stress_level')
for col in cat_cols:
    df_model[col] = pd.Categorical(df_model[col]).codes

# Prepare X and y
X = df_model.drop('stress_level', axis=1)
y = df_model['stress_level']

# Encode target
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"✓ Train: {len(X_train)}, Test: {len(X_test)}, Features: {X_train.shape[1]}")

# Train Models
print("\n[4/6] Training Models...")
models = {
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'XGBoost': xgb.XGBClassifier(n_estimators=100, random_state=42, eval_metric='mlogloss'),
    'Gradient Boost': GradientBoostingClassifier(n_estimators=100, random_state=42),
    'SVM': SVC(kernel='rbf', probability=True, random_state=42)
}

results = {}
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    results[name] = {'model': model, 'accuracy': acc}
    print(f"  {name}: {acc:.4f}")

# Select best model
best_name = max(results, key=lambda x: results[x]['accuracy'])
best_model = results[best_name]['model']
print(f"\n✓ Best Model: {best_name} ({results[best_name]['accuracy']:.4f})")

# Model Comparison Chart
plt.figure(figsize=(10, 6))
names = list(results.keys())
accs = [results[n]['accuracy'] for n in names]
bars = plt.bar(names, accs, color=['#3b82f6', '#10b981', '#f59e0b', '#ef4444'])
plt.ylabel('Accuracy', fontweight='bold')
plt.title('Model Performance Comparison', fontsize=14, weight='bold')
plt.ylim([0.75, 1.0])
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.3f}', ha='center', va='bottom', fontweight='bold')
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('visualizations/model_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ model_comparison.png")

# Evaluate Best Model
print("\n[5/6] Evaluating Best Model...")
y_pred = best_model.predict(X_test_scaled)
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.title('Confusion Matrix', fontsize=14, weight='bold')
plt.ylabel('True Label', fontweight='bold')
plt.xlabel('Predicted Label', fontweight='bold')
plt.tight_layout()
plt.savefig('visualizations/confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ confusion_matrix.png")

# Save Models
print("\n[6/6] Saving Models...")
os.makedirs('models', exist_ok=True)
joblib.dump(best_model, 'models/model.pkl')
joblib.dump(scaler, 'models/scaler.pkl')
joblib.dump(le, 'models/encoder.pkl')

metrics = {
    'model_name': best_name,
    'accuracy': float(results[best_name]['accuracy']),
    'train_size': int(len(X_train)),
    'test_size': int(len(X_test))
}
with open('models/metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)

print("✓ model.pkl, scaler.pkl, encoder.pkl, metrics.json saved")

print("\n" + "="*70)
print("  TRAINING COMPLETE! ✓")
print("="*70)
print("\nNext: Run the web app")
print("  cd web_app")
print("  python app.py")