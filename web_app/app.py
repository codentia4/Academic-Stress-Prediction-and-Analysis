"""
Academic Stress Prediction - Complete Application
"""

from flask import Flask, render_template, request, jsonify, redirect, url_for, flash, session
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
import pandas as pd
import numpy as np
import joblib
import json
import sys
import os
from datetime import datetime
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import plotly.express as px
import plotly.graph_objects as go
import plotly.utils
from textblob import TextBlob
from wordcloud import WordCloud
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

app = Flask(__name__)
app.config['SECRET_KEY'] = 'stress-predictor-secret-key-2025'

# Flask-Login setup
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

USERS_FILE = 'users.json'
ACTIVITY_FILE = 'user_activity.json'

def load_users():
    if os.path.exists(USERS_FILE):
        with open(USERS_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_users(users):
    with open(USERS_FILE, 'w') as f:
        json.dump(users, f, indent=2)

def load_activities():
    if os.path.exists(ACTIVITY_FILE):
        with open(ACTIVITY_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_activities(activities):
    with open(ACTIVITY_FILE, 'w') as f:
        json.dump(activities, f, indent=2)

def log_activity(user_id, activity_type, details=None):
    activities = load_activities()
    if user_id not in activities:
        activities[user_id] = []
    
    activities[user_id].append({
        'type': activity_type,
        'details': details,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    })
    
    activities[user_id] = activities[user_id][-50:]
    save_activities(activities)

class User(UserMixin):
    def __init__(self, user_id, username, email, role='user'):
        self.id = user_id
        self.username = username
        self.email = email
        self.role = role

@login_manager.user_loader
def load_user(user_id):
    users = load_users()
    if user_id in users:
        u = users[user_id]
        return User(user_id, u['username'], u['email'], u.get('role', 'user'))
    return None

# Load models
try:
    model = joblib.load('../models/model.pkl')
    scaler = joblib.load('../models/scaler.pkl')
    encoder = joblib.load('../models/encoder.pkl')
    with open('../models/metrics.json') as f:
        metrics = json.load(f)
    print("✓ Models loaded")
except Exception as e:
    model = scaler = encoder = None
    metrics = {}
    print(f"⚠ Models not loaded: {e}")

# Load dataset
try:
    # Try multiple possible paths
    possible_paths = [
        'academic_stress_dataset.csv',  # Same folder as app.py
        './academic_stress_dataset.csv',  # Explicit current directory
        '../data/raw/academic_stress_dataset.csv',
        '../data/raw/Academic_Stress_Dataset.csv',
        'data/raw/academic_stress_dataset.csv',
        'data/raw/Academic_Stress_Dataset.csv',
        '../academic_stress_dataset.csv',
        'Academic_Stress_Dataset.csv',
        '../Academic_Stress_Dataset.csv'
    ]
    
    df = None
    for path in possible_paths:
        if os.path.exists(path):
            df = pd.read_csv(path)
            print(f"✓ Dataset loaded from: {path} ({len(df)} records)")
            break
    
    if df is None:
        print("⚠ Dataset not found! Tried these locations:")
        for path in possible_paths:
            abs_path = os.path.abspath(path)
            exists = "✓" if os.path.exists(path) else "✗"
            print(f"  {exists} {abs_path}")
        raise FileNotFoundError("Dataset not found in any expected location")
    
    # Ensure stress_level column has correct values (force recreate for consistency)
    if 'stress_score' in df.columns:
        # Always recreate stress_level based on stress_score (0-100 scale)
        def categorize_stress(score):
            if score <= 40:
                return 'Low'
            elif score <= 70:
                return 'Medium'
            else:
                return 'High'
        
        df['stress_level'] = df['stress_score'].apply(categorize_stress)
        print("  ✓ Created/Updated stress_level column from stress_score (0-100 scale)")
    else:
        print(f"  ⚠ No stress_score column! Available: {df.columns.tolist()}")
    
    print(f"✓ Dataset loaded ({len(df)} records)")
    if 'stress_level' in df.columns:
        print(f"  Stress levels: {df['stress_level'].value_counts().to_dict()}")
        # Ensure only valid values
        unique_levels = df['stress_level'].unique()
        print(f"  Unique values: {sorted(unique_levels.tolist())}")
except Exception as e:
    df = None
    print(f"⚠ Dataset not loaded: {e}")
    print(f"  Current working directory: {os.getcwd()}")
    print(f"  Please ensure the dataset is in the correct location")

@app.route('/')
def landing():
    if current_user.is_authenticated:
        return redirect(url_for('home'))
    return render_template('landing.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('home'))
    
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        
        users = load_users()
        
        for user_id, user_data in users.items():
            if user_data['email'] == email:
                flash('Email already registered!', 'error')
                return redirect(url_for('register'))
        
        user_id = str(len(users) + 1)
        
        # First user becomes admin
        is_first_user = len(users) == 0
        
        users[user_id] = {
            'username': username,
            'email': email,
            'password': generate_password_hash(password),
            'registered_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'role': 'admin' if is_first_user else 'user',
            'predictions_count': 0,
            'last_login': None
        }
        
        save_users(users)
        
        if is_first_user:
            flash('Registration successful! You are the admin.', 'success')
        else:
            flash('Registration successful! Please login.', 'success')
        
        return redirect(url_for('login'))
    
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('home'))
    
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        
        users = load_users()
        
        for user_id, user_data in users.items():
            if user_data['email'] == email:
                if check_password_hash(user_data['password'], password):
                    users[user_id]['last_login'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    save_users(users)
                    
                    user = User(user_id, user_data['username'], user_data['email'], user_data.get('role', 'user'))
                    login_user(user)
                    
                    log_activity(user_id, 'login', {'email': email})
                    
                    flash(f'Welcome back, {user_data["username"]}!', 'success')
                    return redirect(url_for('home'))
                else:
                    flash('Invalid password!', 'error')
                    return redirect(url_for('login'))
        
        flash('Email not found!', 'error')
        return redirect(url_for('login'))
    
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    log_activity(current_user.id, 'logout')
    logout_user()
    flash('Logged out successfully!', 'success')
    return redirect(url_for('landing'))

@app.route('/home')
@login_required
def home():
    log_activity(current_user.id, 'page_view', {'page': 'home'})
    return render_template('home.html', username=current_user.username, role=current_user.role)

@app.route('/dashboard')
@login_required
def dashboard():
    log_activity(current_user.id, 'page_view', {'page': 'dashboard'})
    
    if df is None:
        return "Dataset not found!", 404
    
    # Ensure stress_level exists with correct categorization (0-100 scale)
    if 'stress_score' in df.columns:
        def categorize_stress(score):
            if score <= 40:
                return 'Low'
            elif score <= 70:
                return 'Medium'
            else:
                return 'High'
        df['stress_level'] = df['stress_score'].apply(categorize_stress)
    
    # Validate stress_level column exists
    if 'stress_level' not in df.columns:
        return "Error: Dataset missing required 'stress_level' column. Please check data processing.", 500
    
    graphs = []
    
    fig1 = px.pie(df, names='stress_level', title='Stress Distribution',
                  color='stress_level',
                  color_discrete_map={'Low':'#10b981', 'Medium':'#f59e0b', 'High':'#ef4444'},
                  hole=0.4)
    graphs.append(json.dumps(fig1, cls=plotly.utils.PlotlyJSONEncoder))
    
    year_avg = df.groupby('year_of_study')['stress_score'].mean().reset_index()
    fig2 = px.bar(year_avg, x='year_of_study', y='stress_score',
                  title='Average Stress by Year', color='stress_score',
                  color_continuous_scale='Reds')
    graphs.append(json.dumps(fig2, cls=plotly.utils.PlotlyJSONEncoder))
    
    fig3 = px.scatter(df, x='study_hours_per_day', y='sleep_hours',
                      color='stress_level', size='stress_score',
                      title='Study Hours vs Sleep',
                      color_discrete_map={'Low':'#10b981', 'Medium':'#f59e0b', 'High':'#ef4444'})
    graphs.append(json.dumps(fig3, cls=plotly.utils.PlotlyJSONEncoder))
    
    fig4 = px.violin(df, x='stress_level', y='cgpa', title='CGPA by Stress',
                     color='stress_level', box=True,
                     color_discrete_map={'Low':'#10b981', 'Medium':'#f59e0b', 'High':'#ef4444'})
    graphs.append(json.dumps(fig4, cls=plotly.utils.PlotlyJSONEncoder))
    
    stats = {
        'total': len(df),
        'avg_stress': round(df['stress_score'].mean(), 1),
        'high_count': len(df[df['stress_level']=='High']),
        'avg_cgpa': round(df['cgpa'].mean(), 2)
    }
    
    return render_template('dashboard.html', graphs=graphs, stats=stats, 
                         metrics=metrics, username=current_user.username, role=current_user.role)

@app.route('/predict')
@login_required
def predict_page():
    log_activity(current_user.id, 'page_view', {'page': 'predict'})
    return render_template('predict.html', username=current_user.username, role=current_user.role)

@app.route('/api/predict', methods=['POST'])
@login_required
def predict_api():
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        data = request.json
        features = pd.DataFrame([{
            'age': int(data['age']),
            'gender': data['gender'],
            'year_of_study': int(data['year_of_study']),
            'cgpa': float(data['cgpa']),
            'study_hours_per_day': int(data['study_hours_per_day']),
            'sleep_hours': float(data['sleep_hours']),
            'extracurricular_activities': data['extracurricular_activities'],
            'social_support': data['social_support'],
            'financial_stress': data['financial_stress'],
            'part_time_job': data['part_time_job'],
            'family_expectations': data['family_expectations'],
            'relationship_status': data['relationship_status'],
            'physical_activity': data['physical_activity'],
            'screen_time_hours': int(data['screen_time_hours']),
            'assignment_load': data['assignment_load'],
            'exam_frequency': int(data['exam_frequency']),
            'peer_pressure': data['peer_pressure'],
            'diet_quality': data['diet_quality'],
            'meditation_practice': data['meditation_practice']
        }])
        
        for col in features.select_dtypes(include=['object']).columns:
            features[col] = pd.Categorical(features[col]).codes
        
        features_scaled = scaler.transform(features)
        pred = model.predict(features_scaled)[0]
        proba = model.predict_proba(features_scaled)[0]
        
        result = encoder.classes_[pred]
        confidence = {encoder.classes_[i]: round(float(proba[i])*100, 1) 
                     for i in range(len(encoder.classes_))}
        
        log_activity(current_user.id, 'prediction', {
            'result': result,
            'confidence': confidence
        })
        
        users = load_users()
        users[current_user.id]['predictions_count'] = users[current_user.id].get('predictions_count', 0) + 1
        save_users(users)
        
        recs = generate_recommendations(data, result)
        
        return jsonify({
            'stress_level': result,
            'confidence': confidence,
            'recommendations': recs
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def generate_recommendations(features, result):
    recs = []
    if result == 'High':
        recs = ["🚨 High stress detected. Immediate action needed.",
               "😴 Get 7-8 hours sleep nightly.",
               "🏃 Exercise 30+ minutes daily.",
               "🧘 Practice meditation for 10-15 minutes.",
               "👥 Reach out to counselor or support groups."]
    elif result == 'Medium':
        recs = ["⚠️ Moderate stress. Monitor carefully.",
               "😴 Maintain 7-8 hours sleep.",
               "🎯 Practice time management.",
               "🏃 Regular physical activity."]
    else:
        recs = ["✅ Excellent! Managing stress well.",
               "💪 Keep maintaining healthy habits.",
               "🌟 Continue monitoring your well-being."]
    
    return recs

@app.route('/analysis')
@login_required
def analysis():
    log_activity(current_user.id, 'page_view', {'page': 'analysis'})
    
    if df is None:
        return "Dataset not found!", 404
    
    # Ensure stress_level exists with correct categorization (0-100 scale)
    if 'stress_score' in df.columns:
        def categorize_stress(score):
            if score <= 40:
                return 'Low'
            elif score <= 70:
                return 'Medium'
            else:
                return 'High'
        df['stress_level'] = df['stress_score'].apply(categorize_stress)
    
    # Validate stress_level column exists
    if 'stress_level' not in df.columns:
        return "Error: Dataset missing required 'stress_level' column. Please check data processing.", 500
    
    try:
        sentiments = []
        subjectivity = []
        
        for feedback in df['student_feedback']:
            blob = TextBlob(str(feedback))
            sentiments.append(blob.sentiment.polarity)
            subjectivity.append(blob.sentiment.subjectivity)
        
        df_temp = df.copy()
        df_temp['sentiment'] = sentiments
        df_temp['subjectivity'] = subjectivity
        
        graphs = []
        
        # Chart 1: Box Plot
        fig1 = px.box(df_temp, x='stress_level', y='sentiment',
                      title='Sentiment by Stress Level',
                      color='stress_level',
                      color_discrete_map={'Low':'#10b981', 'Medium':'#f59e0b', 'High':'#ef4444'})
        graphs.append(json.dumps(fig1, cls=plotly.utils.PlotlyJSONEncoder))
        
        # Chart 2: Scatter
        fig2 = px.scatter(df_temp, x='sentiment', y='subjectivity',
                         color='stress_level', size='stress_score',
                         title='Sentiment vs Subjectivity',
                         color_discrete_map={'Low':'#10b981', 'Medium':'#f59e0b', 'High':'#ef4444'})
        graphs.append(json.dumps(fig2, cls=plotly.utils.PlotlyJSONEncoder))
        
        # Chart 3: Histogram
        fig3 = px.histogram(df_temp, x='sentiment', color='stress_level',
                            title='Sentiment Distribution',
                            color_discrete_map={'Low':'#10b981', 'Medium':'#f59e0b', 'High':'#ef4444'})
        graphs.append(json.dumps(fig3, cls=plotly.utils.PlotlyJSONEncoder))
        
        # Generate WordClouds
        wordclouds = {}
        for level in ['Low', 'Medium', 'High']:
            texts = ' '.join(df[df['stress_level']==level]['student_feedback'].astype(str))
            wc = WordCloud(width=800, height=400, background_color='white',
                          colormap='RdYlGn_r' if level=='High' else 'YlGn',
                          max_words=50).generate(texts)
            
            # Create new figure and axis
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.imshow(wc, interpolation='bilinear')
            ax.axis('off')
            plt.tight_layout(pad=0)
            
            wc_path = f'static/wordcloud_{level.lower()}.png'
            os.makedirs('static', exist_ok=True)
            plt.savefig(wc_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            wordclouds[level] = f'/static/wordcloud_{level.lower()}.png'
        
        stats = {
            'avg_sentiment_low': round(df_temp[df_temp['stress_level']=='Low']['sentiment'].mean(), 3),
            'avg_sentiment_med': round(df_temp[df_temp['stress_level']=='Medium']['sentiment'].mean(), 3),
            'avg_sentiment_high': round(df_temp[df_temp['stress_level']=='High']['sentiment'].mean(), 3),
            'avg_subjectivity': round(df_temp['subjectivity'].mean(), 3),
            'total_analyzed': len(df_temp)
        }
        
        samples = {
            'Low': df[df['stress_level']=='Low']['student_feedback'].head(5).tolist(),
            'Medium': df[df['stress_level']=='Medium']['student_feedback'].head(5).tolist(),
            'High': df[df['stress_level']=='High']['student_feedback'].head(5).tolist()
        }
        
        return render_template('analysis.html', graphs=graphs, samples=samples,
                             wordclouds=wordclouds, stats=stats, 
                             username=current_user.username, role=current_user.role)
    
    except Exception as e:
        return f"Error in analysis: {str(e)}", 500

@app.route('/profile')
@login_required
def profile():
    log_activity(current_user.id, 'page_view', {'page': 'profile'})
    
    users = load_users()
    user_data = users[current_user.id]
    activities = load_activities().get(current_user.id, [])
    
    activity_stats = {
        'total_activities': len(activities),
        'predictions_count': user_data.get('predictions_count', 0),
        'page_views': len([a for a in activities if a['type'] == 'page_view']),
        'logins': len([a for a in activities if a['type'] == 'login'])
    }
    
    predictions = [a for a in activities if a['type'] == 'prediction']
    recent_activities = activities[-10:][::-1]
    
    return render_template('profile.html', 
                         user=user_data,
                         activities=recent_activities,
                         predictions=predictions[-5:][::-1],
                         stats=activity_stats,
                         username=current_user.username,
                         role=current_user.role)

@app.route('/admin')
@login_required
def admin_dashboard():
    if current_user.role != 'admin':
        flash('Access denied! Admin only.', 'error')
        return redirect(url_for('profile'))
    
    log_activity(current_user.id, 'admin_access')
    
    users = load_users()
    activities = load_activities()
    
    user_list = []
    for uid, udata in users.items():
        user_activities = activities.get(uid, [])
        user_list.append({
            'id': uid,
            'username': udata['username'],
            'email': udata['email'],
            'registered_at': udata.get('registered_at', 'N/A'),
            'last_login': udata.get('last_login', 'Never'),
            'role': udata.get('role', 'user'),
            'predictions_count': udata.get('predictions_count', 0),
            'total_activities': len(user_activities)
        })
    
    total_predictions = sum([u.get('predictions_count', 0) for u in users.values()])
    total_activities = sum([len(a) for a in activities.values()])
    
    admin_stats = {
        'total_users': len(users),
        'admin_count': len([u for u in users.values() if u.get('role') == 'admin']),
        'user_count': len([u for u in users.values() if u.get('role') == 'user']),
        'total_predictions': total_predictions,
        'total_activities': total_activities
    }
    
    return render_template('admin.html', 
                         users=user_list,
                         stats=admin_stats,
                         username=current_user.username,
                         role=current_user.role)

if __name__ == '__main__':
    print("\n" + "="*70)
    print("  🎓 STRESS PREDICTOR - COMPLETE VERSION")
    print("="*70)
    print("\n🚀 Server: http://localhost:5000")
    print("="*70 + "\n")
    app.run(debug=True, port=5000)