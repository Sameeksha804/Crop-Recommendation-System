from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os
import json

app = Flask(__name__)
CORS(app)

# Load the trained model and scaler
try:
    model = joblib.load('model/crop_model.pkl')
    scaler = joblib.load('model/scaler.pkl')
except:
    # If model doesn't exist, train a new one
    model = None
    scaler = None

def add_features(df):
    # Create a copy to avoid modifying the original dataframe
    df = df.copy()
    
    # Add interaction features
    df['NP_ratio'] = df['N'] / df['P']
    df['NK_ratio'] = df['N'] / df['K']
    df['PK_ratio'] = df['P'] / df['K']
    
    # Add polynomial features for important columns
    df['temperature_squared'] = df['temperature'] ** 2
    df['humidity_squared'] = df['humidity'] ** 2
    df['rainfall_squared'] = df['rainfall'] ** 2
    
    # Add combined features
    df['temp_humidity'] = df['temperature'] * df['humidity']
    df['temp_rainfall'] = df['temperature'] * df['rainfall']
    
    # Ensure consistent column order
    feature_columns = [
        'N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall',
        'NP_ratio', 'NK_ratio', 'PK_ratio',
        'temperature_squared', 'humidity_squared', 'rainfall_squared',
        'temp_humidity', 'temp_rainfall'
    ]
    
    return df[feature_columns]

def train_model():
    print("Loading datasets...")
    try:
        # Load both datasets
        df1 = pd.read_csv('Crop_recommendation.csv')
        df2 = pd.read_csv('Crop_recommendation1.csv')
        
        # Combine datasets
        df = pd.concat([df1, df2], ignore_index=True)
        
        # Remove any duplicate entries
        df = df.drop_duplicates()
        
        print(f"Combined dataset loaded successfully with {len(df)} samples")
        print(f"Number of unique crops: {df['label'].nunique()}")
        print("\nCrop distribution:")
        print(df['label'].value_counts())
    except FileNotFoundError as e:
        print(f"Error loading datasets: {str(e)}")
        return None
    
    # Add engineered features
    print("\nAdding engineered features...")
    X = add_features(df)
    y = df['label']
    
    print("\nSplitting data into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print("Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save feature names for later use
    feature_names = X.columns.tolist()
    with open('model/feature_names.json', 'w') as f:
        json.dump(feature_names, f)
    
    print("\nTraining Random Forest model with GridSearchCV...")
    # Define parameter grid for GridSearchCV
    param_grid = {
        'n_estimators': [300, 400, 500],
        'max_depth': [20, 25, 30],
        'min_samples_split': [2, 4, 6],
        'min_samples_leaf': [1, 2, 3],
        'max_features': ['sqrt', 'log2'],
        'class_weight': ['balanced', None]
    }
    
    # Initialize base model
    base_model = RandomForestClassifier(random_state=42, n_jobs=-1)
    
    # Perform GridSearchCV
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=5,
        n_jobs=-1,
        verbose=1,
        scoring='accuracy'
    )
    
    grid_search.fit(X_train_scaled, y_train)
    
    # Get best model
    model = grid_search.best_estimator_
    
    print("\nBest parameters found:")
    print(grid_search.best_params_)
    
    print("\nEvaluating model performance...")
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    # Perform cross-validation with more folds
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=10)
    
    # Get feature importance
    feature_importance = dict(zip(feature_names, model.feature_importances_))
    sorted_features = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))
    
    # Save evaluation metrics
    evaluation_metrics = {
        'accuracy': float(accuracy),
        'cv_scores_mean': float(cv_scores.mean()),
        'cv_scores_std': float(cv_scores.std()),
        'classification_report': report,
        'confusion_matrix': conf_matrix.tolist(),
        'feature_importance': sorted_features,
        'best_parameters': grid_search.best_params_,
        'dataset_info': {
            'total_samples': len(df),
            'unique_crops': int(df['label'].nunique()),
            'crop_distribution': df['label'].value_counts().to_dict()
        }
    }
    
    print("\nSaving model and metrics...")
    os.makedirs('model', exist_ok=True)
    joblib.dump(model, 'model/crop_model.pkl')
    joblib.dump(scaler, 'model/scaler.pkl')
    with open('model/evaluation_metrics.json', 'w') as f:
        json.dump(evaluation_metrics, f)
    
    print("\nModel Training Complete!")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Cross-validation scores: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    print("\nTop 5 Most Important Features:")
    for feature, importance in list(sorted_features.items())[:5]:
        print(f"{feature}: {importance:.4f}")
    
    return evaluation_metrics

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        
        # Create base features
        features = pd.DataFrame([[
            float(data['N']),
            float(data['P']),
            float(data['K']),
            float(data['temperature']),
            float(data['humidity']),
            float(data['ph']),
            float(data['rainfall'])
        ]], columns=['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall'])
        
        # Add engineered features
        features['NP_ratio'] = features['N'] / features['P']
        features['NK_ratio'] = features['N'] / features['K']
        features['PK_ratio'] = features['P'] / features['K']
        features['temperature_squared'] = features['temperature'] ** 2
        features['humidity_squared'] = features['humidity'] ** 2
        features['rainfall_squared'] = features['rainfall'] ** 2
        features['temp_humidity'] = features['temperature'] * features['humidity']
        features['temp_rainfall'] = features['temperature'] * features['rainfall']
        
        # Ensure consistent column order
        feature_columns = [
            'N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall',
            'NP_ratio', 'NK_ratio', 'PK_ratio',
            'temperature_squared', 'humidity_squared', 'rainfall_squared',
            'temp_humidity', 'temp_rainfall'
        ]
        features = features[feature_columns]
        
        # Scale the features
        features_scaled = scaler.transform(features)
        
        # Make prediction
        prediction = model.predict(features_scaled)[0]
        probabilities = model.predict_proba(features_scaled)[0]
        max_probability = max(probabilities)
        
        # Get top 3 predictions
        top_3_idx = np.argsort(probabilities)[-3:][::-1]
        top_3_predictions = [
            {
                'crop': model.classes_[idx],
                'confidence': float(probabilities[idx])
            }
            for idx in top_3_idx
        ]
        
        return jsonify({
            'status': 'success',
            'predictions': top_3_predictions,
            'best_match': {
                'crop': prediction,
                'confidence': float(max_probability)
            }
        })
    
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        })

@app.route('/model-metrics')
def get_model_metrics():
    try:
        with open('model/evaluation_metrics.json', 'r') as f:
            metrics = json.load(f)
        return jsonify(metrics)
    except FileNotFoundError:
        return jsonify({
            'status': 'error',
            'message': 'Model metrics not found. Please train the model first.'
        })

if __name__ == '__main__':
    if model is None:
        metrics = train_model()
        print("\nModel Evaluation Metrics:")
        print(f"Accuracy: {metrics['accuracy']:.2f}")
        print(f"Cross-validation scores: {metrics['cv_scores_mean']:.2f} (+/- {metrics['cv_scores_std']:.2f})")
        print("\nClassification Report:")
        print(json.dumps(metrics['classification_report'], indent=2))
    app.run(debug=True) 