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
import logging
import traceback

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Configure CORS
CORS(app, resources={
    r"/*": {
        "origins": ["*"],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization", "Accept"],
        "expose_headers": ["Content-Type", "Authorization"],
        "supports_credentials": True,
        "max_age": 600
    }
})

@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    response.headers.add('Access-Control-Allow-Credentials', 'true')
    return response

def add_features(df):
    try:
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
        
        # Verify all required columns are present
        missing_columns = [col for col in feature_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
            
        return df[feature_columns]
    except Exception as e:
        logger.error(f"Error in add_features: {str(e)}")
        logger.exception("Full traceback:")
        raise

def train_model():
    print("Loading dataset...")
    try:
        # Load dataset
        df = pd.read_csv('data/Crop_recommendation.csv')
        
        print(f"Dataset loaded successfully with {len(df)} samples")
        print(f"Number of unique crops: {df['label'].nunique()}")
        print("\nCrop distribution:")
        print(df['label'].value_counts())
    except FileNotFoundError as e:
        print(f"Error loading dataset: {str(e)}")
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

def load_model_and_scaler():
    try:
        logger.info("Attempting to load model and scaler...")
        model_path = os.path.join('model', 'crop_model.pkl')
        scaler_path = os.path.join('model', 'scaler.pkl')
        
        if not os.path.exists(model_path) or not os.path.exists(scaler_path):
            logger.info("Model or scaler files not found, will train new model")
            return None, None
            
        try:
            model = joblib.load(model_path)
            scaler = joblib.load(scaler_path)
            
            # Verify model and scaler are loaded correctly
            if model is None or scaler is None:
                logger.error("Model or scaler loaded but is None")
                return None, None
                
            logger.info("Model and scaler loaded successfully")
            logger.info(f"Model type: {type(model)}")
            logger.info(f"Scaler type: {type(scaler)}")
            return model, scaler
        except Exception as e:
            logger.warning(f"Error loading existing model: {str(e)}")
            logger.info("Will train new model")
            return None, None
            
    except Exception as e:
        logger.error(f"Error in load_model_and_scaler: {str(e)}")
        logger.exception("Full traceback:")
        return None, None

# Initialize model and scaler
model, scaler = load_model_and_scaler()

# If model is not loaded, train a new one
if model is None or scaler is None:
    logger.info("Training new model...")
    try:
        metrics = train_model()
        if metrics is None:
            raise Exception("Failed to train model")
        model, scaler = load_model_and_scaler()
        if model is None or scaler is None:
            raise Exception("Failed to load model after training")
        logger.info("New model trained and loaded successfully")
    except Exception as e:
        logger.error(f"Error training new model: {str(e)}")
        logger.exception("Full traceback:")

@app.route('/')
def home():
    try:
        logger.info("Rendering home page...")
        return render_template('index.html')
    except Exception as e:
        logger.error(f"Error rendering home page: {str(e)}")
        return str(e), 500

@app.route('/predict', methods=['POST', 'OPTIONS'])
def predict():
    if request.method == 'OPTIONS':
        return '', 200
        
    try:
        # Check if model and scaler are loaded
        if model is None or scaler is None:
            logger.error("Model or scaler not loaded")
            return jsonify({
                'status': 'error',
                'message': 'Model is not ready. Please try again in a few moments.'
            }), 503

        # Log the raw request data
        logger.info(f"Raw request data: {request.get_data()}")
        
        # Check if request has JSON data
        if not request.is_json:
            logger.error("Request does not contain JSON data")
            return jsonify({
                'status': 'error',
                'message': 'Request must contain JSON data'
            }), 400
        
        data = request.get_json()
        logger.info(f"Received prediction request with data: {data}")
        
        # Validate input data
        required_fields = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
        for field in required_fields:
            if field not in data:
                logger.error(f"Missing required field: {field}")
                return jsonify({
                    'status': 'error',
                    'message': f"Missing required field: {field}"
                }), 400
            try:
                value = float(data[field])
                # Add range validation with more detailed error messages
                if field in ['N', 'P', 'K'] and (value < 0 or value > 140):
                    logger.error(f"Invalid range for {field}: {value}")
                    return jsonify({
                        'status': 'error',
                        'message': f"{field} must be between 0 and 140 kg/ha"
                    }), 400
                elif field == 'temperature' and (value < 8 or value > 44):
                    logger.error(f"Invalid range for temperature: {value}")
                    return jsonify({
                        'status': 'error',
                        'message': "Temperature must be between 8 and 44°C"
                    }), 400
                elif field == 'humidity' and (value < 14 or value > 100):
                    logger.error(f"Invalid range for humidity: {value}")
                    return jsonify({
                        'status': 'error',
                        'message': "Humidity must be between 14 and 100%"
                    }), 400
                elif field == 'ph' and (value < 3.5 or value > 10):
                    logger.error(f"Invalid range for pH: {value}")
                    return jsonify({
                        'status': 'error',
                        'message': "pH must be between 3.5 and 10"
                    }), 400
                elif field == 'rainfall' and (value < 20 or value > 300):
                    logger.error(f"Invalid range for rainfall: {value}")
                    return jsonify({
                        'status': 'error',
                        'message': "Rainfall must be between 20 and 300 mm"
                    }), 400
            except ValueError as e:
                logger.error(f"Invalid value for {field}: {data[field]}, Error: {str(e)}")
                return jsonify({
                    'status': 'error',
                    'message': f"Invalid value for {field}: {data[field]}. Please provide a valid number."
                }), 400
        
        # Create base features
        try:
            features = pd.DataFrame([[
                float(data['N']),
                float(data['P']),
                float(data['K']),
                float(data['temperature']),
                float(data['humidity']),
                float(data['ph']),
                float(data['rainfall'])
            ]], columns=['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall'])
            logger.info(f"Created features DataFrame: {features.to_dict()}")
        except Exception as e:
            logger.error(f"Error creating features DataFrame: {str(e)}")
            logger.exception("Full traceback:")
            return jsonify({
                'status': 'error',
                'message': f"Error processing input data: {str(e)}"
            }), 400
        
        # Add engineered features
        try:
            features = add_features(features)
            logger.info(f"Added engineered features: {features.to_dict()}")
        except Exception as e:
            logger.error(f"Error adding engineered features: {str(e)}")
            logger.exception("Full traceback:")
            return jsonify({
                'status': 'error',
                'message': f"Error processing features: {str(e)}"
            }), 500
        
        # Scale the features
        try:
            features_scaled = scaler.transform(features)
            logger.info(f"Scaled features shape: {features_scaled.shape}")
        except Exception as e:
            logger.error(f"Error scaling features: {str(e)}")
            logger.exception("Full traceback:")
            return jsonify({
                'status': 'error',
                'message': f"Error scaling features: {str(e)}"
            }), 500
        
        # Make prediction
        try:
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
            
            logger.info(f"Prediction successful: {prediction}")
            logger.info(f"Top 3 predictions: {top_3_predictions}")
            return jsonify({
                'status': 'success',
                'predictions': top_3_predictions,
                'best_match': {
                    'crop': prediction,
                    'confidence': float(max_probability)
                }
            })
        except Exception as e:
            logger.error(f"Error making prediction: {str(e)}")
            logger.exception("Full traceback:")
            return jsonify({
                'status': 'error',
                'message': f"Error making prediction: {str(e)}"
            }), 500
    
    except Exception as e:
        logger.error(f"Unexpected error in prediction: {str(e)}")
        logger.error(f"Full traceback: {traceback.format_exc()}")
        return jsonify({
            'status': 'error',
            'message': f"An unexpected error occurred: {str(e)}",
            'details': traceback.format_exc()
        }), 500

@app.route('/model-metrics')
def get_model_metrics():
    try:
        logger.info("Fetching model metrics...")
        with open('model/evaluation_metrics.json', 'r') as f:
            metrics = json.load(f)
        return jsonify(metrics)
    except Exception as e:
        logger.error(f"Error fetching model metrics: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

if __name__ == '__main__':
    # Run the Flask application
    app.run(host='0.0.0.0', port=8080, debug=True) 