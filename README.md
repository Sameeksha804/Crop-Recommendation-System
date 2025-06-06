# Crop Recommendation System

An intelligent machine learning-based system that recommends optimal crops based on soil conditions, climate data, and environmental factors.

## Features

- Soil analysis integration
- Weather data processing
- Yield prediction
- User-friendly interface
- Machine learning-based recommendations

## Setup Instructions

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
python app.py
```

4. Open your browser and navigate to `http://localhost:5000`

## Project Structure

- `app.py`: Main Flask application
- `model/`: Machine learning model and training scripts
- `static/`: Static files (CSS, JavaScript, images)
- `templates/`: HTML templates
- `data/`: Dataset and data processing scripts

## Technologies Used

- Python
- Flask
- Scikit-learn
- Pandas
- HTML/CSS/JavaScript
- Bootstrap 