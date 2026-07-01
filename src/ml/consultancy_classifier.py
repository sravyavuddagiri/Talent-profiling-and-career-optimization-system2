# src/ml/consultancy_classifier.py
import os
import joblib
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.utils.class_weight import compute_class_weight

from src.config import CONSULTANCY_MODEL_PATH, CONSULTANCIES_CSV_PATH

# Try to load trained model globally
model = None

def train_and_save_model():
    """Trains the consultancy classification model and saves it."""
    global model
    try:
        if not os.path.exists(CONSULTANCIES_CSV_PATH):
            print(f"Error: {CONSULTANCIES_CSV_PATH} not found. Cannot train model.")
            return False
            
        print("Training consultancy model automatically...")
        df = pd.read_csv(CONSULTANCIES_CSV_PATH)
        df.fillna("", inplace=True)
        df["text"] = df["name"] + " " + df["description"]
        
        X = df["text"]
        y = df["label"]
        
        classes = np.unique(y)
        class_weights = compute_class_weight(
            class_weight="balanced",
            classes=classes,
            y=y
        )
        class_weight_dict = dict(zip(classes, class_weights))
        
        lr_model = LogisticRegression(
            max_iter=2000,
            class_weight=class_weight_dict,
            random_state=42
        )
        
        pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(
                ngram_range=(1, 2),
                min_df=1,
                max_df=0.9,
                sublinear_tf=True
            )),
            ("classifier", lr_model)
        ])
        
        pipeline.fit(X, y)
        
        os.makedirs(os.path.dirname(CONSULTANCY_MODEL_PATH), exist_ok=True)
        joblib.dump(pipeline, CONSULTANCY_MODEL_PATH)
        model = pipeline
        print(f"Model trained and saved to {CONSULTANCY_MODEL_PATH}")
        return True
    except Exception as e:
        print(f"Failed to auto-train model: {e}")
        return False

# Initial load attempt
if os.path.exists(CONSULTANCY_MODEL_PATH):
    try:
        model = joblib.load(CONSULTANCY_MODEL_PATH)
    except Exception as e:
        print(f"Error loading ML model from {CONSULTANCY_MODEL_PATH}: {e}")
        # Try to retrain if load failed (e.g., due to scikit-learn version mismatch)
        train_and_save_model()
else:
    # Auto-train if it doesn't exist
    train_and_save_model()

def get_ml_model():
    global model
    if model is None:
        # Retry training/loading if somehow not initialized
        if os.path.exists(CONSULTANCY_MODEL_PATH):
            try:
                model = joblib.load(CONSULTANCY_MODEL_PATH)
            except:
                train_and_save_model()
        else:
            train_and_save_model()
    return model

def predict_consultancy_ml(name: str, description: str, asks_fee: bool):
    """
    Run the Logistic Regression classifier purely on the text and return
    prediction details including confidence and per-class probabilities.
    """
    global model
    current_model = get_ml_model()
    if current_model is None:
        return {
            "error": (
                "Consultancy ML model file not found or could not be loaded, and auto-training failed. "
                "Train it manually and save as 'models/fake_consultancy_research.pkl' to enable ML analysis."
            )
        }
        
    text = name + " " + description

    # The model/pipeline expects an iterable of strings
    pred = current_model.predict([text])[0]
    proba = current_model.predict_proba([text])[0]

    fake_prob = float(proba[1])
    real_prob = float(proba[0])
    is_fake = int(pred) == 1

    label = "Fake Consultancy" if is_fake else "Real Consultancy"
    confidence = fake_prob if is_fake else real_prob

    return {
        "label": label,
        "confidence": round(confidence * 100.0, 2),
        "fake_probability": round(fake_prob * 100.0, 2),
        "real_probability": round(real_prob * 100.0, 2),
        "source": "ML Model",
    }
