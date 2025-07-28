import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pandas as pd
import numpy as np

def train_model(X, y, model_path="trained_model.pkl"):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.2, random_state=42
    )
    clf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print("\n📊 Classification Report:")
    print(classification_report(y_test, y_pred))
    joblib.dump(clf, model_path)
    print(f"✅ Model saved to {model_path}")

def predict_headings(model_path: str, feature_matrix: np.ndarray, confidence_threshold: float = 0.8):
    model = joblib.load(model_path)
    proba = model.predict_proba(feature_matrix)
    preds = model.predict(feature_matrix)
    results = []
    for i, probs in enumerate(proba):
        max_prob = max(probs)
        if max_prob >= confidence_threshold:
            results.append(preds[i])
        else:
            results.append("None")
    return results
