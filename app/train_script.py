import pandas as pd
import joblib
import os
import json
import numpy as np
import sys

# Ensure we can import from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score

# PROJECT IMPORTS
from src.config_loader import load_config
from src.model import build_model
from src.train_with_validation import train_with_validation

# PATHS
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")
DATA_PATH = os.path.join(BASE_DIR, "iris.csv")
MODEL_PATH = os.path.join(MODEL_DIR, "iris_sgd_pipeline.pkl")
LE_PATH = os.path.join(MODEL_DIR, "label_encoder.pkl")
LOSS_PATH = os.path.join(MODEL_DIR, "loss.json")

os.makedirs(MODEL_DIR, exist_ok=True)

def main():
    print("Loading config...")
    config = load_config()
    epochs = config["model"]["max_iter"]
    test_size = config["data"]["test_size"]
    random_state = config["data"]["random_state"]

    print("Loading data...")
    df = pd.read_csv(DATA_PATH)
    
    # === FIX: MAP NUMERIC LABELS TO NAMES ===
    # Standard Iris Mapping: 0=Setosa, 1=Versicolor, 2=Virginica
    label_map = {0: "Iris-setosa", 1: "Iris-versicolor", 2: "Iris-virginica"}
    if df.iloc[:, 4].dtype != object:
        print("Mapping numeric labels to Iris class names...")
        df.iloc[:, 4] = df.iloc[:, 4].map(label_map).fillna(df.iloc[:, 4])

    X = df.iloc[:, :4].values
    y_raw = df.iloc[:, 4].values

    # Encode Labels
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    print(f"Classes: {le.classes_}")

    # Split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Preprocessing
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # Build Model
    print("Training model...")
    clf = build_model(config)
    
    # Train
    clf, train_loss, val_loss = train_with_validation(
        clf, X_train_scaled, y_train, X_val_scaled, y_val, epochs=epochs
    )

    # Create Pipeline
    pipeline = Pipeline([
        ("scaler", scaler),
        ("clf", clf)
    ])

    losses = {"train": train_loss, "val": val_loss}

    # Save
    print("Saving artifacts...")
    joblib.dump(pipeline, MODEL_PATH)
    joblib.dump(le, LE_PATH)
    with open(LOSS_PATH, "w") as f:
        json.dump(losses, f)

    # Evaluate
    train_acc = accuracy_score(y_train, clf.predict(X_train_scaled))
    val_acc = accuracy_score(y_val, clf.predict(X_val_scaled))

    print("="*30)
    print("Training Completed Successfully")
    print(f"Train Accuracy: {train_acc:.4f}")
    print(f"Val Accuracy  : {val_acc:.4f}")
    print("="*30)

if __name__ == "__main__":
    main()
