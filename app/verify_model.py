import joblib
import pandas as pd
import numpy as np
import os

def verify():
    print("Loading model and encoder...")
    try:
        model = joblib.load("model/iris_sgd_pipeline.pkl")
        le = joblib.load("model/label_encoder.pkl")
        print("Model loaded.")
        print(f"Classes: {le.classes_}")
    except Exception as e:
        print(f"Failed to load: {e}")
        return

    print("Loading data...")
    df = pd.read_csv("iris.csv")
    X = df.iloc[:, :4].values
    
    # Predict
    print("Predicting...")
    preds_idx = model.predict(X)
    preds_label = le.inverse_transform(preds_idx)
    
    unique_preds = np.unique(preds_label)
    print(f"Unique Predictions: {unique_preds}")
    
    if len(unique_preds) < 3:
        print("WARNING: Model is predicting fewer than 3 classes!")
    else:
        print("OK: Model predicts multiple classes.")

    # Acc check logic roughly
    label_map = {0: "Iris-setosa", 1: "Iris-versicolor", 2: "Iris-virginica"}
    y_raw = df.iloc[:, 4]
    if y_raw.dtype != object:
         y_raw = y_raw.map(label_map).fillna(y_raw)
    
    acc = np.mean(preds_label == y_raw)
    print(f"Accuracy on full dataset: {acc:.4f}")

if __name__ == "__main__":
    verify()
