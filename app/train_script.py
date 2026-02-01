import pandas as pd
import joblib
import os
import json
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from train import train_model

# LOAD DATA
df = pd.read_csv("iris.csv")

X = df.iloc[:, :4].values
y = df.iloc[:, 4].astype(str).values   

# SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# TRAIN
model, losses = train_model(
    X_train,
    y_train,
    epochs=50,
    shuffle=True         
)

# EVALUATE
train_acc = accuracy_score(y_train, model.predict(X_train))
test_acc  = accuracy_score(y_test,  model.predict(X_test))

# SAVE
os.makedirs("model", exist_ok=True)

joblib.dump(model, "model/iris_sgd_pipeline.pkl")

with open("model/loss.json", "w") as f:
    json.dump(losses, f)

with open("model/meta.json", "w") as f:
    json.dump({
        "epochs": len(losses),
        "train_accuracy": train_acc,
        "test_accuracy": test_acc,
        "model": "SGDClassifier (log_loss)"
    }, f, indent=4)

# =========================
print("Model + loss curve saved")
print(f"Train accuracy: {train_acc:.4f}")
print(f"Test accuracy : {test_acc:.4f}")
