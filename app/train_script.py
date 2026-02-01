import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from train import train_model

# Load data
df = pd.read_csv("iris.csv")
X = df.iloc[:, :4].values
y = df.iloc[:, 4].values   # giữ label STRING (Setosa, ...)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# Train
model = train_model(X_train, y_train)

# Save
os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/iris_pipeline.pkl")

print("✅ Model trained & saved")
