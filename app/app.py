from fastapi import FastAPI
import joblib
import numpy as np
import os

app = FastAPI(
    title="Iris Softmax Regression API",
    description="Dự đoán loài hoa Iris bằng Softmax Regression",
    version="1.0"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "iris_softmax.pkl")

model = joblib.load(MODEL_PATH)

iris_classes = {
    0: "Setosa",
    1: "Versicolor",
    2: "Virginica"
}

@app.get("/")
def home():
    return {"message": "Iris Softmax Regression API is running 🚀"}

@app.post("/predict")
def predict(
    sepal_length: float,
    sepal_width: float,
    petal_length: float,
    petal_width: float
):
    X = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

    prediction = model.predict(X)[0]
    probs = model.predict_proba(X)[0]

    return {
        "predicted_class": iris_classes[int(prediction)],
        "confidence": float(np.max(probs)),
        "probabilities": {
            iris_classes[i]: float(probs[i]) for i in range(3)
        }
    }
