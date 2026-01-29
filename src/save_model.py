import joblib
import os

def train_model(model, X_train, y_train):
    model.fit(X_train, y_train)

    # Lưu model
    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(model, f"{model_dir}/softmax_iris.pkl")

    return model
