import os
import joblib
import numpy as np
from sklearn.model_selection import train_test_split

from src.load_data import load_data
from src.preprocess import preprocess
from src.model import build_model
from src.train import train_model
from src.config_loader import load_config


def main():
    config = load_config()

    X, y = load_data("data/iris.csv")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config["data"]["test_size"],
        random_state=config["data"]["random_state"]
    )

    X_train, X_test = preprocess(X_train, X_test)

    model = build_model(config)
    model = train_model(model, X_train, y_train)

    # Tạo thư mục model nếu chưa tồn tại
    os.makedirs("model", exist_ok=True)

    joblib.dump(model, "model/iris_softmax.pkl")
    print("Đã lưu model: model/iris_softmax.pkl")


if __name__ == "__main__":
    main()
