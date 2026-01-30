import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.load_data import load_data
from src.preprocess import preprocess
from src.model import build_model
from src.train import train_model
from src.evaluate import evaluate
from src.visualize import (
    plot_confusion_matrix,
    plot_pca_3d,
    plot_cross_validation
)
from src.cross_validation import cross_validate
from src.config_loader import load_config


def main():
    #1.Load config
    config = load_config()

    #2.Load data
    X, y = load_data("data/iris.csv")

    print("---Thực hiện Cross-Validation (K=5)---")

    pipeline = make_pipeline(
        StandardScaler(),
        build_model(config)
    )

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=config["data"]["random_state"]
    )

    scores = cross_val_score(
        pipeline,
        X, y,
        cv=cv,
        scoring="accuracy"
    )

    print(f"Kết quả từng fold: {scores}")
    print(f"Accuracy trung bình: {np.mean(scores):.4f} ± {np.std(scores):.4f}")
    print("-------------------------------------------\n")

    #3.Train / test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config["data"]["test_size"],
        random_state=config["data"]["random_state"]
    )

    # 4.Preprocess
    X_train, X_test = preprocess(X_train, X_test)

    # 5.Train model
    model = build_model(config)
    model = train_model(model, X_train, y_train)

    # 6.Evaluate
    y_pred = evaluate(model, X_test, y_test)

    # 7.Visualization
    plot_confusion_matrix(
        y_test, y_pred,
        "results/confusion_matrix.png"
    )

    plot_pca_3d(
    X, y,
    "results/pca_3d.png"
    )

    cv_scores = cross_validate(model, X_train, y_train, cv=5)

    plot_cross_validation(
        cv_scores,
        "results/cross_validation_chart.png"
    )


if __name__ == "__main__":
    main()
