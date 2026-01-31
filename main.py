import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.load_data import load_data

from src.model import build_model
from src.train import train_model
from src.evaluate import evaluate
from src.visualize import (
    plot_confusion_matrix,
    plot_pca_3d,
    plot_pca_2d_with_boundary,
    plot_cross_validation,
    plot_train_val_loss
)
from sklearn.linear_model import LogisticRegression
from src.cross_validation import cross_validate
from src.config_loader import load_config
from src.train_with_validation import train_with_validation


def main():
    # 1. Load config
    config = load_config()

    # 2. Load data
    X, y = load_data("data/iris.csv")

    # 3. Cross-Validation (Scaler + Model)
    print("--- Thực hiện Cross-Validation (K=5) ---")

    pipeline = make_pipeline(
        StandardScaler(),
        build_model(config)
    )

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=config["data"]["random_state"]
    )

    cv_scores = cross_val_score(
        pipeline,
        X, y,
        cv=cv,
        scoring="accuracy"
    )

    print(f"Kết quả từng fold: {cv_scores}")
    print(f"Accuracy trung bình: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")
    print("----------------------------------------\n")

    # 4. Train / Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config["data"]["test_size"],
        random_state=config["data"]["random_state"],
        stratify=y
    )

    # 5. Train / Validation split
    X_train, X_val, y_train, y_val = train_test_split(
        X_train,
        y_train,
        test_size=0.2,
        random_state=config["data"]["random_state"],
        stratify=y_train
    )

    # 6. Feature scaling (fit on train only)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    # 7. Train model with validation loss
    model = build_model(config)
    model, train_losses, val_losses = train_with_validation(
        model,
        X_train, y_train,
        X_val, y_val,
        epochs=200
    )

    # 8. Evaluate on test set
    y_pred = evaluate(model, X_test, y_test)

    # 9. Visualization
    plot_confusion_matrix(
        y_test, y_pred,
        "results/confusion_matrix.png"
    )

    # PCA visualization only (not used for training or evaluation)
    plot_pca_2d_with_boundary(
        X, y,
        LogisticRegression(max_iter=1000),
        "results/pca_2d_boundary.png"
    )

    plot_pca_3d(
        X, y,
        "results/pca_3d.png"
    )

    plot_train_val_loss(
        train_losses,
        val_losses,
        "results/train_val_loss.png"
    )

    plot_cross_validation(
        cv_scores,
        "results/cross_validation_chart.png"
    )



if __name__ == "__main__":
    main()
