import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np

def plot_confusion_matrix(y_test, y_pred, save_path):
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.savefig(save_path)
    plt.show()

def plot_decision_boundary(model, X, y, save_path):
    X2 = X[:, 2:4]  # petal length & petal width
    model.fit(X2, y)

    x_min, x_max = X2[:, 0].min() - 1, X2[:, 0].max() + 1
    y_min, y_max = X2[:, 1].min() - 1, X2[:, 1].max() + 1

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 200),
        np.linspace(y_min, y_max, 200)
    )

    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(6, 5))
    plt.contourf(xx, yy, Z, alpha=0.3)
    plt.scatter(X2[:, 0], X2[:, 1], c=y)
    plt.xlabel("Petal length (cm)")
    plt.ylabel("Petal width (cm)")
    plt.title("Decision Boundary - Softmax Regression")
    plt.savefig(save_path)
    plt.show()

def plot_cross_validation(scores, save_path):
    folds = np.arange(1, len(scores) + 1)

    plt.figure(figsize=(6, 5))
    sns.lineplot(x=folds, y=scores, marker="o")
    plt.axhline(
        y=np.mean(scores),
        linestyle="--",
        color="red",
        label=f"Mean accuracy = {np.mean(scores):.2f}"
    )

    plt.xlabel("Fold")
    plt.ylabel("Accuracy")
    plt.title("Cross-Validation Accuracy")
    plt.legend()
    plt.grid(True)

    plt.savefig(save_path)
    plt.show()
