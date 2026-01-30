import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D


# ================= CONFUSION MATRIX =================
def plot_confusion_matrix(y_test, y_pred, save_path):
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()


# ================= PCA 3D VISUALIZATION =================
def plot_pca_3d(X, y, save_path):
    """
    Dùng toàn bộ 4 feature -> PCA -> 3D
    """

    # 1. Chuẩn hóa
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 2. PCA 4 -> 3
    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(X_scaled)

    # 3. Vẽ 3D
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")

    scatter = ax.scatter(
        X_pca[:, 0],
        X_pca[:, 1],
        X_pca[:, 2],
        c=y,
        cmap="viridis",
        s=40
    )

    ax.set_xlabel("Principal Component 1")
    ax.set_ylabel("Principal Component 2")
    ax.set_zlabel("Principal Component 3")
    ax.set_title("3D PCA Visualization (from 4 features)")

    legend = ax.legend(
        *scatter.legend_elements(),
        title="Classes"
    )
    ax.add_artist(legend)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()


# ================= CROSS VALIDATION =================
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
    plt.tight_layout()

    plt.savefig(save_path)
    plt.show()
