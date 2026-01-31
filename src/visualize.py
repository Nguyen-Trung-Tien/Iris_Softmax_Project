import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# ================= CONFUSION MATRIX =================
def plot_confusion_matrix(y_test, y_pred, save_path):
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(6, 5))
    plt.imshow(cm, cmap="Blues")
    plt.colorbar()

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j], ha="center", va="center")

    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()



# ================= PCA 3D VISUALIZATION =================
def plot_pca_3d(X, y, save_path):
    """
    PCA visualization only (NOT used for training or evaluation)
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

def plot_pca_2d_with_boundary(X, y, model, save_path):
 

    # 1. Scale + PCA
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    # 2. Train model trên PCA space
    model.fit(X_pca, y)

    # 3. Mesh grid
    x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
    y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 300),
        np.linspace(y_min, y_max, 300)
    )

    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # 4. Plot
    plt.figure(figsize=(7, 6))
    plt.contourf(xx, yy, Z, alpha=0.25, cmap="viridis")

    scatter = plt.scatter(
        X_pca[:, 0],
        X_pca[:, 1],
        c=y,
        cmap="viridis",
        edgecolor="k",
        s=40
    )

    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.title("PCA 2D with Decision Boundary")

    legend = plt.legend(
        *scatter.legend_elements(),
        title="Classes"
    )
    plt.gca().add_artist(legend)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()


# ================= CROSS VALIDATION =================
def plot_cross_validation(scores, save_path):
    folds = np.arange(1, len(scores) + 1)

    plt.figure(figsize=(6, 5))
    plt.plot(folds, scores, marker="o", label="Fold accuracy")
    plt.axhline(
        y=np.mean(scores),
        linestyle="--",
        color="red",
        label=f"Mean = {np.mean(scores):.2f}"
    )

    plt.xlabel("Fold")
    plt.ylabel("Accuracy")
    plt.title("Cross-Validation Accuracy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()


# ================= TRAIN VS VALIDATION LOSS =================
def plot_train_val_loss(train_losses, val_losses, save_path):
    epochs = np.arange(1, len(train_losses) + 1)

    plt.figure(figsize=(6, 5))
    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(epochs, val_losses, label="Validation Loss")

    plt.xlabel("Epoch")
    plt.ylabel("Log Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()

