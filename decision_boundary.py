import numpy as np
import matplotlib.pyplot as plt
import os

from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# ===================== 1. LOAD DATA =====================
iris = datasets.load_iris()

# Chỉ lấy 2 feature để vẽ decision boundary
# Petal length & Petal width
X = iris.data[:, 2:]
y = iris.target

# ===================== 2. DEFINE MODELS =====================
models = [
    make_pipeline(
        StandardScaler(),
        LogisticRegression(
            solver="lbfgs",
            C=1e5,
            max_iter=1000
        )
    ),
    make_pipeline(
        StandardScaler(),
        SVC(
            kernel="rbf",
            gamma=0.7
        )
    ),
    make_pipeline(
        StandardScaler(),
        KNeighborsClassifier(
            n_neighbors=3
        )
    )
]

titles = [
    "Softmax Regression (Linear Decision Boundary)",
    "SVM with RBF Kernel (Non-linear)",
    "KNN (Non-linear)"
]

# ===================== 3. PLOT =====================
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for i, model in enumerate(models):
    # Train model
    model.fit(X, y)

    # Decision boundary
    DecisionBoundaryDisplay.from_estimator(
        model,
        X,
        cmap=plt.cm.coolwarm,
        alpha=0.6,
        ax=axes[i],
        response_method="predict"
    )

    # Scatter points
    axes[i].scatter(
        X[:, 0],
        X[:, 1],
        c=y,
        cmap=plt.cm.coolwarm,
        edgecolors="k"
    )

    axes[i].set_title(titles[i])
    axes[i].set_xlabel("Petal length")
    axes[i].set_ylabel("Petal width")

plt.tight_layout()

# ===================== 4. SAVE FIGURE =====================
output_dir = "results"
os.makedirs(output_dir, exist_ok=True)

file_path = os.path.join(
    output_dir,
    "comparison_decision_boundaries.png"
)

plt.savefig(file_path)
print(f"Đã lưu hình ảnh tại: {file_path}")

plt.show()
