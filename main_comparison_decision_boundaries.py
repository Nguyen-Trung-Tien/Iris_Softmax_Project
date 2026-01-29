import numpy as np
import matplotlib.pyplot as plt
import os  # <--- Thêm thư viện này để xử lý thư mục
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.inspection import DecisionBoundaryDisplay

# 1. Chuẩn bị dữ liệu
iris = datasets.load_iris()
X = iris.data[:, 2:]
y = iris.target

# 2. Khởi tạo 3 mô hình
models = [
    LogisticRegression(solver='lbfgs', C=1e5),
    SVC(kernel='rbf', gamma=0.7),
    KNeighborsClassifier(n_neighbors=3)
]
titles = ['Softmax Regression (Tuyen tinh)', 'SVM - RBF Kernel (Phi tuyen)', 'KNN (Phi tuyen)']

# 3. Vẽ hình so sánh
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for i, model in enumerate(models):
    model.fit(X, y)
    
    DecisionBoundaryDisplay.from_estimator(
        model, X, cmap=plt.cm.coolwarm, alpha=0.6, ax=axes[i], response_method="predict"
    )
    
    axes[i].scatter(X[:, 0], X[:, 1], c=y, edgecolors="k", cmap=plt.cm.coolwarm)
    axes[i].set_title(titles[i])
    axes[i].set_xlabel('Petal length')
    axes[i].set_ylabel('Petal width')

plt.tight_layout()

# 4. Lưu hình vào folder 'results'
output_folder = 'results'

# Kiểm tra nếu folder chưa tồn tại thì tạo mới
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Tạo đường dẫn file: results/comparison_decision_boundaries.png
file_path = os.path.join(output_folder, 'comparison_decision_boundaries.png')

# Lưu file
plt.savefig(file_path)
print(f"Đã lưu hình ảnh thành công tại: {file_path}")

plt.show()