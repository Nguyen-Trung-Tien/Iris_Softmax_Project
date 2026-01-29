from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np

# 1. Load dữ liệu (Giả sử bạn đã có code load)
df = pd.read_csv('iris.csv') # Thay đường dẫn file của bạn
X = df.iloc[:, :4].values  # 4 cột đặc trưng
y = df.iloc[:, 4].values   # Cột nhãn (loài hoa)

# 2. Tạo một Pipeline (Gồm Chuẩn hóa + Model Softmax)
pipeline = make_pipeline(
    StandardScaler(),
    LogisticRegression(solver='lbfgs', multi_class='multinomial', max_iter=1000)
)

# 3. Cấu hình K-Fold (Thường chọn K=5 hoặc K=10)
# StratifiedKFold đảm bảo tỉ lệ các lớp hoa trong mỗi fold là đều nhau
k_fold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# 4. Thực hiện Cross-Validation
print("Đang chạy Cross-Validation (k=5)...")
scores = cross_val_score(pipeline, X, y, cv=k_fold, scoring='accuracy')

# 5. Xuất kết quả
print(f"Kết quả từng lần chạy (Fold): {scores}")
print(f"Độ chính xác trung bình: {scores.mean():.4f} (+/- {scores.std():.4f})")