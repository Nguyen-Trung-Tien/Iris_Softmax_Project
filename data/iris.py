from sklearn.datasets import load_iris
import pandas as pd

iris = load_iris()

df = pd.DataFrame(
    iris.data,
    columns=iris.feature_names
)

df["label"] = iris.target

df.to_csv("iris.csv", index=False)

print("Đã tạo file iris.csv")
