from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from model import build_model

def train_model(X_train, y_train):
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", build_model())
    ])

    pipeline.fit(X_train, y_train)
    return pipeline
