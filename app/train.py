import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss
from model import build_model

def train_model(X_train, y_train, epochs=50, shuffle=True):
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", build_model())
    ])

    losses = []

    for epoch in range(epochs):
        if shuffle:
            idx = np.random.permutation(len(X_train))
            X_epoch = X_train[idx]
            y_epoch = y_train[idx]
        else:
            X_epoch = X_train
            y_epoch = y_train

        pipeline.fit(X_epoch, y_epoch)

        probs = pipeline.predict_proba(X_epoch)
        loss = log_loss(y_epoch, probs)
        losses.append(loss)

        print(f"Epoch {epoch+1}/{epochs} - loss: {loss:.4f}")

    return pipeline, losses
