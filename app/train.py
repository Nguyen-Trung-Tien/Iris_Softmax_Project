import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss
from model import build_model   

def train_model(X, y, epochs=50, shuffle=True):
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", build_model())
    ])

    classes = np.unique(y)
    losses = []

    for epoch in range(epochs):

        if shuffle:
            idx = np.random.permutation(len(X))
            X_epoch = X[idx]
            y_epoch = y[idx]
        else:
            X_epoch = X
            y_epoch = y

        # SGD update
        model.named_steps["clf"].partial_fit(
            model.named_steps["scaler"].fit_transform(X_epoch),
            y_epoch,
            classes=classes
        )

        #LOSS thật (cross-entropy)
        probs = model.predict_proba(X)
        loss = log_loss(y, probs)
        losses.append(loss)

        print(f"Epoch {epoch+1}/{epochs} - loss={loss:.4f}")

    return model, losses
