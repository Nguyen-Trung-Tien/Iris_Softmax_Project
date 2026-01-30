import numpy as np
from sklearn.metrics import log_loss

def train_with_loss(model, X_train, y_train, X_val, y_val, epochs=200):
    train_losses = []
    val_losses = []
    classes = np.unique(y_train)

    for epoch in range(epochs):
        if epoch == 0:
            model.partial_fit(X_train, y_train, classes=classes)
        else:
            model.partial_fit(X_train, y_train)

        # Train loss
        y_train_prob = model.predict_proba(X_train)
        train_loss = log_loss(y_train, y_train_prob)
        train_losses.append(train_loss)

        # Validation loss
        y_val_prob = model.predict_proba(X_val)
        val_loss = log_loss(y_val, y_val_prob)
        val_losses.append(val_loss)

    return model, train_losses, val_losses
