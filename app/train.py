import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss
from sklearn.utils import shuffle as sklearn_shuffle
from model import build_model   

def train_model(X, y, epochs=50, shuffle=True):
    # 1. Initialize logic
    scaler = StandardScaler()
    clf = build_model()
    
    # 2. Fit scaler ONCE
    X_scaled = scaler.fit_transform(X)
    classes = np.unique(y)
    losses = []

    for epoch in range(epochs):
        # 3. Shuffle (if requested)
        if shuffle:
            X_Epoch, y_Epoch = sklearn_shuffle(X_scaled, y, random_state=None)
        else:
            X_Epoch, y_Epoch = X_scaled, y

        # 4. Partial Fit
        clf.partial_fit(X_Epoch, y_Epoch, classes=classes)

        # 5. Calculate Loss (on full training set for monitoring)
        # Note: In a real large dataset, you might want to use a validation set or batch loss.
        # Here we check loss on the specific epoch data or full data? 
        # The original code checked on full X. Let's stick to full X for consistency with original intent,
        # but using the transformed X (X_scaled).
        
        probs = clf.predict_proba(X_scaled)
        loss = log_loss(y, probs)
        losses.append(loss)

        print(f"Epoch {epoch+1}/{epochs} - loss={loss:.4f}")

    # 6. Create Pipeline for export
    # We must reconstruct a pipeline with the fitted scaler and trained classifier
    model = Pipeline([
        ("scaler", scaler),
        ("clf", clf)
    ])

    return model, losses
