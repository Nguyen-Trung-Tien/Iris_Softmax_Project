import numpy as np
from sklearn.model_selection import cross_val_score

def cross_validate(model, X, y, cv=6):
   
    scores = cross_val_score(
        model,
        X,
        y,
        cv=cv,
        scoring="accuracy"
    )
    return scores
