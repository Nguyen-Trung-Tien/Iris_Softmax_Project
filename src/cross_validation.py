from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

def cross_validate(model, X, y, cv=6):
    pipeline = make_pipeline(
        StandardScaler(),
        model
    )

    scores = cross_val_score(
        pipeline,
        X,
        y,
        cv=cv,
        scoring="accuracy"
    )
    return scores
