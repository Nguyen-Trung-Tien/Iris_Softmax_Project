from sklearn.linear_model import LogisticRegression

def build_model():
    return LogisticRegression(
        solver="lbfgs",
        max_iter=1000,
        random_state=42
    )
