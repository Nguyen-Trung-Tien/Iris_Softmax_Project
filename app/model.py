from sklearn.linear_model import SGDClassifier

def build_model():
    return SGDClassifier(
        loss="log_loss",       
        learning_rate="constant",
        eta0=0.01,             
        max_iter=1,             
        warm_start=True,       
        tol=None,               
        random_state=42
    )
