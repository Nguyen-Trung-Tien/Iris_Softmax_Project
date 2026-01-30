from sklearn.linear_model import SGDClassifier

def build_model(config):
    return SGDClassifier(
        loss="log_loss",                
        learning_rate="constant",
        eta0=config["model"]["learning_rate"],
        random_state=42
    )
