from sklearn.linear_model import LogisticRegression

def build_model(config):
    model = LogisticRegression(
        solver="lbfgs",
        max_iter=config["model"]["max_iter"]
    )
    return model
