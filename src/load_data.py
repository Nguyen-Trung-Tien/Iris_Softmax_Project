import pandas as pd
import os

def load_data(path):
    project_root = os.path.dirname(os.path.dirname(__file__))
    full_path = os.path.join(project_root, path)

    data = pd.read_csv(full_path)
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    return X, y
