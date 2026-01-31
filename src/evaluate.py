from sklearn.metrics import accuracy_score, classification_report

def evaluate(model, X_test, y_test, verbose=True):
    y_pred = model.predict(X_test)

    if verbose:
        print(f"Test Accuracy: {accuracy_score(y_test, y_pred):.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))

    return y_pred
