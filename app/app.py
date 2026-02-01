from flask import Flask, request, render_template_string
import joblib
import numpy as np
import pandas as pd
import os

# ===== PLOT =====
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ===== METRIC =====
from sklearn.metrics import log_loss

app = Flask(__name__)

# LOAD MODEL
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "iris_pipeline.pkl")
STATIC_DIR = os.path.join(BASE_DIR, "static")

os.makedirs(STATIC_DIR, exist_ok=True)

model = joblib.load(MODEL_PATH)

# HTML
HTML_FORM = """
<!DOCTYPE html>
<html>
<head>
    <title>Iris ML Dashboard</title>
</head>
<body>
    <h2>🌸 Iris Softmax Dashboard</h2>

    <h3>🔢 Predict one sample</h3>
    <form method="post" action="/predict">
        Sepal length: <input type="number" step="0.1" name="sepal_length" required><br>
        Sepal width: <input type="number" step="0.1" name="sepal_width" required><br>
        Petal length: <input type="number" step="0.1" name="petal_length" required><br>
        Petal width: <input type="number" step="0.1" name="petal_width" required><br><br>
        <button type="submit">Predict</button>
    </form>

    {% if result %}
        <p><b>Class:</b> {{ result.label }}</p>
        <p><b>Confidence:</b> {{ result.confidence }}</p>
    {% endif %}

    <hr>

    <h3>📂 Predict from CSV</h3>
    <form method="post" action="/predict_csv" enctype="multipart/form-data">
        <input type="file" name="file" accept=".csv" required>
        <br><br>
        <button type="submit">Upload & Analyze</button>
    </form>

    {% if error %}
        <p style="color:red;">{{ error }}</p>
    {% endif %}
</body>
</html>
"""

# ROUTES
@app.route("/")
def home():
    return render_template_string(HTML_FORM)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        X = np.array([[ 
            float(request.form["sepal_length"]),
            float(request.form["sepal_width"]),
            float(request.form["petal_length"]),
            float(request.form["petal_width"])
        ]])

        pred = model.predict(X)[0]
        prob = model.predict_proba(X)[0]

        result = {
            "label": pred,
            "confidence": round(float(np.max(prob)), 4)
        }

        return render_template_string(HTML_FORM, result=result)

    except Exception:
        return render_template_string(HTML_FORM, error="Invalid input")

@app.route("/predict_csv", methods=["POST"])
def predict_csv():
    try:
        file = request.files["file"]
        df = pd.read_csv(file)

        X = df.iloc[:, :4].values

        # ===== Predict =====
        preds = model.predict(X)
        probs = model.predict_proba(X)
        confidence = np.max(probs, axis=1)

        df["predicted_class"] = preds
        df["confidence"] = confidence

        #LOSS (LOG LOSS – Cross Entropy)
        loss = log_loss(preds, probs)

        plt.figure()
        plt.bar(["Log Loss"], [loss])
        plt.title("Cross-Entropy Loss")
        plt.ylabel("Loss")
        plt.tight_layout()
        plt.savefig(f"{STATIC_DIR}/logloss_bar.png")
        plt.close()

        # CLASS DISTRIBUTION
        plt.figure()
        df["predicted_class"].value_counts().plot(kind="bar", color="skyblue")
        plt.title("Predicted Class Distribution")
        plt.xlabel("Class")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(f"{STATIC_DIR}/class_distribution.png")
        plt.close()

        # CONFIDENCE HIST
        plt.figure()
        plt.hist(confidence, bins=10, color="orange", edgecolor="black")
        plt.title("Confidence Distribution")
        plt.xlabel("Confidence")
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.savefig(f"{STATIC_DIR}/confidence_hist.png")
        plt.close()

        # 3D SCATTER (Feature space)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        ax.scatter(
            X[:, 0], X[:, 1], X[:, 2],
            c=confidence,
            cmap="viridis",
            s=40
        )

        ax.set_xlabel("Sepal Length")
        ax.set_ylabel("Sepal Width")
        ax.set_zlabel("Petal Length")
        ax.set_title("3D Feature Space (Colored by Confidence)")

        plt.tight_layout()
        plt.savefig(f"{STATIC_DIR}/scatter_3d.png")
        plt.close()

        # RESULT PAGE
        return f"""
        <h2>📊 Prediction Analysis</h2>

        <p><b>Log Loss:</b> {loss:.4f}</p>

        {df.head(10).to_html(index=False)}

        <h3>Loss</h3>
        <img src="/static/logloss_bar.png">

        <h3>Class Distribution</h3>
        <img src="/static/class_distribution.png">

        <h3>Confidence Histogram</h3>
        <img src="/static/confidence_hist.png">

        <h3>3D Feature Visualization</h3>
        <img src="/static/scatter_3d.png">

        <br><br>
        <a href="/">⬅ Back</a>
        """

    except Exception as e:
        return render_template_string(
            HTML_FORM,
            error=f"CSV error: {e}"
        )

# =================================================
if __name__ == "__main__":
    app.run(debug=True)
