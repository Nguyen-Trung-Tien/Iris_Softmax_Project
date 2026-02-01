from flask import Flask, request, render_template_string, jsonify
import joblib
import numpy as np
import pandas as pd
import os, json, threading, time

# ===== PLOT =====
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from train import build_model

app = Flask(__name__)

# PATHS
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")
STATIC_DIR = os.path.join(BASE_DIR, "static")

MODEL_PATH = os.path.join(MODEL_DIR, "iris_sgd_pipeline.pkl")
LOSS_PATH  = os.path.join(MODEL_DIR, "loss.json")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)

# TRAINING STATUS (GLOBAL)
training_status = {
    "running": False,
    "epoch": 0,
    "total": 0
}

# LOAD MODEL (IF EXISTS)
model = None
train_losses = []

if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)

if os.path.exists(LOSS_PATH):
    with open(LOSS_PATH) as f:
        train_losses = json.load(f)

# BACKGROUND TRAINING
def background_train(epochs=50):
    global model, train_losses, training_status

    df = pd.read_csv("iris.csv")
    X = df.iloc[:, :4].values
    y = df.iloc[:, 4].values
    classes = np.unique(y)

    model = build_model()
    train_losses = []

    training_status["running"] = True
    training_status["epoch"] = 0
    training_status["total"] = epochs

    for epoch in range(epochs):
        model.partial_fit(X, y, classes=classes)

        probs = model.predict_proba(X)
        loss = -np.mean(np.log(np.max(probs, axis=1)))
        train_losses.append(float(loss))

        training_status["epoch"] = epoch + 1
        time.sleep(0.1)  # để thấy loading

    joblib.dump(model, MODEL_PATH)
    with open(LOSS_PATH, "w") as f:
        json.dump(train_losses, f)

    training_status["running"] = False

# HTML
HTML_FORM = """
<!DOCTYPE html>
<html>
<head>
    <title>Iris SGD Dashboard</title>
</head>
<body>

<h2>🌸 Iris SGD ML Dashboard</h2>

<h3>🔥 Train Model</h3>
<button onclick="startTrain()">Train SGD</button>
<p id="train_status"></p>

<hr>

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

<h3>📂 Predict CSV</h3>
<form method="post" action="/predict_csv" enctype="multipart/form-data">
    <input type="file" name="file" accept=".csv" required>
    <br><br>
    <button type="submit">Upload & Analyze</button>
</form>

{% if error %}
<p style="color:red;">{{ error }}</p>
{% endif %}

<script>
function startTrain() {
    fetch("/train", { method: "POST" });

    const interval = setInterval(() => {
        fetch("/train_status")
            .then(r => r.json())
            .then(s => {
                if (s.running) {
                    document.getElementById("train_status").innerHTML =
                        `⏳ Training epoch ${s.epoch} / ${s.total}`;
                } else {
                    document.getElementById("train_status").innerHTML =
                        "✅ Training completed!";
                    clearInterval(interval);
                }
            });
    }, 500);
}
</script>

</body>
</html>
"""

# ROUTES
@app.route("/")
def home():
    return render_template_string(HTML_FORM)

@app.route("/train", methods=["POST"])
def train():
    if training_status["running"]:
        return jsonify({"status": "already running"})

    t = threading.Thread(target=background_train)
    t.start()
    return jsonify({"status": "started"})

@app.route("/train_status")
def train_status():
    return jsonify(training_status)

@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return render_template_string(HTML_FORM, error="Model not trained yet")

    X = np.array([[ 
        float(request.form["sepal_length"]),
        float(request.form["sepal_width"]),
        float(request.form["petal_length"]),
        float(request.form["petal_width"])
    ]])

    pred = model.predict(X)[0]
    prob = model.predict_proba(X)[0]

    return render_template_string(
        HTML_FORM,
        result={
            "label": pred,
            "confidence": round(float(np.max(prob)), 4)
        }
    )

@app.route("/predict_csv", methods=["POST"])
def predict_csv():
    if model is None:
        return render_template_string(HTML_FORM, error="Model not trained yet")

    df = pd.read_csv(request.files["file"])
    X = df.iloc[:, :4].values

    preds = model.predict(X)
    probs = model.predict_proba(X)
    confidence = np.max(probs, axis=1)

    df["predicted_class"] = preds
    df["confidence"] = confidence

    # ===== Sample-wise loss =====
    class_idx = {c: i for i, c in enumerate(model.classes_)}
    sample_loss = [
        -np.log(probs[i][class_idx[preds[i]]])
        for i in range(len(preds))
    ]

    # ===== Plot training loss =====
    plt.figure()
    plt.plot(train_losses, marker="o")
    plt.title("Training Loss (SGD)")
    plt.xlabel("Epoch")
    plt.ylabel("Log Loss")
    plt.tight_layout()
    plt.savefig(f"{STATIC_DIR}/loss_curve.png")
    plt.close()

    # ===== Sample loss =====
    plt.figure()
    plt.plot(sample_loss)
    plt.title("Sample-wise Loss")
    plt.xlabel("Sample")
    plt.ylabel("Loss")
    plt.tight_layout()
    plt.savefig(f"{STATIC_DIR}/sample_loss.png")
    plt.close()

    # ===== 3D Scatter =====
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(X[:,0], X[:,1], X[:,2], c=confidence, cmap="viridis")
    ax.set_xlabel("Sepal Length")
    ax.set_ylabel("Sepal Width")
    ax.set_zlabel("Petal Length")
    plt.tight_layout()
    plt.savefig(f"{STATIC_DIR}/scatter_3d.png")
    plt.close()

    return f"""
    <h2>📊 Analysis Result</h2>
    {df.head(10).to_html(index=False)}

    <h3>Training Loss</h3>
    <img src="/static/loss_curve.png">

    <h3>Sample Loss</h3>
    <img src="/static/sample_loss.png">

    <h3>3D Feature Space</h3>
    <img src="/static/scatter_3d.png">

    <br><a href="/">⬅ Back</a>
    """

# =================================================
if __name__ == "__main__":
    app.run(debug=True)
