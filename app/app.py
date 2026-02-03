import sys
import os
import json
import threading
import time

# Add parent directory to path to import from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from flask import Flask, request, render_template, jsonify, url_for
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder

# PROJECT IMPORTS
from src.config_loader import load_config
from src.model import build_model
from src.train_with_validation import train_with_validation

app = Flask(__name__)

# CONFIG & PATHS
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")
STATIC_DIR = os.path.join(BASE_DIR, "static")
DATA_PATH = os.path.join(BASE_DIR, "iris.csv") 

MODEL_PATH = os.path.join(MODEL_DIR, "iris_sgd_pipeline.pkl")
LE_PATH = os.path.join(MODEL_DIR, "label_encoder.pkl")
LOSS_PATH  = os.path.join(MODEL_DIR, "loss.json")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)

# GLOBAL STATE
training_status = {
    "running": False,
    "epoch": 0,
    "total": 0,
    "error": None
}

model_pipeline = None
label_encoder = None
losses_history = {"train": [], "val": []}
prediction_history = []  # Feature 3: Prediction History
current_model_meta = {} # store metadata like accuracy

# LOAD EXISTING MODEL
def load_artifacts():
    global model_pipeline, label_encoder, losses_history, current_model_meta
    if os.path.exists(MODEL_PATH):
        try:
            model_pipeline = joblib.load(MODEL_PATH)
        except Exception as e:
            print(f"Error loading model: {e}")
    
    if os.path.exists(LE_PATH):
        try:
            label_encoder = joblib.load(LE_PATH)
        except Exception as e:
            print(f"Error loading label encoder: {e}")

    if os.path.exists(LOSS_PATH):
        try:
            with open(LOSS_PATH, "r") as f:
                losses_history = json.load(f)
                # Try to infer metadata
                if isinstance(losses_history, dict) and "val" in losses_history and len(losses_history["val"]) > 0:
                     # loss is log_loss, not accuracy. Just placeholder.
                     current_model_meta["accuracy"] = 0.0 # Unknown
        except Exception:
            losses_history = {"train": [], "val": []}

load_artifacts()

def run_training_task():
    global model_pipeline, label_encoder, losses_history, training_status, current_model_meta

    training_status["running"] = True
    training_status["error"] = None
    
    try:
        # 1. Load Config
        config = load_config()
        epochs = config["model"]["max_iter"]
        test_size = config["data"]["test_size"]
        random_state = config["data"]["random_state"]

        training_status["total"] = epochs
        training_status["epoch"] = 0 

        # 2. Load Data
        if os.path.exists(DATA_PATH):
            df = pd.read_csv(DATA_PATH)
        else:
             # Fallback
            df = pd.read_csv(os.path.join(BASE_DIR, "../src/iris.csv"))
        
        # Check simple validation
        if df.shape[1] < 5:
            raise ValueError("Dataset must have at least 5 columns (4 features + 1 label)")

        X = df.iloc[:, :4].values
        y_raw = df.iloc[:, 4].values
        
        # === MAP NUMERIC LABELS TO NAMES ===
        label_map = {0: "Iris-setosa", 1: "Iris-versicolor", 2: "Iris-virginica"}
        # Check if first element is numeric (int or float) to decide whether to map
        if np.issubdtype(y_raw.dtype, np.number):
             y_raw = pd.Series(y_raw).map(label_map).fillna(pd.Series(y_raw)).values

        # Encode Labels
        le = LabelEncoder()
        y = le.fit_transform(y_raw)
        
        # 3. Split
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        # 4. Preprocessing (Scaler)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        # 5. Build Model
        clf = build_model(config)

        # 6. Train
        # Passing integer classes to train_with_validation/partial_fit avoids log_loss string issues
        clf, train_loss, val_loss = train_with_validation(
            clf, X_train_scaled, y_train, X_val_scaled, y_val, epochs=epochs
        )

        # 7. Create Pipeline
        # Note: We must save the Fitted attributes.
        pipeline = Pipeline([
            ("scaler", scaler),
            ("clf", clf)
        ])

        # Save Training Stats for Drift Detection
        STATS_PATH = os.path.join(MODEL_DIR, "training_stats.json")
        stats = {
            "mean": list(scaler.mean_),
            "scale": list(scaler.scale_),
            "features": ["Sepal Length", "Sepal Width", "Petal Length", "Petal Width"]
        }
        with open(STATS_PATH, "w") as f:
            json.dump(stats, f)

        losses_history = {
            "train": train_loss,
            "val": val_loss
        }

        # 8. Save
        joblib.dump(pipeline, MODEL_PATH)
        joblib.dump(le, LE_PATH)
        with open(LOSS_PATH, "w") as f:
            json.dump(losses_history, f)

        # VERSIONING: Save timestamped copy
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        # Calculate final accuracy on validation set
        final_val_acc = accuracy_score(y_val, clf.predict(X_val_scaled))
        current_model_meta["accuracy"] = final_val_acc
        
        SAVED_MODELS_DIR = os.path.join(MODEL_DIR, "saved_models")
        os.makedirs(SAVED_MODELS_DIR, exist_ok=True)
        version_name = f"model_{timestamp}_acc{final_val_acc:.2f}.pkl"
        joblib.dump(pipeline, os.path.join(SAVED_MODELS_DIR, version_name))

        # Update global memory
        model_pipeline = pipeline
        label_encoder = le

        training_status["epoch"] = epochs
        
    except Exception as e:
        print(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        training_status["error"] = str(e)
    finally:
        training_status["running"] = False


# ROUTES
@app.route("/")
def home():
    return render_template("index.html", history=prediction_history)

@app.route("/about")
def about():
    # Feature 4: Model Info Page
    # Load config for display
    try:
        config = load_config()
    except:
        config = {"model": {"max_iter": "Unknown", "alpha": "Unknown"}, "data": {"test_size": "Unknown"}}

    model_info = {
        "name": "Stochastic Gradient Descent (SGD) Classifier",
        "type": "Linear Classifier with Log Loss (Softmax)",
        "features": ["Sepal Length", "Sepal Width", "Petal Length", "Petal Width"],
        "classes": ["Iris-setosa", "Iris-versicolor", "Iris-virginica"],
        "dataset_size": 150,
    }

    # Pass check to see if images exist
    images = {
        "loss_curve": os.path.exists(os.path.join(STATIC_DIR, "loss_curve.png")),
        "confusion_matrix": os.path.exists(os.path.join(STATIC_DIR, "confusion_matrix.png")),
        "scatter_2d": os.path.exists(os.path.join(STATIC_DIR, "scatter_2d.png")),
        "accuracy_bar": os.path.exists(os.path.join(STATIC_DIR, "accuracy_bar.png")),
    }
    return render_template("about.html", images=images, training_status=training_status, config=config, model_info=model_info)

@app.route("/train", methods=["POST"])
def train():
    if training_status["running"]:
        return jsonify({"status": "already running"})
    
    # Reset history on new train
    global losses_history
    losses_history = {"train": [], "val": []}

    thread = threading.Thread(target=run_training_task)
    thread.start()
    return jsonify({"status": "started"})

@app.route("/train_status")
def train_status():
    return jsonify(training_status)

@app.route("/predict", methods=["POST"])
def predict():
    if model_pipeline is None or label_encoder is None:
        return render_template("index.html", error="Model has not been trained yet.", history=prediction_history)

    try:
        # Extract features
        features_dict = {
            "sepal_length": float(request.form["sepal_length"]),
            "sepal_width": float(request.form["sepal_width"]),
            "petal_length": float(request.form["petal_length"]),
            "petal_width": float(request.form["petal_width"])
        }
        features = list(features_dict.values())
        
        # Predict
        # Pipeline handles scaling automatically
        # Prediction returns Int index because we trained on Encoded y
        pred_idx = model_pipeline.predict([features])[0]
        probs = model_pipeline.predict_proba([features])[0]
        
        # Decode label
        pred_label = label_encoder.inverse_transform([pred_idx])[0]
        confidence = float(np.max(probs))

        # Feature 2: Probability Distribution
        class_probs = []
        for i, prob in enumerate(probs):
            class_name = label_encoder.inverse_transform([i])[0]
            class_probs.append({"name": class_name, "prob": float(prob)})
        
        # Sort by probability descending
        class_probs.sort(key=lambda x: x["prob"], reverse=True)

        # Feature 3: Update History
        prediction_record = {
            "time": time.strftime("%H:%M:%S"),
            "input": features_dict,
            "prediction": pred_label,
            "confidence": confidence
        }
        # Prepend to history, keep max 10
        prediction_history.insert(0, prediction_record)
        if len(prediction_history) > 10:
            prediction_history.pop()

        return render_template(
            "index.html",
            result={
                "label": pred_label,
                "confidence": confidence,
                "probs": class_probs
            },
            history=prediction_history
        )
    except Exception as e:
        return render_template("index.html", error=f"Prediction Error: {str(e)}", history=prediction_history)

@app.route("/api/predict", methods=["POST"])
def api_predict():
    # Feature 5: API Endpoint
    if model_pipeline is None or label_encoder is None:
        return jsonify({"error": "Model has not been trained yet."}), 503

    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
            
        required_fields = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
        if not all(k in data for k in required_fields):
            return jsonify({"error": f"Missing fields. Required: {required_fields}"}), 400

        features = [
            float(data["sepal_length"]),
            float(data["sepal_width"]),
            float(data["petal_length"]),
            float(data["petal_width"])
        ]

        pred_idx = model_pipeline.predict([features])[0]
        probs = model_pipeline.predict_proba([features])[0]
        pred_label = label_encoder.inverse_transform([pred_idx])[0]
        
        # Create probability dict
        prob_dict = {}
        for i, prob in enumerate(probs):
            class_name = label_encoder.inverse_transform([i])[0]
            prob_dict[class_name] = float(prob)

        return jsonify({
            "prediction": pred_label,
            "confidence": float(np.max(probs)),
            "probabilities": prob_dict
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/predict_csv", methods=["POST"])
def predict_csv():
    if model_pipeline is None or label_encoder is None:
        return render_template("index.html", error="Model has not been trained yet.", history=prediction_history)

    file = request.files.get("file")
    if not file:
        return render_template("index.html", error="No file uploaded.", history=prediction_history)

    try:
        df = pd.read_csv(file)
        
        if df.shape[1] < 4:
            return render_template("index.html", error="CSV must have at least 4 feature columns.", history=prediction_history)

        X = df.iloc[:, :4].values
        
        # Predict
        pred_idxs = model_pipeline.predict(X)
        probs = model_pipeline.predict_proba(X)
        confidences = np.max(probs, axis=1)
        
        # Decode predictions
        pred_labels = label_encoder.inverse_transform(pred_idxs)

        df["Predicted"] = pred_labels
        df["Confidence"] = confidences

        # Metrics & Plots
        metrics = {"total_samples": len(df), "accuracy": "N/A", "has_labels": False}
        
        # If labels exist (5th column) and assuming they are Strings or whatever original format
        if df.shape[1] >= 5: 
            metrics["has_labels"] = True
            y_true_raw = df.iloc[:, 4].values

            # === MAP NUMERIC LABELS TO NAMES (Fix for 0% Accuracy) ===
            label_map = {0: "Iris-setosa", 1: "Iris-versicolor", 2: "Iris-virginica"}
            if np.issubdtype(y_true_raw.dtype, np.number):
                 y_true_raw = pd.Series(y_true_raw).map(label_map).fillna(pd.Series(y_true_raw)).values
            
            # We compare Strings vs Strings
            # Ensure y_true_raw are strings
            y_true_str = y_true_raw.astype(str)
            y_pred_str = pred_labels.astype(str)
            
            acc = accuracy_score(y_true_str, y_pred_str)
            metrics["accuracy"] = f"{acc:.2%}"

            # Confusion Matrix
            cm = confusion_matrix(y_true_str, y_pred_str)
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='RdPu', 
                        xticklabels=np.unique(y_pred_str), yticklabels=np.unique(y_pred_str)) 
            
            plt.xlabel("Predicted")
            plt.ylabel("Actual")
            plt.title("Confusion Matrix")
            plt.tight_layout()
            plt.savefig(f"{STATIC_DIR}/confusion_matrix.png")
            plt.close()

            c_data = y_true_str
            legend_prefix = "True"
        else:
            c_data = pred_labels
            legend_prefix = "Pred"

        # Loss Curve Plot
        plt.figure(figsize=(10, 5))
        if isinstance(losses_history, dict):
            plt.plot(losses_history.get("train", []), label="Training Loss", color="#6366f1", linewidth=2)
            if "val" in losses_history and len(losses_history["val"]) > 0:
                plt.plot(losses_history["val"], label="Validation Loss", color="#ec4899", linewidth=2, linestyle="--")
        else:
            plt.plot(losses_history, label="Training Loss", color="#6366f1")
            
        plt.title("Training Progress (Log Loss)")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig(f"{STATIC_DIR}/loss_curve.png")
        plt.close()

        # Sample Loss / Uncertainty
        sample_losses = [ -np.log(p + 1e-9) for p in confidences ]
        plt.figure(figsize=(10, 5))
        plt.plot(sample_losses, color="#f59e0b", alpha=0.7)
        plt.fill_between(range(len(sample_losses)), sample_losses, color="#f59e0b", alpha=0.1)
        plt.title("Prediction Uncertainty (Entropy)")
        plt.xlabel("Sample Index")
        plt.ylabel("Uncertainty")
        plt.tight_layout()
        plt.savefig(f"{STATIC_DIR}/sample_loss.png")
        plt.close()

        # 3D Scatter Plot
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")
        
        unique_labels = np.unique(c_data)
        
        for i, label in enumerate(unique_labels):
            idx = (c_data == str(label))
            ax.scatter(X[idx,0], X[idx,1], X[idx,2], label=f"{legend_prefix} {label}", s=50, alpha=0.8)
            
        ax.set_xlabel("Sepal Length")
        ax.set_ylabel("Sepal Width")
        ax.set_zlabel("Petal Length")
        ax.legend()
        plt.tight_layout()
        plt.savefig(f"{STATIC_DIR}/scatter_3d.png")
        plt.close()

        # 2D Feature Space Plot with Decision Boundary
        plt.figure(figsize=(10, 8))
        
        # Create Meshgrid for Decision Boundary (Feature 0 vs Feature 1)
        x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
        y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                             np.arange(y_min, y_max, 0.02))
        
        # Fill other features (2 and 3) with MEAN values from current batch
        mean_feat2 = X[:, 2].mean()
        mean_feat3 = X[:, 3].mean()
        
        mesh_data = np.c_[xx.ravel(), yy.ravel(), 
                          np.full(xx.ravel().shape, mean_feat2), 
                          np.full(xx.ravel().shape, mean_feat3)]
        
        # Predict on mesh
        Z = model_pipeline.predict(mesh_data)
        Z = Z.reshape(xx.shape)
        
        # Plot Contours
        plt.contourf(xx, yy, Z, alpha=0.3, cmap="viridis")
        
        # Scatter Plot
        for i, label in enumerate(unique_labels):
            idx = (c_data == str(label))
            plt.scatter(X[idx, 0], X[idx, 1], label=f"{legend_prefix} {label}", s=50, edgecolors='k', alpha=0.9)
            
        plt.xlabel("Sepal Length")
        plt.ylabel("Sepal Width")
        plt.title(f"2D Feature Space & Decision Boundary\n(Fixed Petal L={mean_feat2:.2f}, W={mean_feat3:.2f})")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{STATIC_DIR}/scatter_2d.png")
        plt.close()

        # Accuracy Bar Chart with Percentage Labels
        plt.figure(figsize=(8, 5))
        
        if metrics["has_labels"]:
            # Per-class accuracy
            classes = np.unique(y_true_str)
            accs = []
            for c in classes:
                mask = (y_true_str == c)
                acc = accuracy_score(y_true_str[mask], y_pred_str[mask]) if np.sum(mask) > 0 else 0
                accs.append(acc)
            
            ax = sns.barplot(x=classes, y=accs, palette="viridis")
            plt.ylim(0, 1.15) # More space for labels
            plt.ylabel("Accuracy (Recall)")
            plt.title("Accuracy per Class")
            
            # Update DataFrame to show Mapped Labels key instead of numbers
            df.iloc[:, 4] = y_true_str

            # Add Percentage Labels
            for i, p in enumerate(ax.patches):
                 height = p.get_height()
                 ax.text(p.get_x() + p.get_width()/2., height + 0.02, 
                         f'{height:.1%}', ha="center", fontsize=11, fontweight='bold', color='black')

        else:
            # Distribution
            ax = sns.countplot(x=pred_labels, palette="viridis")
            plt.ylabel("Count")
            plt.title("Prediction Distribution")
            
            # Add Count Labels
            for i, p in enumerate(ax.patches):
                 height = p.get_height()
                 ax.text(p.get_x() + p.get_width()/2., height + 0.1, 
                         f'{int(height)}', ha="center", fontsize=11, fontweight='bold')
            
        plt.tight_layout()
        plt.savefig(f"{STATIC_DIR}/accuracy_bar.png")
        plt.close()
        
        # Show Full Table (150 rows is manageable) so user sees all classes
        table_html = df.to_html(classes="table", index=False)

        return render_template(
            "index.html",
            csv_result=True,
            metrics=metrics,
            table_html=table_html,
            history=prediction_history
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        return render_template("index.html", error=f"Analysis failed: {str(e)}", history=prediction_history)


# === LABS APIs ===

@app.route("/api/explain", methods=["GET"])
def explain_model():
    if model_pipeline is None:
        return jsonify({"error": "Model not trained"}), 503
        
    try:
        clf = model_pipeline.named_steps['clf']
        # For multi-class, coef_ is (n_classes, n_features). We take mean absolute value per feature.
        # Check if coef_ exists (Linear models)
        if hasattr(clf, "coef_"):
            importances = np.mean(np.abs(clf.coef_), axis=0)
        else:
            # Fallback for non-linear models if we ever switch
            return jsonify({"error": "Model does not support coefficient extraction"}), 400

        feature_names = ["Sepal Length", "Sepal Width", "Petal Length", "Petal Width"]
        
        explanations = []
        for name, val in zip(feature_names, importances):
            explanations.append({"name": name, "weight": float(val)})
            
        # Sort by weight desc
        explanations.sort(key=lambda x: x["weight"], reverse=True)
        
        return jsonify(explanations)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/detect_drift", methods=["POST"])
def detect_drift():
    # Load training stats
    STATS_PATH = os.path.join(MODEL_DIR, "training_stats.json")
    if not os.path.exists(STATS_PATH):
        return jsonify({"error": "No training stats found. Train model first."}), 404
        
    with open(STATS_PATH, "r") as f:
        train_stats = json.load(f)
        
    file = request.files.get("file")
    threshold = float(request.form.get("threshold", 20))
    
    if not file:
        return jsonify({"error": "No file uploaded"}), 400
        
    try:
        df = pd.read_csv(file)
        if df.shape[1] < 4:
            return jsonify({"error": "CSV needs 4 feature columns"}), 400
            
        current_data = df.iloc[:, :4].values
        current_means = np.mean(current_data, axis=0)
        
        ref_means = np.array(train_stats["mean"])
        
        alerts = []
        features = train_stats["features"]
        
        max_drift = 0
        
        for i, (curr, ref) in enumerate(zip(current_means, ref_means)):
            # Avoid div by zero
            if abs(ref) < 1e-9: ref = 1e-9
            
            drift_pct = abs(curr - ref) / abs(ref) * 100
            max_drift = max(max_drift, drift_pct)
            
            if drift_pct > threshold:
                alerts.append({
                    "feature": features[i],
                    "drift_pct": float(f"{drift_pct:.1f}"),
                    "message": f"Deviated by {drift_pct:.1f}%"
                })
                
        status = "safe" if not alerts else "warning"
        
        return jsonify({
            "status": status,
            "alerts": alerts,
            "max_drift": float(f"{max_drift:.1f}"),
            "msg": "Data Drift Detected" if alerts else "No significant data drift detected"
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/lab/train", methods=["POST"])
def lab_train():
    # Interactive Training Lab - Trains a temporary model and returns results immediately
    try:
        data = request.get_json()
        alpha = float(data.get("alpha", 0.0001))
        epochs = int(data.get("epochs", 50))
        
        # Load Data (Quickly re-load for playground)
        if os.path.exists(DATA_PATH):
            df = pd.read_csv(DATA_PATH)
        else:
             return jsonify({"error": "Dataset not found"}), 404
             
        X = df.iloc[:, :4].values
        y_raw = df.iloc[:, 4].values
        
        # Map/Encode
        label_map = {0: "Iris-setosa", 1: "Iris-versicolor", 2: "Iris-virginica"}
        if np.issubdtype(y_raw.dtype, np.number):
             y_raw = pd.Series(y_raw).map(label_map).fillna(pd.Series(y_raw)).values
        
        le = LabelEncoder()
        y = le.fit_transform(y_raw)
        
        X_train, X_test_lab, y_train, y_test_lab = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test_lab)
        
        # Override config for this run
        from sklearn.linear_model import SGDClassifier
        clf = SGDClassifier(loss='log_loss', max_iter=1, alpha=alpha, random_state=42, warm_start=True)
        
        losses = []
        import sklearn.metrics
        
        for epoch in range(epochs):
            clf.fit(X_train_scaled, y_train) # warm_start=True keeps weights
            probs = clf.predict_proba(X_train_scaled)
            l = sklearn.metrics.log_loss(y_train, probs)
            losses.append(l)
            
        # Final evaluation
        final_acc = clf.score(X_test_scaled, y_test_lab)
        
        # Current Global Model Acc (for comparison)
        current_acc = "N/A"
        if 'accuracy' in current_model_meta:
            current_acc = f"{current_model_meta['accuracy'] * 100:.1f}%"
            
        return jsonify({
            "status": "success",
            "epochs": list(range(1, epochs+1)),
            "loss": losses,
            "new_accuracy": f"{final_acc * 100:.1f}%",
            "current_accuracy": current_acc,
            "msg": "🎯 Model converged successfully"
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

# === MODEL VERSIONING APIs ===

@app.route("/api/models", methods=["GET"])
def list_models():
    """List all available models in the saved_models directory."""
    try:
        SAVED_MODELS_DIR = os.path.join(MODEL_DIR, "saved_models")
        if not os.path.exists(SAVED_MODELS_DIR):
            os.makedirs(SAVED_MODELS_DIR, exist_ok=True)
            
        files = [f for f in os.listdir(SAVED_MODELS_DIR) if f.endswith(".pkl")]
        
        # Sort by creation time (newest first)
        files.sort(key=lambda x: os.path.getmtime(os.path.join(SAVED_MODELS_DIR, x)), reverse=True)
        
        models = []
        for f in files:
            path = os.path.join(SAVED_MODELS_DIR, f)
            timestamp = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(os.path.getmtime(path)))
            size = f"{os.path.getsize(path) / 1024:.1f} KB"
            
            models.append({
                "filename": f,
                "created": timestamp,
                "size": size,
                "active": False # UI can infer this based on selection or simple check
            })
            
        return jsonify(models)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/models/activate", methods=["POST"])
def activate_model():
    """Load a specific model file."""
    global model_pipeline
    data = request.get_json()
    filename = data.get("filename")
    
    if not filename:
        return jsonify({"error": "Filename required"}), 400
        
    path = os.path.join(MODEL_DIR, "saved_models", filename)
    if not os.path.exists(path):
        return jsonify({"error": "File not found"}), 404
        
    try:
        model_pipeline = joblib.load(path)
        return jsonify({"status": "success", "msg": f"Activated model: {filename}"})
    except Exception as e:
        return jsonify({"error": f"Failed to load model: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)
