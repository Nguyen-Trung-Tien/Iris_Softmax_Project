document.addEventListener("DOMContentLoaded", function () {
  // Initialize Tabs
  const tabs = document.querySelectorAll(".tab-btn");
  tabs.forEach((tab) => {
    tab.addEventListener("click", () => {
      document
        .querySelectorAll(".tab-btn")
        .forEach((t) => t.classList.remove("active"));
      document
        .querySelectorAll(".tab-pane")
        .forEach((p) => p.classList.remove("active"));

      tab.classList.add("active");
      const target = document.getElementById(tab.dataset.target);
      if (target) target.classList.add("active");

      // Trigger data load if needed
      if (tab.dataset.target === "models") loadModels();
      if (tab.dataset.target === "insights") loadInsights();
    });
  });

  // Handle Training Lab Slider real-time values
  const alphaSlider = document.getElementById("labAlpha");
  const epochSlider = document.getElementById("labEpochs");
  if (alphaSlider) {
    alphaSlider.addEventListener(
      "input",
      (e) => (document.getElementById("valAlpha").textContent = e.target.value),
    );
  }
  if (epochSlider) {
    epochSlider.addEventListener(
      "input",
      (e) =>
        (document.getElementById("valEpochs").textContent = e.target.value),
    );
  }

  // === FILE UPLOAD INTERACTIVITY (BATCH) ===
  setupDropZone("dropZone", "fileInput", ".file-msg");

  // === FILE UPLOAD INTERACTIVITY (DRIFT) ===
  setupDropZone("driftDropZone", "driftFile", "#driftFileMsg");

  function setupDropZone(zoneId, inputId, msgSelector) {
    const dropZone = document.getElementById(zoneId);
    const fileInput = document.getElementById(inputId);
    // For drift, the msg selector is an ID, for batch it was a class in the previous code,
    // but let's handle both querySelector styles.
    const fileMsg = dropZone
      ? dropZone.querySelector(msgSelector) ||
        document.querySelector(msgSelector)
      : null;

    if (dropZone && fileInput) {
      ["dragenter", "dragover", "dragleave", "drop"].forEach((eventName) => {
        dropZone.addEventListener(eventName, preventDefaults, false);
      });

      function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
      }

      ["dragenter", "dragover"].forEach((eventName) => {
        dropZone.addEventListener(
          eventName,
          () => dropZone.classList.add("dragover"),
          false,
        );
      });

      ["dragleave", "drop"].forEach((eventName) => {
        dropZone.addEventListener(
          eventName,
          () => dropZone.classList.remove("dragover"),
          false,
        );
      });

      fileInput.addEventListener("change", function () {
        if (this.files.length > 0) {
          if (fileMsg) {
            fileMsg.textContent = "Selected: " + this.files[0].name;
            fileMsg.style.color = "var(--primary)";
            fileMsg.style.fontWeight = "700";
          }
        }
      });
    }
  }
});

// === MODEL MANAGER ===
function loadModels() {
  const listEl = document.getElementById("modelList");
  listEl.innerHTML = '<div class="spinner"></div> Loading models...';

  fetch("/api/models")
    .then((r) => r.json())
    .then((data) => {
      if (data.error) {
        listEl.innerHTML = `<div class="error">${data.error}</div>`;
        return;
      }
      if (data.length === 0) {
        listEl.innerHTML =
          '<div class="empty">No models saved yet. Train a model first.</div>';
        return;
      }

      let html =
        '<table class="data-table"><thead><tr><th>Created</th><th>Filename</th><th>Size</th><th>Action</th></tr></thead><tbody>';
      data.forEach((m) => {
        html += `<tr>
                    <td>${m.created}</td>
                    <td>${m.filename}</td>
                    <td>${m.size}</td>
                    <td>
                        <button class="btn-sm btn-action" onclick="activateModel('${m.filename}', this)" title="Activate this version">
                            <i class="fas fa-check-circle"></i> Activate
                        </button>
                        <button class="btn-sm btn-action" onclick="deleteModel('${m.filename}', this)" style="background: white; color: var(--danger); border: 1px solid var(--danger); margin-left: 0.5rem;" title="Delete this version">
                            <i class="fas fa-trash"></i> Delete
                        </button>
                    </td>
                </tr>`;
      });
      html += "</tbody></table>";
      listEl.innerHTML = html;
    })
    .catch((e) => {
      listEl.innerHTML = `<div class="error">Failed to load models: ${e}</div>`;
    });
}

function activateModel(filename) {
  if (!confirm(`Activate model ${filename}?`)) return;

  fetch("/api/models/activate", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ filename: filename }),
  })
    .then((r) => r.json())
    .then((data) => {
      if (data.status === "success") {
        showToast(data.msg, "success");
        // Update current model info (optional reload)
        setTimeout(() => loadModels(), 500); // Reload list to show active status if we had it
      } else {
        showToast("Error: " + (data.error || "Unknown"), "error");
      }
    })
    .catch((e) => showToast("Network error: " + e, "error"));
}

function deleteModel(filename, btn) {
  if (!confirm(`Are you sure you want to PERMANENTLY DELETE ${filename}?`))
    return;

  // Show spinner
  let originalText = "";
  if (btn) {
    originalText = btn.innerHTML;
    btn.disabled = true;
    btn.innerHTML = '<i class="fas fa-spinner fa-spin"></i>';
  }

  fetch("/api/models", {
    method: "DELETE",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ filename: filename }),
  })
    .then((r) => r.json())
    .then((data) => {
      if (data.status === "success") {
        showToast(data.msg, "success");
        setTimeout(() => loadModels(), 500);
      } else {
        showToast("Error: " + data.error, "error");
        // Reset button only on error (success reloads list)
        if (btn) {
          btn.disabled = false;
          btn.innerHTML = originalText;
        }
      }
    })
    .catch((e) => {
      showToast("Network Model Error: " + e, "error");
      if (btn) {
        btn.disabled = false;
        btn.innerHTML = originalText;
      }
    });
}

// === TOAST NOTIFICATION ===
function showToast(message, type = "success") {
  // Create toast element if not exists
  let toast = document.getElementById("toast");
  if (!toast) {
    toast = document.createElement("div");
    toast.id = "toast";
    toast.className = "toast";
    document.body.appendChild(toast);
  }

  // Set Content
  const icon =
    type === "success"
      ? '<i class="fas fa-check-circle"></i>'
      : '<i class="fas fa-exclamation-circle"></i>';
  toast.innerHTML = icon + message;
  toast.className = `toast show ${type}`;

  // Hide after 3s
  setTimeout(() => {
    toast.className = toast.className.replace("show", "");
  }, 3000);
}

function activateModel(filename) {
  if (!confirm(`Activate model ${filename}?`)) return;

  fetch("/api/models/activate", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ filename: filename }),
  })
    .then((r) => r.json())
    .then((data) => {
      if (data.status === "success") {
        showToast(data.msg, "success");
        // Update current model info (optional reload)
        setTimeout(() => loadModels(), 500); // Reload list to show active status if we had it
      } else {
        showToast("Error: " + (data.error || "Unknown"), "error");
      }
    })
    .catch((e) => showToast("Network error: " + e, "error"));
}

// === INSIGHTS ===
let importanceChart = null;

function loadInsights() {
  loadFeatureImportance();
  // Drift is manual upload
}

function loadFeatureImportance() {
  const ctx = document.getElementById("importanceChart").getContext("2d");

  fetch("/api/explain")
    .then((r) => r.json())
    .then((data) => {
      if (data.error) {
        // Model not trained
        // render empty or error
        return;
      }

      const labels = data.map((d) => d.name);
      const values = data.map((d) => d.weight);

      if (importanceChart) importanceChart.destroy();

      importanceChart = new Chart(ctx, {
        type: "bar",
        data: {
          labels: labels,
          datasets: [
            {
              label: "Feature Importance (Mean |Coef|)",
              data: values,
              backgroundColor: "rgba(99, 102, 241, 0.7)",
              borderColor: "rgba(99, 102, 241, 1)",
              borderWidth: 1,
            },
          ],
        },
        options: {
          indexAxis: "y",
          responsive: true,
          plugins: {
            legend: { display: false },
          },
        },
      });
    });
}

// === DRIFT DETECTION ===
function checkDrift() {
  const fileInput = document.getElementById("driftFile");
  const threshold = document.getElementById("driftThreshold").value;
  const resultDiv = document.getElementById("driftResult");

  if (fileInput.files.length === 0) {
    alert("Please select a CSV file first.");
    return;
  }

  const formData = new FormData();
  formData.append("file", fileInput.files[0]);
  formData.append("threshold", threshold);

  resultDiv.innerHTML = '<div class="spinner"></div> Analyzing...';

  fetch("/api/detect_drift", {
    method: "POST",
    body: formData,
  })
    .then((r) => r.json())
    .then((data) => {
      if (data.error) {
        resultDiv.innerHTML = `<div class="error-box">${data.error}</div>`;
        return;
      }

      let html = "";
      if (data.status === "warning") {
        html += `<div class="alert-box warning"><i class="fas fa-exclamation-triangle"></i> <strong>Data Drift Detected!</strong> Max Drift: ${data.max_drift}%</div>`;
        html += '<ul class="drift-list">';
        data.alerts.forEach((a) => {
          html += `<li>${a.feature}: ${a.message}</li>`;
        });
        html += "</ul>";
        html +=
          '<div class="suggestion">💡 Recommendation: Retrain model with new data.</div>';
      } else {
        html += `<div class="alert-box success"><i class="fas fa-check-circle"></i> No significant drift detected. Max deviation: ${data.max_drift}%</div>`;
      }

      resultDiv.innerHTML = html;
    })
    .catch((e) => {
      resultDiv.innerHTML = `<div class="error-box">Analysis failed: ${e}</div>`;
    });
}

// === LAB TRAINING ===
let labLossChart = null;

function runLab() {
  const alpha = parseFloat(document.getElementById("labAlpha").value);
  const epochs = parseInt(document.getElementById("labEpochs").value);
  const resultDiv = document.getElementById("labResult");
  const ctx = document.getElementById("labChart").getContext("2d");

  const btn = document.querySelector(".btn-lab");
  btn.disabled = true;
  btn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Training...';

  fetch("/api/lab/train", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ alpha, epochs }),
  })
    .then((r) => r.json())
    .then((data) => {
      btn.disabled = false;
      btn.innerHTML = '<i class="fas fa-flask"></i> Run Experiment';

      if (data.error) {
        resultDiv.innerHTML = `<div class="error">${data.error}</div>`;
        return;
      }

      // Render Metrics
      document.getElementById("labNewAcc").textContent = data.new_accuracy;
      document.getElementById("labCurrAcc").textContent = data.current_accuracy;

      // Render Chart
      if (labLossChart) labLossChart.destroy();

      labLossChart = new Chart(ctx, {
        type: "line",
        data: {
          labels: data.epochs,
          datasets: [
            {
              label: "Training Log Loss",
              data: data.loss,
              borderColor: "#f59e0b",
              backgroundColor: "rgba(245, 158, 11, 0.1)",
              tension: 0.3,
              fill: true,
            },
          ],
        },
        options: {
          responsive: true,
          scales: {
            y: {
              beginAtZero: false,
              title: { display: true, text: "Log Loss" },
            },
            x: { title: { display: true, text: "Epoch" } },
          },
        },
      });
    })
    .catch((e) => {
      btn.disabled = false;
      btn.innerHTML = '<i class="fas fa-flask"></i> Run Experiment';
      alert("Lab Error: " + e);
    });
}
