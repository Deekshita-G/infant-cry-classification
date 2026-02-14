from flask import Flask, render_template, request, jsonify
import numpy as np
import librosa
import joblib
import json
import os
import uuid

app = Flask(__name__)

# =========================
# LOAD MODELS
# =========================

asphyxia_model = joblib.load("models/asphyxia_rf_model.pkl")
asphyxia_scaler = joblib.load("models/asphyxia_scaler.pkl")
ASPHYXIA_THRESHOLD = joblib.load("models/asphyxia_threshold.pkl")

severity_threshold = joblib.load("models/severity_threshold.pkl")
csi_min = joblib.load("models/csi_min.pkl")
csi_max = joblib.load("models/csi_max.pkl")

with open("models/preprocess_config.json") as f:
    config = json.load(f)

SR = config.get("sample_rate", 16000)
DURATION = config.get("duration", 3.0)
N_MFCC = config.get("n_mfcc", 20)
MAX_LEN = int(SR * DURATION)

# =========================
# AUDIO PROCESSING
# =========================

def load_audio(path):
    y, _ = librosa.load(path, sr=SR)

    if len(y) < MAX_LEN:
        y = np.pad(y, (0, MAX_LEN - len(y)))
    else:
        y = y[:MAX_LEN]

    return y


def extract_features(audio):
    mfcc = librosa.feature.mfcc(y=audio, sr=SR, n_mfcc=20)

    return np.hstack([
        mfcc.mean(axis=1),
        mfcc.std(axis=1),
        librosa.feature.rms(y=audio).mean(),
        librosa.feature.zero_crossing_rate(audio).mean(),
        librosa.feature.spectral_centroid(y=audio, sr=SR).mean(),
        librosa.feature.spectral_bandwidth(y=audio, sr=SR).mean()
    ])

# =========================
# CRY SEVERITY INDEX
# =========================

def compute_csi(features):
    mfcc_std = np.mean(features[20:40])
    rms = features[40]
    centroid = features[42]
    bandwidth = features[43]

    csi = (
        0.4 * rms +
        0.2 * mfcc_std +
        0.2 * bandwidth +
        0.2 * centroid
    )
    return csi

# =========================
# ROUTES
# =========================

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:

        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files["file"]

        # SAFE TEMP FILE FOR RENDER
        temp_path = f"/tmp/{uuid.uuid4().hex}.wav"
        file.save(temp_path)

        # ---------- Stage 1 ----------
        audio = load_audio(temp_path)
        features = extract_features(audio).reshape(1, -1)

        features_scaled = asphyxia_scaler.transform(features)
        asphyxia_prob = asphyxia_model.predict_proba(features_scaled)[0, 1]

        if asphyxia_prob >= ASPHYXIA_THRESHOLD:
            os.remove(temp_path)
            return jsonify({
                "prediction": "Asphyxia Detected",
                "confidence": round(float(asphyxia_prob), 3)
            })

        # ---------- Stage 2 ----------
        csi_value = compute_csi(features.flatten())

        # SAFE NORMALIZATION
        if csi_max != csi_min:
            csi_normalized = (csi_value - csi_min) / (csi_max - csi_min)
        else:
            csi_normalized = 0.5

        severity = "High Distress" if csi_value >= severity_threshold else "Mild Distress"

        os.remove(temp_path)

        return jsonify({
            "prediction": severity,
            "confidence": round(float(csi_normalized), 3)
        })

    except Exception as e:
        print("SERVER ERROR:", str(e))
        return jsonify({
            "prediction": "Error",
            "confidence": "-",
            "error": str(e)
        }), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
