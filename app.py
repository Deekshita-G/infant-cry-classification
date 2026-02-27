from flask import Flask, render_template, request, jsonify
from pydub import AudioSegment
import numpy as np
import librosa
import joblib
import json
import os
import uuid

app = Flask(__name__)

# =========================
# SAFE BASE DIRECTORY (IMPORTANT FOR HUGGINGFACE)
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
TEMP_DIR = os.path.join(BASE_DIR, "temp")
os.makedirs(TEMP_DIR, exist_ok=True)

# =========================
# LOAD MODELS (LOAD ONCE ONLY)
# =========================
asphyxia_model = joblib.load(os.path.join(MODELS_DIR, "asphyxia_rf_model.pkl"))
asphyxia_scaler = joblib.load(os.path.join(MODELS_DIR, "asphyxia_scaler.pkl"))
ASPHYXIA_THRESHOLD = float(joblib.load(os.path.join(MODELS_DIR, "asphyxia_threshold.pkl")))

severity_threshold = float(joblib.load(os.path.join(MODELS_DIR, "severity_threshold.pkl")))
csi_min = float(joblib.load(os.path.join(MODELS_DIR, "csi_min.pkl")))
csi_max = float(joblib.load(os.path.join(MODELS_DIR, "csi_max.pkl")))

with open(os.path.join(MODELS_DIR, "preprocess_config.json")) as f:
    config = json.load(f)

SR = config.get("sample_rate", 16000)
DURATION = config.get("duration", 3.0)
N_MFCC = config.get("n_mfcc", 20)
MAX_LEN = int(SR * DURATION)

# =========================
# AUDIO FUNCTIONS
# =========================

def load_audio(path):
    y, _ = librosa.load(path, sr=SR, mono=True)
    return y

import librosa
import numpy as np

def is_valid_cry(audio, sr):

    # Minimum duration check
    duration = len(audio) / sr
    if duration < 0.5:
        return False

    # Basic energy check
    rms = np.mean(librosa.feature.rms(y=audio))

    # Lower threshold (more tolerant)
    if rms < 0.002:
        return False

    return True
def convert_to_wav(input_path):
    try:
        audio = AudioSegment.from_file(input_path)
        wav_path = input_path + ".wav"
        audio.export(wav_path, format="wav")
        return wav_path
    except Exception as e:
        print("Conversion failed:", e)
        return input_path

def extract_features(audio):

    if len(audio) < MAX_LEN:
        audio = np.pad(audio, (0, MAX_LEN - len(audio)))
    else:
        audio = audio[:MAX_LEN]

    mfcc = librosa.feature.mfcc(y=audio, sr=SR, n_mfcc=N_MFCC)

    return np.hstack([
        mfcc.mean(axis=1),
        mfcc.std(axis=1),
        librosa.feature.rms(y=audio).mean(),
        librosa.feature.zero_crossing_rate(audio).mean(),
        librosa.feature.spectral_centroid(y=audio, sr=SR).mean(),
        librosa.feature.spectral_bandwidth(y=audio, sr=SR).mean()
    ])

def compute_csi(features):

    mfcc_std = np.mean(features[N_MFCC:2*N_MFCC])
    rms = features[2*N_MFCC]
    centroid = features[2*N_MFCC+2]
    bandwidth = features[2*N_MFCC+3]

    csi = (
        0.4 * rms +
        0.2 * mfcc_std +
        0.2 * bandwidth +
        0.2 * centroid
    )

    return float(csi)

def guess_possible_cause(audio, features):

    rms = float(np.mean(librosa.feature.rms(y=audio)))
    zcr = float(np.mean(librosa.feature.zero_crossing_rate(audio)))
    centroid = float(np.mean(librosa.feature.spectral_centroid(y=audio, sr=SR)))

    if rms > 0.05 and centroid > 2000:
        return "Possible cause: Hungry cry pattern detected 🍼"

    if zcr > 0.12 and centroid > 2500:
        return "Possible cause: Pain or discomfort cry pattern 🤕"

    if rms < 0.02 and centroid < 1800:
        return "Possible cause: Tired / sleepy cry pattern 😴"

    return "Possible cause: General distress pattern 👶"

# 🔴 CORE FUNCTION
def predict_from_full_audio(full_audio):

    window = MAX_LEN
    step = window

    energies = []

    for start in range(0, len(full_audio)-window+1, step):
        segment = full_audio[start:start+window]
        rms = float(np.mean(librosa.feature.rms(y=segment)))
        energies.append((rms, start))

    energies.sort(reverse=True, key=lambda x: x[0])

    best_prob = 0
    best_segment = None

    for _, start in energies[:3]:

        segment = full_audio[start:start+window]

        features = extract_features(segment).reshape(1,-1)
        features_scaled = asphyxia_scaler.transform(features)

        probs = asphyxia_model.predict_proba(features_scaled)[0]
        asphyxia_prob = float(probs[1])  # Make sure 1 = Asphyxia class
        classes = asphyxia_model.classes_
        asphyxia_index = list(classes).index(1)  # change to 0 if reversed
        prob = float(probs[asphyxia_index])

        if prob > best_prob:
            best_prob = prob
            best_segment = segment

    if best_segment is None:
        best_segment = full_audio[:window]
        if len(best_segment) < window:
            best_segment = np.pad(best_segment, (0, window-len(best_segment)))

        features = extract_features(best_segment).reshape(1,-1)
        features_scaled = asphyxia_scaler.transform(features)
        best_prob = float(asphyxia_model.predict_proba(features_scaled)[0,1])

    return asphyxia_prob, best_segment

# =========================
# ROUTES
# =========================

@app.route("/")
def home():
    return render_template("index.html")

from pydub import AudioSegment
import io
import tempfile

@app.route("/predict", methods=["POST"])
def predict():

    temp_original = None
    temp_path = None

    try:
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files["file"]

        temp_original = os.path.join(
            TEMP_DIR, f"{uuid.uuid4().hex}_{file.filename}"
        )
        file.save(temp_original)

        temp_path = os.path.join(
            TEMP_DIR, f"{uuid.uuid4().hex}.wav"
        )

        audio = AudioSegment.from_file(temp_original)
        audio = audio.set_channels(1).set_frame_rate(SR)
        audio.export(temp_path, format="wav")

        full_audio = load_audio(temp_path)

        # 🔴 Stage 1: Asphyxia Detection (Threshold = 0.4)

        asphyxia_prob, best_segment = predict_from_full_audio(full_audio)

        if asphyxia_prob >= 0.4:
            return jsonify({
                "prediction": "Asphyxia Detected",
                "confidence": round(asphyxia_prob, 3),
                "advice": "Immediate medical evaluation recommended."
            })

        # 🟢 Stage 2: Severity Detection

        features = extract_features(best_segment)
        csi_value = compute_csi(features)

        if csi_max != csi_min:
            csi_normalized = (csi_value - csi_min) / (csi_max - csi_min)
            csi_normalized = max(0.0, min(1.0, csi_normalized))
        else:
            csi_normalized = 0.5

        severity = (
            "Needs Attention Soon"
            if csi_value >= severity_threshold
            else "Baby Seems Okay"
        )

        return jsonify({
            "prediction": severity,
            "confidence": round(float(csi_normalized), 3)
        })

    except Exception as e:
        return jsonify({
            "prediction": "Error",
            "confidence": "-",
            "error": str(e)
        }), 500

    finally:
        try:
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)
            if temp_original and os.path.exists(temp_original):
                os.remove(temp_original)
        except:
            pass
if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 7860))
    app.run(host="0.0.0.0", port=port)