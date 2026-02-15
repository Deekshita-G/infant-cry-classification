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
# LOAD MODELS
# =========================
asphyxia_model = joblib.load("models/asphyxia_rf_model.pkl")
asphyxia_scaler = joblib.load("models/asphyxia_scaler.pkl")
ASPHYXIA_THRESHOLD = float(joblib.load("models/asphyxia_threshold.pkl"))

severity_threshold = float(joblib.load("models/severity_threshold.pkl"))
csi_min = float(joblib.load("models/csi_min.pkl"))
csi_max = float(joblib.load("models/csi_max.pkl"))

with open("models/preprocess_config.json") as f:
    config = json.load(f)

SR = config.get("sample_rate", 16000)
DURATION = config.get("duration", 3.0)
N_MFCC = config.get("n_mfcc", 20)
MAX_LEN = int(SR * DURATION)

TEMP_DIR = "temp"
os.makedirs(TEMP_DIR, exist_ok=True)

# =========================
# AUDIO FUNCTIONS
# =========================

def load_audio(path):
    y, _ = librosa.load(path, sr=SR)
    return y


def is_valid_cry(audio):
    # Only reject silence (do NOT reject real cries accidentally)
    rms = float(np.mean(librosa.feature.rms(y=audio)))
    return rms > 0.002


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

    # ensure fixed length for feature extraction
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
    """
    Lightweight heuristic cause hint.
    DOES NOT affect main prediction.
    """

    rms = float(np.mean(librosa.feature.rms(y=audio)))
    zcr = float(np.mean(librosa.feature.zero_crossing_rate(audio)))
    centroid = float(np.mean(librosa.feature.spectral_centroid(y=audio, sr=SR)))

    # simple interpretable patterns
    if rms > 0.05 and centroid > 2000:
        return "Possible cause: Hungry cry pattern detected 🍼"

    if zcr > 0.12 and centroid > 2500:
        return "Possible cause: Pain or discomfort cry pattern 🤕"

    if rms < 0.02 and centroid < 1800:
        return "Possible cause: Tired / sleepy cry pattern 😴"

    return "Possible cause: General distress pattern 👶"


# 🔴 CORE FUNCTION: scan entire audio file
def predict_from_full_audio(full_audio):

    window = MAX_LEN
    step = int(window * 0.5)

    best_prob = 0
    best_segment = None

    # If shorter than window → pad once
    if len(full_audio) <= window:
        segment = np.pad(full_audio, (0, window - len(full_audio)))
        features = extract_features(segment).reshape(1,-1)
        features_scaled = asphyxia_scaler.transform(features)
        prob = float(asphyxia_model.predict_proba(features_scaled)[0,1])
        return prob, segment

    # Slide window across full file
    for start in range(0, len(full_audio)-window+1, step):

        segment = full_audio[start:start+window]

        # Skip silence segments
        rms = float(np.mean(librosa.feature.rms(y=segment)))
        if rms < 0.002:
            continue

        features = extract_features(segment).reshape(1,-1)
        features_scaled = asphyxia_scaler.transform(features)

        prob = float(asphyxia_model.predict_proba(features_scaled)[0,1])

        if prob > best_prob:
            best_prob = prob
            best_segment = segment

    # fallback if all silent
    if best_segment is None:
        segment = full_audio[:window]
        features = extract_features(segment).reshape(1,-1)
        features_scaled = asphyxia_scaler.transform(features)
        best_prob = float(asphyxia_model.predict_proba(features_scaled)[0,1])
        best_segment = segment

    return best_prob, best_segment


# =========================
# ROUTES
# =========================

@app.route("/")
def home():
    return render_template("index.html")


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

        # Convert to wav if needed
        if not temp_original.lower().endswith(".wav"):
            temp_path = convert_to_wav(temp_original)
        else:
            temp_path = temp_original

        # LOAD FULL AUDIO
        full_audio = load_audio(temp_path)

        # Validate
        if not is_valid_cry(full_audio):
            return jsonify({
                "prediction": "❌ No meaningful sound detected",
                "confidence": "-"
            })

        # SCAN FULL AUDIO
        asphyxia_prob, best_segment = predict_from_full_audio(full_audio)

        # STAGE 1: ASPHYXIA
        if asphyxia_prob >= ASPHYXIA_THRESHOLD:
            return jsonify({
                "prediction": "⚠️ Asphyxia Detected",
                "confidence": round(asphyxia_prob,3)
            })

        # STAGE 2: SEVERITY
        features = extract_features(best_segment)
        csi_value = compute_csi(features)

        if csi_max != csi_min:
            csi_normalized = (csi_value - csi_min) / (csi_max - csi_min)
            csi_normalized = max(0.0, min(1.0, csi_normalized))
        else:
            csi_normalized = 0.5

        severity = (
            "Needs Attention Soon 👶💙"
            if csi_value >= severity_threshold
            else "Baby Seems Okay 🙂🍼"
        )
        cause_hint = guess_possible_cause(full_audio, features.flatten())
        confidence_text = round(float(csi_normalized), 3)

        if confidence_text < 0.55:
            reliability = "Low confidence – overlapping cry patterns"
        elif confidence_text < 0.75:
            reliability = "Moderate confidence"
        else:
            reliability = "High confidence"



        return jsonify({
            "prediction": severity,
            "confidence": confidence_text,
            "hint": cause_hint,
            "reliability": reliability
        })



    except Exception as e:
        print("SERVER ERROR:", str(e))
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
    app.run(host="0.0.0.0", port=5000, debug=True)
