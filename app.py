from flask import Flask, render_template, request, jsonify
from pydub import AudioSegment
import numpy as np
import librosa
import joblib
import json
import os
import uuid

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
TEMP_DIR = os.path.join(BASE_DIR, "temp")
os.makedirs(TEMP_DIR, exist_ok=True)

# =========================
# Load models
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
# Audio Helpers
# =========================

def load_audio(path):
    y, _ = librosa.load(path, sr=SR, mono=True)
    return y


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

# =========================
# NEW FEATURE EXTRACTION
# =========================

def extract_pitch(audio):

    try:
        pitches = librosa.yin(audio, fmin=80, fmax=600, sr=SR)
        pitches = pitches[pitches > 0]

        if len(pitches) == 0:
            return 0, 0

        pitch_mean = np.mean(pitches)

        pitch_slope = pitches[-1] - pitches[0]

        return pitch_mean, pitch_slope

    except:
        return 0, 0


def harmonic_energy(audio):

    harmonic, percussive = librosa.effects.hpss(audio)

    energy = np.mean(librosa.feature.rms(y=harmonic))

    return energy

# =========================
# Cause Detection
# =========================

def guess_possible_cause(audio):

    rms = float(np.mean(librosa.feature.rms(y=audio)))
    zcr = float(np.mean(librosa.feature.zero_crossing_rate(audio)))
    centroid = float(np.mean(librosa.feature.spectral_centroid(y=audio, sr=SR)))

    pitch_mean, pitch_slope = extract_pitch(audio)

    harm_energy = harmonic_energy(audio)

    # Hungry
    if rms > 0.05 and pitch_slope > 40:
        return "Hungry"

    # Discomfort
    if zcr > 0.12 and centroid > 2500:
        return "Discomfort"

    # Sleepy
    if rms < 0.02 and pitch_slope < -30:
        return "Sleepy"

    # Scared
    if centroid > 2700:
        return "Scared"

    # Tired
    if rms < 0.03 and pitch_slope < -10:
        return "Tired"

    # Lonely
    if rms < 0.025 and harm_energy < 0.02:
        return "Lonely"

    return "General Cry"

# =========================
# Asphyxia Detection
# =========================

def predict_from_full_audio(full_audio):

    window = MAX_LEN
    step = window
    energies = []

    for start in range(0, len(full_audio)-window+1, step):

        segment = full_audio[start:start+window]

        rms = float(np.mean(librosa.feature.rms(y=segment)))

        energies.append((rms, start))

    energies.sort(reverse=True)

    best_prob = 0
    best_segment = None

    for _, start in energies[:3]:

        segment = full_audio[start:start+window]

        features = extract_features(segment).reshape(1,-1)

        features_scaled = asphyxia_scaler.transform(features)

        probs = asphyxia_model.predict_proba(features_scaled)[0]

        classes = asphyxia_model.classes_

        asphyxia_index = list(classes).index(1)

        prob = float(probs[asphyxia_index])

        if prob > best_prob:
            best_prob = prob
            best_segment = segment

    return best_prob, best_segment

# =========================
# Routes
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
            return jsonify({"classification":"Invalid Input","confidence":0})

        file = request.files["file"]

        temp_original = os.path.join(TEMP_DIR, f"{uuid.uuid4().hex}.wav")
        file.save(temp_original)

        temp_path = os.path.join(TEMP_DIR, f"{uuid.uuid4().hex}_proc.wav")

        audio = AudioSegment.from_file(temp_original)
        audio = audio.set_channels(1).set_frame_rate(SR)
        audio.export(temp_path, format="wav")

        full_audio = load_audio(temp_path)

        # Reject silence
        if np.mean(librosa.feature.rms(y=full_audio)) < 0.003:

            return jsonify({
                "classification":"No Cry Detected",
                "confidence":0,
                "reliability":"Low"
            })

        # Stage 1 Asphyxia
        asphyxia_prob, best_segment = predict_from_full_audio(full_audio)

        if asphyxia_prob >= ASPHYXIA_THRESHOLD:

            reliability = "High" if asphyxia_prob > 0.6 else "Moderate"

            return jsonify({
                "classification":"Asphyxia Detected",
                "confidence":round(asphyxia_prob,3),
                "reliability":reliability
            })

        # Stage 2 Normal Cry
        features = extract_features(best_segment)

        csi_value = compute_csi(features)

        confidence_score = (csi_value - csi_min) / (csi_max - csi_min)

        confidence_score = max(0.0, min(1.0, confidence_score))

        if confidence_score > 0.75:
            reliability = "High"
        elif confidence_score > 0.45:
            reliability = "Moderate"
        else:
            reliability = "Low"

        possible_cause = guess_possible_cause(best_segment)

        return jsonify({
            "classification":"Baby Seems Okay",
            "possible_cause":possible_cause,
            "confidence":round(confidence_score,3),
            "reliability":reliability
        })

    except Exception as e:

        print(e)

        return jsonify({
            "classification":"Analysis Error",
            "confidence":0,
            "reliability":"Low"
        })

    finally:

        try:
            if temp_original and os.path.exists(temp_original):
                os.remove(temp_original)
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)
        except:
            pass


if __name__ == "__main__":
    port = int(os.environ.get("PORT",7860))
    app.run(host="0.0.0.0",port=port)