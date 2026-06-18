import os
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from PIL import Image
import io

# ── Load .env ──────────────────────────────────────────────
load_dotenv()

app = Flask(__name__)
CORS(app)

# ── Config ─────────────────────────────────────────────────
MODEL_PATH = os.getenv("MODEL_PATH", "model/model.h5")
IMG_SIZE   = (224, 224)   # MobileNetV2 default

CLASS_LABELS = {
    0: "Normal",
    1: "Mild Acne",
    2: "Moderate Acne",
    3: "Severe Acne",
}

RECOMMENDATIONS = {
    "Normal": (
        "Kulit kamu terlihat sehat! Pertahankan dengan rutin membersihkan "
        "wajah 2x sehari, gunakan moisturizer, dan selalu pakai sunscreen SPF 30+."
    ),
    "Mild Acne": (
        "Gunakan face wash dengan kandungan salicylic acid, hindari memencet jerawat, "
        "dan gunakan spot treatment berbahan benzoyl peroxide atau tea tree oil."
    ),
    "Moderate Acne": (
        "Disarankan menggunakan produk dengan niacinamide atau retinol. "
        "Konsultasikan dengan dokter kulit untuk penanganan lebih lanjut."
    ),
    "Severe Acne": (
        "Segera konsultasikan ke dokter kulit (dermatologis). "
        "Jangan mencoba mengobati sendiri karena dapat meninggalkan bekas permanen."
    ),
}

# ── Load model (opsional, skip jika belum ada) ──────────────
model = None

def load_model():
    global model
    try:
        from tensorflow.keras.models import load_model as keras_load
        model = keras_load(MODEL_PATH)
        print(f"[INFO] Model loaded from {MODEL_PATH}")
    except Exception as e:
        print(f"[WARN] Model tidak ditemukan atau gagal dimuat: {e}")
        print("[WARN] Berjalan dalam mode dummy.")

# ── Helper: preprocess gambar ───────────────────────────────
def preprocess_image(file_bytes: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    img = img.resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)   # shape: (1, 224, 224, 3)

# ── Helper: dummy prediction (sebelum model tersedia) ───────
def dummy_predict() -> tuple[int, float]:
    idx   = int(np.random.randint(0, 4))
    conf  = float(np.random.uniform(0.70, 0.99))
    return idx, conf

# ── Routes ──────────────────────────────────────────────────
@app.route("/", methods=["GET"])
def index():
    return jsonify({"message": "SkinCare Analyzer API is running."})

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "Tidak ada file yang dikirim."}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "File kosong."}), 400

    try:
        file_bytes = file.read()

        if model is not None:
            # ── Real prediction ──
            img_array  = preprocess_image(file_bytes)
            preds      = model.predict(img_array)[0]       # shape: (4,)
            class_idx  = int(np.argmax(preds))
            confidence = float(np.max(preds))
        else:
            # ── Dummy prediction ──
            class_idx, confidence = dummy_predict()

        label          = CLASS_LABELS[class_idx]
        recommendation = RECOMMENDATIONS[label]

        return jsonify({
            "label":          label,
            "confidence":     round(confidence * 100, 2),   # dalam persen
            "recommendation": recommendation,
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ── Entry point ─────────────────────────────────────────────
if __name__ == "__main__":
    load_model()
    debug_mode = os.getenv("FLASK_DEBUG", "true").lower() == "true"
    app.run(debug=debug_mode, host="0.0.0.0", port=5000)