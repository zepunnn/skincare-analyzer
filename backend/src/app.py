import os
import glob
import numpy as np
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from dotenv import load_dotenv
from PIL import Image
import io

# ── Load .env ──────────────────────────────────────────────
load_dotenv()

# ── Arahkan Flask ke frontend/ ─────────────────────────────
# app.py ada di backend/src/app.py → naik 2x untuk sampai ke root proyek
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

# HTML ada di frontend/html/
template_dir = os.path.join(BASE_DIR, 'frontend')

# CSS & JS ada di frontend/css/ dan frontend/js/
static_dir = os.path.join(BASE_DIR, 'frontend')

app = Flask(
    __name__,
    template_folder=template_dir,
    static_folder=static_dir,
    static_url_path='',
)
CORS(app)

application = app

print("[DEBUG] template_folder:", app.template_folder)
print("[DEBUG] static_folder  :", app.static_folder)
print("[DEBUG] html files     :", glob.glob(app.template_folder + '/*.html'))
print("[DEBUG] css files      :", glob.glob(os.path.join(static_dir, 'css', '*.css')))
print("[DEBUG] js files       :", glob.glob(os.path.join(static_dir, 'js', '*.js')))

# ── Config ─────────────────────────────────────────────────
MODEL_PATH = os.getenv("MODEL_PATH", "model/model.h5")
IMG_SIZE   = (224, 224)

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

# ── Load model ──────────────────────────────────────────────
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

# ── Helper ──────────────────────────────────────────────────
def preprocess_image(file_bytes: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    img = img.resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)

def dummy_predict() -> tuple[int, float]:
    idx  = int(np.random.randint(0, 4))
    conf = float(np.random.uniform(0.70, 0.99))
    return idx, conf

# ── Routes — Halaman ─────────────────────────────────────────
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/index.html")
def home():
    return render_template("index.html")

@app.route("/analyze.html")
def analyze():
    return render_template("analyze.html")

@app.route("/about.html")
def about():
    return render_template("about.html")

# ── Routes — API ──────────────────────────────────────────────
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
            img_array  = preprocess_image(file_bytes)
            preds      = model.predict(img_array)[0]
            class_idx  = int(np.argmax(preds))
            confidence = float(np.max(preds))
        else:
            class_idx, confidence = dummy_predict()

        label          = CLASS_LABELS[class_idx]
        recommendation = RECOMMENDATIONS[label]

        return jsonify({
            "label":          label,
            "confidence":     round(confidence * 100, 2),
            "recommendation": recommendation,
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ── Entry point ─────────────────────────────────────────────
if __name__ == "__main__":
    load_model()
    debug_mode = os.getenv("FLASK_DEBUG", "true").lower() == "true"
    app.run(debug=debug_mode, host="0.0.0.0", port=5000)