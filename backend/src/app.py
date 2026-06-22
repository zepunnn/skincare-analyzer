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
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

template_dir = os.path.join(BASE_DIR, 'frontend', 'html')
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

# ── Config ─────────────────────────────────────────────────
ACNE_MODEL_PATH = os.path.join(BASE_DIR, "resnet50_acne_analyzer.h5")
SKIN_MODEL_PATH_MB = os.path.join(BASE_DIR, "model", "oiliness", "mobilenetv2_skin_type.h5")
SKIN_MODEL_PATH_RN = os.path.join(BASE_DIR, "model", "oiliness", "resnet50_skin_type.h5")

IMG_SIZE = (224, 224)

ACNE_CLASSES = ["Normal", "Mild Acne", "Moderate Acne", "Severe Acne"]
SKIN_CLASSES = ["Dry", "Oil"]

# ── Rekomendasi ─────────────────────────────────────────────
# Menggabungkan tipe kulit dan kondisi jerawat
def get_recommendations(acne_class, skin_class):
    recs = []
    
    # 1. Pembersih Wajah (Berdasarkan Skin Type)
    if skin_class == "Oil":
        recs.append({"icon": "water", "text": "Gunakan pembersih wajah berbusa (foaming cleanser) untuk mengontrol minyak berlebih."})
    else: # Dry
        recs.append({"icon": "water", "text": "Gunakan pembersih wajah lembut yang menghidrasi (gentle/hydrating cleanser)."})
        
    # 2. Perawatan Jerawat (Berdasarkan Acne Severity)
    if acne_class == "Normal":
        recs.append({"icon": "sun", "text": "Kulit terlihat sehat. Pertahankan rutinitas dengan moisturizer dan sunscreen."})
    elif acne_class == "Mild Acne":
        recs.append({"icon": "sun", "text": "Gunakan spot treatment (seperti Salicylic Acid atau Benzoyl Peroxide) pada jerawat."})
    elif acne_class == "Moderate Acne":
        recs.append({"icon": "doctor", "text": "Gunakan produk dengan Niacinamide/Retinol. Mulai pertimbangkan konsultasi."})
    else: # Severe Acne
        recs.append({"icon": "doctor", "text": "Segera konsultasi ke dokter kulit (dermatologis). Hindari memencet jerawat."})
        
    # 3. Sunscreen / Extra
    if skin_class == "Oil":
        recs.append({"icon": "sun", "text": "Pilih sunscreen dengan label 'non-comedogenic' atau hasil akhir matte."})
    else:
        recs.append({"icon": "water", "text": "Gunakan pelembap yang kaya akan Ceramide/Hyaluronic Acid sebelum sunscreen."})
        
    return recs

# ── Load model ──────────────────────────────────────────────
acne_model = None
skin_model = None
skin_preprocess_fn = None

def load_models():
    global acne_model, skin_model, skin_preprocess_fn
    try:
        from tensorflow.keras.models import load_model as keras_load
        from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_prep
        from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_prep
        
        # Load Acne Model
        if os.path.exists(ACNE_MODEL_PATH):
            acne_model = keras_load(ACNE_MODEL_PATH, compile=False)
            print(f"[INFO] Acne Model loaded from {ACNE_MODEL_PATH}")
            
        # Load Skin Model (Prioritize MobileNetV2)
        if os.path.exists(SKIN_MODEL_PATH_MB):
            skin_model = keras_load(SKIN_MODEL_PATH_MB, compile=False)
            skin_preprocess_fn = mobilenet_prep
            print(f"[INFO] Skin Model loaded (MobileNetV2)")
        elif os.path.exists(SKIN_MODEL_PATH_RN):
            skin_model = keras_load(SKIN_MODEL_PATH_RN, compile=False)
            skin_preprocess_fn = resnet_prep
            print(f"[INFO] Skin Model loaded (ResNet50)")
            
    except Exception as e:
        print(f"[WARN] Error loading models: {e}")
        print("[WARN] Running in dummy mode.")

# ── Helper ──────────────────────────────────────────────────
def preprocess_base(file_bytes: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    img = img.resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32)
    return np.expand_dims(arr, axis=0)

def dummy_predict():
    a_idx = int(np.random.randint(0, 4))
    a_conf = float(np.random.uniform(0.70, 0.99))
    s_idx = int(np.random.randint(0, 2))
    s_conf = float(np.random.uniform(0.70, 0.99))
    return a_idx, a_conf, s_idx, s_conf

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

        if acne_model is not None and skin_model is not None:
            # Preprocess
            img_base = preprocess_base(file_bytes)
            
            # Predict Acne (ResNet50)
            from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_prep
            img_acne = resnet_prep(img_base.copy())
            acne_preds = acne_model.predict(img_acne, verbose=0)
            if isinstance(acne_preds, dict): acne_preds = list(acne_preds.values())[0]
            acne_preds = acne_preds[0]
            acne_idx = int(np.argmax(acne_preds))
            acne_conf = float(np.max(acne_preds))
            
            # Predict Skin Type
            img_skin = skin_preprocess_fn(img_base.copy())
            skin_preds = skin_model.predict(img_skin, verbose=0)
            if isinstance(skin_preds, dict): skin_preds = list(skin_preds.values())[0]
            skin_preds = skin_preds[0]
            skin_idx = int(np.argmax(skin_preds))
            skin_conf = float(np.max(skin_preds))
            
        else:
            acne_idx, acne_conf, skin_idx, skin_conf = dummy_predict()

        acne_label = ACNE_CLASSES[acne_idx]
        skin_label = SKIN_CLASSES[skin_idx]
        
        # Mapping ke format yang diharapkan frontend
        # Jerawat: Normal -> rendah, Mild -> sedang, Moderate/Severe -> tinggi
        jerawat_level = "rendah"
        if "Mild" in acne_label: jerawat_level = "sedang"
        elif "Moderate" in acne_label or "Severe" in acne_label: jerawat_level = "tinggi"
        
        # Berminyak: Dry -> rendah, Oil -> tinggi
        berminyak_level = "tinggi" if skin_label == "Oil" else "rendah"
        
        # Recommendations
        recs = get_recommendations(acne_label, skin_label)

        return jsonify({
            "jerawat": jerawat_level,
            "berminyak": berminyak_level,
            "acne_confidence": round(acne_conf * 100, 1),
            "oil_confidence": round(skin_conf * 100, 1),
            "recommendations": recs,
            "raw_labels": {
                "acne": acne_label,
                "skin_type": skin_label
            }
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

# ── Entry point ─────────────────────────────────────────────
if __name__ == "__main__":
    load_models()
    debug_mode = os.getenv("FLASK_DEBUG", "true").lower() == "true"
    app.run(debug=debug_mode, host="0.0.0.0", port=5000)