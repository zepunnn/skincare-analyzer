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
# Menggunakan format .tflite agar ringan di Vercel (tidak butuh TF)
ACNE_MODEL_PATH = os.path.join(BASE_DIR, "resnet50_acne_analyzer.tflite")
SKIN_MODEL_PATH = os.path.join(BASE_DIR, "model", "oiliness", "mobilenetv2_skin_type.tflite")

IMG_SIZE = (224, 224)

ACNE_CLASSES = ["Normal", "Mild Acne", "Moderate Acne", "Severe Acne"]
SKIN_CLASSES = ["Dry", "Oil"]

# ── Rekomendasi ─────────────────────────────────────────────
def get_recommendations(acne_class, skin_class):
    recs = []
    
    # 1. Pembersih Wajah
    if skin_class == "Oil":
        recs.append({"icon": "water", "text": "Gunakan pembersih wajah berbusa (foaming cleanser) untuk mengontrol minyak berlebih."})
    else: # Dry
        recs.append({"icon": "water", "text": "Gunakan pembersih wajah lembut yang menghidrasi (gentle/hydrating cleanser)."})
        
    # 2. Perawatan Jerawat
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

# ── Load model (TFLite) ─────────────────────────────────────
acne_interpreter = None
skin_interpreter = None

def load_models():
    global acne_interpreter, skin_interpreter
    try:
        from ai_edge_litert import interpreter as tflite
        
        # Load Acne Model
        if os.path.exists(ACNE_MODEL_PATH):
            acne_interpreter = tflite.Interpreter(model_path=ACNE_MODEL_PATH)
            acne_interpreter.allocate_tensors()
            print(f"[INFO] Acne TFLite Model loaded from {ACNE_MODEL_PATH}")
            
        # Load Skin Model
        if os.path.exists(SKIN_MODEL_PATH):
            skin_interpreter = tflite.Interpreter(model_path=SKIN_MODEL_PATH)
            skin_interpreter.allocate_tensors()
            print(f"[INFO] Skin TFLite Model loaded from {SKIN_MODEL_PATH}")
            
    except Exception as e:
        print(f"[WARN] Error loading models: {e}")
        print("[WARN] Running in dummy mode.")

# ── Preprocessing Manual ────────────────────────────────────
# Karena kita tidak menggunakan keras, kita harus memproses gambar secara manual
def preprocess_resnet50_caffe(img_array):
    # ResNet50 Caffe style: RGB -> BGR, then zero-center with ImageNet mean
    # Input is numpy array RGB [0, 255]
    img = img_array.copy()[..., ::-1] # RGB to BGR
    # ImageNet mean (BGR)
    mean = [103.939, 116.779, 123.68]
    img[..., 0] -= mean[0]
    img[..., 1] -= mean[1]
    img[..., 2] -= mean[2]
    return img

def preprocess_mobilenetv2_tf(img_array):
    # MobileNetV2 TF style: normalize between -1 and 1
    # Input is numpy array RGB [0, 255]
    img = img_array.copy()
    img = (img / 127.5) - 1.0
    return img

def dummy_predict():
    a_idx = int(np.random.randint(0, 4))
    a_conf = float(np.random.uniform(0.70, 0.99))
    s_idx = int(np.random.randint(0, 2))
    s_conf = float(np.random.uniform(0.70, 0.99))
    return a_idx, a_conf, s_idx, s_conf

def run_tflite_inference(interpreter, input_tensor):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    interpreter.set_tensor(input_details[0]['index'], input_tensor)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return output_data[0]

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

        if acne_interpreter is not None and skin_interpreter is not None:
            # Base Image Array [0, 255]
            img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
            img = img.resize(IMG_SIZE)
            img_base = np.array(img, dtype=np.float32)
            img_batch = np.expand_dims(img_base, axis=0)
            
            # Predict Acne (ResNet50 Caffe Preprocess)
            acne_input = preprocess_resnet50_caffe(img_batch)
            acne_preds = run_tflite_inference(acne_interpreter, acne_input)
            acne_idx = int(np.argmax(acne_preds))
            acne_conf = float(np.max(acne_preds))
            
            # Predict Skin Type (MobileNetV2 TF Preprocess)
            skin_input = preprocess_mobilenetv2_tf(img_batch)
            skin_preds = run_tflite_inference(skin_interpreter, skin_input)
            skin_idx = int(np.argmax(skin_preds))
            skin_conf = float(np.max(skin_preds))
            
        else:
            acne_idx, acne_conf, skin_idx, skin_conf = dummy_predict()

        acne_label = ACNE_CLASSES[acne_idx]
        skin_label = SKIN_CLASSES[skin_idx]
        
        # Mapping ke format frontend (4 level sesuai class detection)
        jerawat_map = {
            "Normal": "normal",
            "Mild Acne": "rendah",
            "Moderate Acne": "sedang",
            "Severe Acne": "tinggi",
        }
        jerawat_level = jerawat_map.get(acne_label, "normal")
        
        berminyak_level = "tinggi" if skin_label == "Oil" else "rendah"
        
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

# ── Load models on import (needed for Vercel) ───────────────
load_models()

# ── Entry point (local development only) ─────────────────────
if __name__ == "__main__":
    debug_mode = os.getenv("FLASK_DEBUG", "true").lower() == "true"
    app.run(debug=debug_mode, host="0.0.0.0", port=5000)
