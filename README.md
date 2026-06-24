# 🧴 SkinCare Analyzer

Aplikasi web berbasis Deep Learning untuk menganalisis kondisi kulit wajah melalui citra digital. Sistem mendeteksi **tingkat keparahan jerawat** dan **tipe kulit (oiliness)** secara bersamaan, lalu memberikan rekomendasi perawatan yang personal.

---

## 📋 Daftar Isi

- [Deskripsi](#-deskripsi)
- [Fitur](#-fitur)
- [Demo](#-demo)
- [Struktur Proyek](#-struktur-proyek)
- [Tech Stack](#-tech-stack)
- [Instalasi](#-instalasi)
- [Cara Penggunaan](#-cara-penggunaan)
- [Dataset](#-dataset)
- [Model](#-model)
- [API Endpoint](#-api-endpoint)
- [Deploy](#-deploy)
- [Tim](#-tim)

---

## 📌 Deskripsi

SkinCare Analyzer adalah aplikasi web yang memanfaatkan teknologi Deep Learning untuk menganalisis kondisi kulit wajah secara otomatis. Pengguna cukup mengunggah foto wajah, dan sistem akan menjalankan dua model secara paralel:

1. **Acne Detection** — mengklasifikasikan tingkat keparahan jerawat (Normal / Mild / Moderate / Severe)  
2. **Oiliness Detection** — mendeteksi tipe kulit (Dry / Oily)

Hasil analisis dikombinasikan untuk menghasilkan **rekomendasi perawatan yang disesuaikan**.

Proyek ini dibuat sebagai bagian dari **Magang IMV** di Universitas Telkom.

---

## ✨ Fitur

- 📤 **Upload Gambar** — drag & drop atau pilih file (JPG/PNG)
- 🔍 **Preview Gambar** — tampilkan gambar sebelum diproses
- 🤖 **Dual Model Inference** — analisis jerawat (ResNet50) + tipe kulit (MobileNetV2) secara bersamaan
- 📊 **Confidence Score** — tingkat keyakinan model untuk masing-masing prediksi
- 💊 **Rekomendasi Personal** — saran perawatan berdasarkan kombinasi hasil acne + oiliness
- 🚫 **Tanpa Registrasi** — langsung pakai, tidak perlu login
- 🌐 **Gratis & Aksesibel** — tersedia online via Vercel

---

## 🚀 Demo

> 🔗 [Live Demo](https://skincareanalyzer.vercel.app)

---

## 📁 Struktur Proyek

```
skincare-analyzer/
├── backend/
│   └── src/
│       └── app.py              # Flask app (routes + inference logic)
├── frontend/
│   ├── assets/                 # SVG ilustrasi & foto tim
│   ├── css/
│   │   └── style.css           # Shared stylesheet (semua halaman)
│   ├── html/
│   │   ├── index.html          # Landing page
│   │   ├── analyze.html        # Halaman analisis utama
│   │   └── about.html          # Info proyek & tim
│   └── js/
│       └── main.js
├── model/
│   ├── acne/
│   │   ├── preprocess.py       # Data generator (ResNet50)
│   │   ├── train.py            # Training script (warm-up + fine-tuning)
│   │   ├── predict.py          # Standalone inference
│   │   └── Model Evaluation.py # Evaluasi + confusion matrix
│   └── oiliness/
│       ├── preprocess.py       # Data generator (MobileNetV2)
│       ├── train.py            # Training script
│       ├── train_mobilenetv2.py
│       ├── predict.py
│       └── mobilenetv2_skin_type.tflite  # Model oiliness (sudah terkonversi)
├── notebooks/
│   └── explorations.ipynb
├── resnet50_acne_analyzer.tflite   # Model acne (sudah terkonversi)
├── convert_tflite.py               # Script konversi H5 → TFLite
├── requirements.txt
├── pyproject.toml
└── vercel.json
```

---

## 🛠 Tech Stack

| Kategori | Teknologi |
|----------|-----------|
| **ML/AI** | Python, TensorFlow / Keras |
| **Model Arsitektur** | ResNet50 (acne), MobileNetV2 (oiliness) |
| **Model Format** | TFLite (via `ai-edge-litert`) |
| **Backend** | Flask + Flask-CORS |
| **Frontend** | HTML, CSS, JavaScript, Tailwind CSS |
| **Font** | Outfit, Inter (Google Fonts) |
| **Dataset** | Kaggle (ACNE04, Skin Issues v2) |
| **Deploy** | Vercel |
| **Version Control** | GitHub |

---

## ⚙️ Instalasi

### Prerequisites

- Python 3.12+
- pip
- Git

### Clone Repository

```bash
git clone https://github.com/zepunnn/skincare-analyzer.git
cd skincare-analyzer
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### (Opsional) Konversi Model ke TFLite

Jika kamu sudah punya model `.h5` hasil training, konversi ke `.tflite` dengan:

```bash
python convert_tflite.py
```

File `.tflite` yang sudah jadi tersedia di Google Drive:  
📦 [Download Model](https://drive.google.com/drive/folders/1uzvSas1s6o-lHTm0MeyYYMan-D87AjaB?usp=drive_link)

Setelah download, tempatkan file di:
- `resnet50_acne_analyzer.tflite` → root folder
- `mobilenetv2_skin_type.tflite` → `model/oiliness/`

### Jalankan Aplikasi (Local)

```bash
python backend/src/app.py
```

Buka browser dan akses `http://localhost:5000`

> Jika file `.tflite` tidak ditemukan, backend otomatis berjalan dalam **dummy mode** (prediksi acak untuk keperluan testing UI).

---

## 📖 Cara Penggunaan

1. Buka aplikasi di browser
2. Klik **"ANALISIS SEKARANG"** di halaman utama
3. Upload atau drag & drop foto wajah (JPG/PNG)
4. Preview gambar akan tampil di panel kanan
5. Klik tombol **"Analisis"**
6. Lihat hasil:
   - Tingkat jerawat + confidence score
   - Tipe kulit (dry/oily) + confidence score
   - Rekomendasi perawatan personal

---

## 📊 Dataset

| Model | Dataset | Kelas |
|-------|---------|-------|
| **Acne** | ACNE Severity Classification (ACNE04) — IGA Scale 0–3 | Normal, Mild Acne, Moderate Acne, Severe Acne |
| **Oiliness** | Skin Issues Version 2 (Balanced Dataset) | Dry, Oil |

Pembagian dataset: **Train / Validation / Test**

---

## 🤖 Model

### Acne Detection — ResNet50

- **Arsitektur**: Transfer Learning ResNet50 (pretrained ImageNet)
- **Training**: 2 tahap — warm-up (base frozen) + fine-tuning (base unfrozen)
- **Optimizer**: Adam (lr 1e-4 → 1e-5)
- **Loss**: Categorical Crossentropy
- **Input**: 224 × 224 px, preprocessing Caffe-style (zero-center BGR)
- **Output**: 4 kelas (Normal / Mild / Moderate / Severe) + confidence

### Oiliness Detection — MobileNetV2

- **Arsitektur**: Transfer Learning MobileNetV2 (pretrained ImageNet)
- **Input**: 224 × 224 px, preprocessing TF-style (normalisasi -1 hingga 1)
- **Output**: 2 kelas (Dry / Oil) + confidence
- **Format deploy**: TFLite (`.tflite`) via `ai-edge-litert`

---

## 🔌 API Endpoint

### `POST /predict`

Endpoint utama untuk inference.

**Request:**
```
Content-Type: multipart/form-data
Body: file=<image file>
```

**Response (sukses):**
```json
{
  "jerawat": "rendah",
  "berminyak": "tinggi",
  "acne_confidence": 87.3,
  "oil_confidence": 92.1,
  "recommendations": [
    { "icon": "water", "text": "Gunakan pembersih wajah berbusa (foaming cleanser)..." },
    { "icon": "sun",   "text": "Gunakan spot treatment (Salicylic Acid / Benzoyl Peroxide)..." },
    { "icon": "sun",   "text": "Pilih sunscreen dengan label 'non-comedogenic'..." }
  ],
  "raw_labels": {
    "acne": "Mild Acne",
    "skin_type": "Oil"
  }
}
```

| Field `jerawat` | Keterangan |
|-----------------|------------|
| `normal` | Tidak ada jerawat signifikan |
| `rendah` | Mild Acne |
| `sedang` | Moderate Acne |
| `tinggi` | Severe Acne |

---

## ☁️ Deploy

Proyek ini di-deploy ke **Vercel** menggunakan konfigurasi `vercel.json`:

- Static files (`/assets`, `/css`, `/js`) → `@vercel/static`
- Backend (`backend/src/app.py`) → `@vercel/python`

Semua route yang tidak cocok dialihkan ke Flask app.

---

## 👥 Tim

| Nama | Jurusan | Quote |
|------|---------|-------|
| **Fakhri Muhammad Al Hisyam** | Teknik Telekomunikasi | *"If you find the one, hold on and fight for it like it's your only truth."* |
| **Muhammad Hilman Dzakwanurrofiq** | Teknik Fisika | *"Don't judge website by the frontend."* |
| **Muhammad Azriel Saputra** | Teknik Biomedis | *"Nikmati akses konten Platinum di Aplikasi Vidio..."* |
| **Muhammad Fauzan Alviansyah** | Teknik Elektro | *"Keep calm and blame it on the lag."* |

---

## 📄 Lisensi

Proyek ini dibuat untuk keperluan akademis (Magang IMV). Seluruh hak cipta milik tim pengembang.

---

<p align="center">Made with ❤️ by Tim SkinCare Analyzer — Magang IMV © 2026</p>