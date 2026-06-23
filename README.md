# 🧴 SkinCare Analyzer

Aplikasi web berbasis Deep Learning untuk menganalisis kondisi kulit wajah melalui citra digital. Sistem mengklasifikasikan kondisi kulit (normal, jerawat ringan, sedang, berat) dan memberikan rekomendasi perawatan yang sesuai.

---

## 📋 Daftar Isi

- [Deskripsi](#-deskripsi)
- [Fitur](#-fitur)
- [Demo](#-demo)
- [Tech Stack](#-tech-stack)
- [Instalasi](#-instalasi)
- [Cara Penggunaan](#-cara-penggunaan)
- [Dataset](#-dataset)
- [Model](#-model)
- [Tim](#-tim)

---

## 📌 Deskripsi

SkinCare Analyzer adalah aplikasi berbasis web yang memanfaatkan teknologi Deep Learning untuk menganalisis kondisi kulit wajah. Pengguna cukup mengunggah foto wajah, dan sistem akan secara otomatis mendeteksi kondisi kulit serta memberikan rekomendasi perawatan dasar.

Proyek ini dibuat sebagai tugas kelompok mata kuliah [Nama Mata Kuliah] di [Nama Institusi].

---

## ✨ Fitur

- 📤 **Upload Gambar** — Unggah foto wajah langsung dari perangkat
- 🔍 **Preview Gambar** — Tampilkan gambar sebelum diproses
- 🤖 **Prediksi Kondisi Kulit** — Klasifikasi menggunakan model Deep Learning
- 📊 **Confidence Score** — Tingkat keyakinan model terhadap hasil prediksi
- 💊 **Rekomendasi Perawatan** — Saran perawatan berdasarkan kondisi yang terdeteksi

---

## 🚀 Demo

> 🔗 [Live Demo](https://skincare-analyzer.vercel.app) *(akan diupdate setelah deploy)*

---

## 🛠 Tech Stack

| Kategori | Teknologi |
|----------|-----------|
| **ML/AI** | Python, TensorFlow/Keras |
| **Computer Vision** | OpenCV |
| **Backend** | Flask / FastAPI |
| **Frontend** | HTML, CSS, JavaScript, Bootstrap/Tailwind |
| **Dataset** | Kaggle |
| **Deploy** | Vercel |
| **Version Control** | GitHub |

---

## ⚙️ Instalasi

### Prerequisites
- Python 3.8+
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

### Jalankan Aplikasi
```bash
# Jika menggunakan Flask
python backend/app.py

# Jika menggunakan FastAPI
uvicorn backend.app:app --reload
```

Buka browser dan akses `http://localhost:5000`

---

## 📖 Cara Penggunaan

1. Buka aplikasi di browser
2. Klik tombol **"Mulai Analisis"** di halaman utama
3. Upload foto wajah (format JPG/PNG)
4. Klik tombol **"Prediksi"**
5. Lihat hasil klasifikasi, confidence score, dan rekomendasi perawatan

---

## 📊 Dataset

Dataset yang digunakan bersumber dari [Kaggle](https://www.kaggle.com) dengan kategori kondisi kulit:

| Kelas | Deskripsi |
|-------|-----------|
| `normal` | Kulit normal tanpa jerawat signifikan |
| `acne_mild` | Jerawat ringan |
| `acne_moderate` | Jerawat sedang |
| `acne_severe` | Jerawat berat |

Pembagian dataset: **70% Train / 15% Validation / 15% Test**

> Sumber dataset: *(akan diupdate)*

---

## 🤖 Model

- **Arsitektur**: Transfer Learning (ResNet50)
- **Framework**: TensorFlow / Keras
- **Input Size**: 224 × 224 px
- **Output**: 4 kelas kondisi kulit + confidence score

> Link Model: https://drive.google.com/drive/folders/1uzvSas1s6o-lHTm0MeyYYMan-D87AjaB?usp=drive_link

> Performa model akan diupdate setelah training selesai.

---

## 👥 Tim

| Nama | NIM | Peran |
|------|-----|-------|
| [Nama Anggota 1] | [NIM] | Dataset Lead |
| [Nama Anggota 2] | [NIM] | ML Engineer |
| [Nama Anggota 3] | [NIM] | UI/UX Designer |
| [Nama Anggota 4] | [NIM] | Project Manager |

---

## 📄 Lisensi

Proyek ini dibuat untuk keperluan akademis. Seluruh hak cipta milik tim pengembang.

---

<p align="center">Made with ❤️ by Tim SkinCare Analyzer</p>s