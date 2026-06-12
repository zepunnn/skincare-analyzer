# 📦 Dataset — SkinCare Analyzer

Branch ini berisi proses pengumpulan, preprocessing, dan augmentasi dataset untuk model klasifikasi kondisi kulit wajah.

---

## 📥 Sumber Dataset

| Info | Detail |
|------|--------|
| **Nama Dataset** | *(update setelah download)* |
| **Sumber** | [Kaggle](https://www.kaggle.com) |
| **Link** | *(update setelah download)* |
| **Lisensi** | *(update setelah download)* |
| **Jumlah Gambar** | *(update setelah download)* |

---

## 🏷 Kategori Kelas

| Kelas | Label | Deskripsi |
|-------|-------|-----------|
| Normal | `normal` | Kulit bersih tanpa jerawat signifikan |
| Jerawat Ringan | `acne_mild` | Komedo dan jerawat kecil, jumlah sedikit |
| Jerawat Sedang | `acne_moderate` | Jerawat lebih banyak, mulai ada peradangan |
| Jerawat Berat | `acne_severe` | Jerawat meradang parah, merata di wajah |

Pembagian dataset: **70% Train / 15% Validation / 15% Test**

---

## ⚙️ Preprocessing

Langkah preprocessing yang diterapkan pada setiap gambar:

1. **Resize** — Semua gambar diubah ke ukuran `224 × 224` px
2. **Normalisasi** — Nilai pixel dibagi 255 menjadi rentang `[0, 1]`
3. **Konversi warna** — Pastikan semua gambar dalam format RGB
4. **Filter gambar rusak** — Hapus file yang tidak valid atau corrupt

```python
import cv2
import numpy as np

def preprocess_image(img_path, target_size=(224, 224)):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, target_size)
    img = img / 255.0
    return img
```

---

## 🔀 Pembagian Dataset

```python
from sklearn.model_selection import train_test_split

# Split train / temp
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)

# Split val / test dari temp
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
)
```

---

## 🔁 Augmentasi

Augmentasi diterapkan **hanya pada data train** untuk memperbanyak variasi data dan mencegah overfitting.

| Teknik | Parameter |
|--------|-----------|
| Horizontal Flip | `p=0.5` |
| Rotasi | `±15°` |
| Zoom | `±10%` |
| Brightness | `±20%` |
| Shift (H/W) | `±10%` |

```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    rotation_range=15,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1,
    brightness_range=[0.8, 1.2]
)

val_test_datagen = ImageDataGenerator(rescale=1./255)
```

---

## 📦 Dependencies
tensorflow>=2.10.0
opencv-python
numpy
scikit-learn
matplotlib
Pillow

---

Install semua dependencies:

```bash
pip install -r requirements.txt
```

---

## ✅ Progress Checklist

- [ ] Dataset ditemukan dan didownload dari Kaggle
- [ ] Kategori kelas ditentukan (normal, mild, moderate, severe)
- [ ] Preprocessing selesai (resize, normalisasi, filter rusak)
- [ ] Dataset dibagi train/val/test (70/15/15)
- [ ] Augmentasi diterapkan pada data train
- [ ] Jumlah data per kelas didokumentasikan

---

> **Catatan:** Folder `raw/` dan `processed/` tidak di-push ke GitHub karena ukuran file besar.
> Tambahkan ke `.gitignore`:
> ```
> dataset/raw/
> dataset/processed/
> ```