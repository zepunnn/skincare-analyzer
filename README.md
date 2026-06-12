# 📦 Dataset — SkinCare Analyzer

Branch ini berisi proses pengumpulan, preprocessing, dan augmentasi dataset untuk model klasifikasi kondisi kulit wajah.

---

## 📥 Sumber Dataset

| Info | Detail |
|------|--------|
| **Nama Dataset** | ACNE Severity Classification |
| **Sumber** | [Kaggle](https://www.kaggle.com) |
| **Link** | https://www.kaggle.com/datasets/lexuanhieu131297/acne-severity-classification/data?select=Classification |
| **Jumlah Gambar** | 1457 gambar (JPEGImages) |
| **Format Label** | File `.txt` per split (NNEW_trainval_*.txt, NNEW_test_*.txt) |

---

## 🏷 Kategori Kelas

Dataset menggunakan skala IGA (Investigator's Global Assessment) dengan label 0–3, dipetakan ke 4 kelas:

| Label Asli | Kelas | Deskripsi |
|-----------|-------|-----------|
| `0` | `Normal` | Kulit bersih tanpa jerawat signifikan |
| `1` | `Mild_Acne` | Komedo dan jerawat kecil, jumlah sedikit |
| `2` | `Moderate_Acne` | Jerawat lebih banyak, mulai ada peradangan |
| `3` | `Severe_Acne` | Jerawat meradang parah, merata di wajah |

---

## 📊 Distribusi Dataset

Pembagian dataset: **85% Train+Val / 15% Test** (test set sudah dipisah oleh pembuat dataset)

| Kelas | Train | Val | Test |
|-------|-------|-----|------|
| Normal | 429 | 84 | 513 |
| Mild_Acne | 551 | 82 | 633 |
| Moderate_Acne | 148 | 34 | 180 |
| Severe_Acne | 111 | 18 | 129 |
| **Total** | **1239** | **218** | **1455** |

> Catatan: Split train/val dilakukan 85/15 dari data trainval secara otomatis oleh script.

---

## ⚙️ Preprocessing

Langkah preprocessing yang diterapkan pada setiap gambar:

1. **Baca label** — Label dibaca dari file `.txt` (format: `nama_file.jpg label`)
2. **Mapping label** — Label IGA 0–3 dipetakan ke 4 kelas (Normal, Mild, Moderate, Severe)
3. **De-duplikasi** — File yang muncul di lebih dari satu txt hanya disalin sekali
4. **Split otomatis** — Data trainval diacak lalu dibagi 85% train / 15% val
5. **Copy ke folder kelas** — Gambar disalin ke `dataset/processed/{split}/{kelas}/`
6. **Resize** — Semua gambar diubah ke ukuran `224 × 224` px saat loading generator
7. **Normalisasi** — Nilai pixel dibagi 255 menjadi rentang `[0, 1]`

```python
# Mapping label IGA → 4 kelas
LABEL_MAP = {
    0: "Normal",
    1: "Mild_Acne",
    2: "Moderate_Acne",
    3: "Severe_Acne",
}

def parse_txt(txt_path):
    entries = []
    with open(txt_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            filename = parts[0]
            label    = int(parts[1])
            cls      = LABEL_MAP[label]
            entries.append((filename, cls))
    return entries
```

---

## 🔀 Pembagian Dataset

File `.txt` dari dataset dibagi menjadi dua kelompok:
- `NNEW_trainval_*.txt` → digabung, de-duplikasi, lalu split 85% train / 15% val
- `NNEW_test_*.txt` → langsung dipakai sebagai test set

```python
# Baca semua trainval otomatis
for txt in sorted(glob.glob("Classification/NNEW_trainval_*.txt")):
    trainval_entries += parse_txt(txt)

# De-duplikasi
seen = set()
unique_trainval = []
for fname, cls in trainval_entries:
    if fname not in seen:
        seen.add(fname)
        unique_trainval.append((fname, cls))

# Split train/val (85/15)
rng = np.random.default_rng(RANDOM_SEED)
indices = rng.permutation(len(unique_trainval))
val_count = int(len(unique_trainval) * val_ratio)  # 15%
val_idx   = set(indices[:val_count])
```

---

## 🔁 Augmentasi

Augmentasi diterapkan **hanya pada data train** untuk memperbanyak variasi data dan mencegah overfitting.

| Teknik | Parameter |
|--------|-----------|
| Horizontal Flip | `True` |
| Rotasi | `±20°` |
| Zoom | `±15%` |
| Brightness | `[0.8, 1.2]` |
| Width Shift | `±10%` |
| Height Shift | `±10%` |
| Shear | `±10%` |

```python
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    zoom_range=0.15,
    brightness_range=[0.8, 1.2],
    shear_range=0.1,
)

# Val & test hanya rescale, tanpa augmentasi
val_test_datagen = ImageDataGenerator(rescale=1.0 / 255)
```

---

## 📦 Dependencies
tensorflow>=2.15.0
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

## 🚀 Cara Menjalankan

```bash
cd skincare-analyzer

python model/preprocess.py
```

Output yang diharapkan:
✅ Dataset berhasil disusun!

Kelas                   Train      Val     Test
Normal                    429       84      513
Mild_Acne                 551       82      633
Moderate_Acne             148       34      180
Severe_Acne               111       18      129

---

## ✅ Progress Checklist

- [x] Dataset ditemukan dan didownload dari Kaggle
- [x] Kategori kelas ditentukan (Normal, Mild, Moderate, Severe)
- [x] Label dibaca dari file `.txt` dan dipetakan ke 4 kelas
- [x] De-duplikasi data dari multiple file txt
- [x] Dataset dibagi train/val/test secara otomatis
- [x] Augmentasi diterapkan pada data train
- [x] Jumlah data per kelas terdokumentasi

---

> **Catatan:** Folder `Classification/` dan `dataset/processed/` tidak di-push ke GitHub karena ukuran file besar.
> Sudah ditambahkan ke `.gitignore`:
> ```
> dataset/raw/
> dataset/processed/
> Classification/
> ```