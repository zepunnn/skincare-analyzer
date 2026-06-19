"""
preprocess.py
Menyiapkan dataset Oiliness
Struktur awal:
Oily-Dry-Skin-Types/
├── train/
│   ├── dry/
│   ├── normal/
│   └── oily/
├── valid/
│   ├── dry/
│   ├── normal/
│   └── oily/
└── test/
    ├── dry/
    ├── normal/
    └── oily/
"""

import os
import shutil
from collections import defaultdict
from keras.src.legacy.preprocessing.image import ImageDataGenerator
from keras.applications.resnet50 import preprocess_input

# ─── KONFIGURASI ────────────────────────────────────────────────────────────
RAW_DATASET_DIR = "Oily-Dry-Skin-Types"
PROCESSED_DIR   = "dataset/oiliness/processed"
IMG_SIZE        = (224, 224)
BATCH_SIZE      = 32

# Nama folder asli di dataset (huruf kecil)
RAW_CLASS_NAMES = ["dry", "normal", "oily"]

# Nama kelas final yang dipakai di processed/ (kapital, konsisten dengan acne)
CLASS_NAMES = ["Dry", "Normal", "Oily"]
CLASS_MAP   = dict(zip(RAW_CLASS_NAMES, CLASS_NAMES))


# ─── BUILD PROCESSED DATASET ────────────────────────────────────────────────
def build_processed_dataset():
    """
    Membaca dataset dari Oily-Dry-Skin-Types/{split}/{kelas}/
    dan menyalin ke dataset/oiliness/processed/{split}/{Kelas}/
    Folder tujuan dibuat otomatis kalau belum ada.
    """
    stats = defaultdict(lambda: defaultdict(int))

    for split in ("train", "valid", "test"):
        for raw_cls, cls in CLASS_MAP.items():
            src_dir = os.path.join(RAW_DATASET_DIR, split, raw_cls)
            dst_dir = os.path.join(PROCESSED_DIR, split, cls)

            # Auto-create folder tujuan jika belum ada
            os.makedirs(dst_dir, exist_ok=True)

            if not os.path.exists(src_dir):
                print(f"⚠️  Folder tidak ditemukan: {src_dir}")
                continue

            for file in os.listdir(src_dir):
                src = os.path.join(src_dir, file)
                if not os.path.isfile(src):
                    continue
                dst = os.path.join(dst_dir, file)
                shutil.copy2(src, dst)
                stats[split][cls] += 1

    print("\n✅ Dataset Oiliness berhasil disusun!\n")
    print(f"{'Class':<15}{'Train':>10}{'Valid':>10}{'Test':>10}")
    print("-" * 45)
    for cls in CLASS_NAMES:
        print(
            f"{cls:<15}"
            f"{stats['train'][cls]:>10}"
            f"{stats['valid'][cls]:>10}"
            f"{stats['test'][cls]:>10}"
        )

    total_files = sum(
        stats[split][cls]
        for split in ("train", "valid", "test")
        for cls in CLASS_NAMES
    )
    if total_files == 0:
        print(
            "\n⚠️  PERINGATAN: 0 file ter-copy. "
            "Cek apakah folder 'Oily-Dry-Skin-Types' ada di root proyek "
            "dan strukturnya sesuai (train/valid/test > dry/normal/oily)."
        )


# ─── DATA GENERATOR ──────────────────────────────────────────────────────────
def get_generators(processed_dir=PROCESSED_DIR,
                   img_size=IMG_SIZE, batch_size=BATCH_SIZE):
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        zoom_range=0.15,
        brightness_range=[0.8, 1.2],
        shear_range=0.1,
    )
    val_test_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input
    )

    train_gen = train_datagen.flow_from_directory(
        os.path.join(processed_dir, "train"),
        target_size=img_size,
        batch_size=batch_size,
        class_mode="categorical",
        classes=CLASS_NAMES,
        shuffle=True,
    )
    val_gen = val_test_datagen.flow_from_directory(
        os.path.join(processed_dir, "valid"),
        target_size=img_size,
        batch_size=batch_size,
        class_mode="categorical",
        classes=CLASS_NAMES,
        shuffle=False,
    )
    test_gen = val_test_datagen.flow_from_directory(
        os.path.join(processed_dir, "test"),
        target_size=img_size,
        batch_size=batch_size,
        class_mode="categorical",
        classes=CLASS_NAMES,
        shuffle=False,
    )
    return train_gen, val_gen, test_gen


# ─── MAIN ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    build_processed_dataset()
    print("\n📦 Mengecek generator...")
    train_gen, val_gen, test_gen = get_generators()
    print(f"Train batches : {len(train_gen)}")
    print(f"Valid batches : {len(val_gen)}")
    print(f"Test batches  : {len(test_gen)}")
    print(f"Class indices : {train_gen.class_indices}")