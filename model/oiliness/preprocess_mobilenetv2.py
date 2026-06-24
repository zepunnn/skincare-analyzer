"""
preprocess_mobilenetv2.py
Menyiapkan dataset dari skin_dataset/ untuk training MobileNetV2.
Menggunakan preprocess_input dari MobileNetV2 dan augmentasi agresif.

Struktur dataset (skin_dataset/):
├── train/
│   ├── acne/
│   ├── dry/
│   └── oil/
└── test/
    ├── acne/
    ├── dry/
    └── oil/

Karena tidak ada folder valid/, data train akan di-split 80/20 menggunakan
validation_split pada ImageDataGenerator (sesuai pendekatan notebook referensi).
"""

import os
import numpy as np
from keras.src.legacy.preprocessing.image import ImageDataGenerator
from keras.applications.mobilenet_v2 import preprocess_input

# ─── KONFIGURASI ────────────────────────────────────────────────────────────
IMG_SIZE    = (224, 224)
BATCH_SIZE  = 64           # Batch size 64 (sesuai notebook referensi)

# Nama kelas: hanya dry dan oil
CLASS_NAMES = ["dry", "oil"]


# ─── DATA GENERATOR (MobileNetV2) ───────────────────────────────────────────
def get_generators(dataset_dir, img_size=IMG_SIZE, batch_size=BATCH_SIZE):
    """
    Membuat data generator dengan augmentasi agresif sesuai notebook MobileNetV2.
    Karena tidak ada folder valid/, menggunakan validation_split=0.2 pada data train.

    Args:
        dataset_dir: Path ke folder skin_dataset/ (berisi train/ dan test/)
        img_size: Ukuran gambar target
        batch_size: Batch size

    Returns:
        train_gen, val_gen, test_gen
    """
    train_dir = os.path.join(dataset_dir, "train")
    test_dir  = os.path.join(dataset_dir, "test")

    # Data augmentation untuk training (sesuai notebook MobileNetV2)
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        validation_split=0.2,         # 80% train, 20% validation
        rotation_range=360,           # Rotasi penuh 360° (dari notebook)
        width_shift_range=0.1,        # Geser horizontal
        height_shift_range=0.1,       # Geser vertikal
        horizontal_flip=True,         # Flip horizontal
        zoom_range=0.15,              # Zoom in sampai 15%
        brightness_range=[0.5, 1.5],  # Brightness 50%-150% (dari notebook)
        shear_range=0.15,             # Shear transformation
        fill_mode="nearest",          # Isi piksel baru dengan nearest
    )

    # Hanya preprocessing untuk validasi dan test (tanpa augmentasi)
    test_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input
    )

    # Training generator (80% dari folder train/)
    train_gen = train_datagen.flow_from_directory(
        train_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode="categorical",
        classes=CLASS_NAMES,
        shuffle=True,
        subset="training",
    )

    # Validation generator (20% dari folder train/)
    val_gen = train_datagen.flow_from_directory(
        train_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode="categorical",
        classes=CLASS_NAMES,
        shuffle=False,
        subset="validation",
    )

    # Test generator (dari folder test/)
    test_gen = test_datagen.flow_from_directory(
        test_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode="categorical",
        classes=CLASS_NAMES,
        shuffle=False,
    )

    return train_gen, val_gen, test_gen


# ─── MAIN ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Default: gunakan skin_dataset di root proyek
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    DATASET_DIR = os.path.join(BASE_DIR, 'skin_dataset')

    print("=" * 55)
    print("  PREPROCESS DATA - MobileNetV2 Skin Classification")
    print("=" * 55)
    print(f"\nDataset : {DATASET_DIR}\n")

    train_gen, val_gen, test_gen = get_generators(DATASET_DIR)

    print(f"\nTrain batches : {len(train_gen)} ({train_gen.samples} gambar)")
    print(f"Valid batches : {len(val_gen)} ({val_gen.samples} gambar)")
    print(f"Test batches  : {len(test_gen)} ({test_gen.samples} gambar)")
    print(f"Class indices : {train_gen.class_indices}")
