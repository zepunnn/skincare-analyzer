"""
preprocess.py
Menyiapkan dataset ACNE04 (format: nama_file.jpg  _  label)
Label asli 0-4 (IGA scale) → dipetakan ke 4 kelas:
  0, 1  →  Normal
  2     →  Mild_Acne
  3     →  Moderate_Acne
  4     →  Severe_Acne
"""

import os
import shutil
import glob
import numpy as np
from collections import defaultdict
from keras.src.legacy.preprocessing.image import ImageDataGenerator
from keras.applications.resnet50 import preprocess_input

RAW_IMAGES_DIR = "Classification/JPEGImages"
TXT_TRAINVAL   = "Classification/NNEW_trainval_0.txt"
TXT_TEST       = "Classification/NNEW_test_0.txt"
PROCESSED_DIR  = "dataset/processed"
IMG_SIZE       = (224, 224)
BATCH_SIZE     = 32
RANDOM_SEED    = 42

LABEL_MAP = {
    0: "Normal",
    1: "Mild_Acne",
    2: "Moderate_Acne",
    3: "Severe_Acne",
}
CLASS_NAMES = ["Normal", "Mild_Acne", "Moderate_Acne", "Severe_Acne"]

def parse_txt(txt_path):
    """
    Membaca file txt ACNE04.
    Format tiap baris: nama_file.jpg  <ignored>  <label>
    Mengembalikan list of (filename, class_name).
    """
    entries = []
    with open(txt_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            filename = parts[0]
            label    = int(parts[1])
            cls      = LABEL_MAP[label]
            entries.append((filename, cls))
    return entries

def build_processed_dataset(val_ratio=0.15):

    for split in ("train", "val", "test"):
        for cls in CLASS_NAMES:
            os.makedirs(os.path.join(PROCESSED_DIR, split, cls), exist_ok=True)

    trainval_entries = []
    for txt in sorted(glob.glob("Classification/NNEW_trainval_*.txt")):
        trainval_entries += parse_txt(txt)

    seen = set()
    unique_trainval = []
    for fname, cls in trainval_entries:
        if fname not in seen:
            seen.add(fname)
            unique_trainval.append((fname, cls))

    rng = np.random.default_rng(RANDOM_SEED)
    indices = rng.permutation(len(unique_trainval))
    val_count = int(len(unique_trainval) * val_ratio)
    val_idx   = set(indices[:val_count])

    train_list, val_list = [], []
    for i, entry in enumerate(unique_trainval):
        (val_list if i in val_idx else train_list).append(entry)

    test_entries = []
    for txt in sorted(glob.glob("Classification/NNEW_test_*.txt")):
        test_entries += parse_txt(txt)
    seen_test = set()
    test_list = []
    for fname, cls in test_entries:
        if fname not in seen_test:
            seen_test.add(fname)
            test_list.append((fname, cls))

    stats = defaultdict(lambda: defaultdict(int))
    missing = 0

    for split_name, split_data in [("train", train_list),
                                    ("val",   val_list),
                                    ("test",  test_list)]:
        for fname, cls in split_data:
            src = os.path.join(RAW_IMAGES_DIR, fname)
            dst = os.path.join(PROCESSED_DIR, split_name, cls, fname)
            if os.path.exists(src):
                shutil.copy(src, dst)
                stats[split_name][cls] += 1
            else:
                missing += 1

    print("\n✅ Dataset berhasil disusun!\n")
    print(f"{'Kelas':<20} {'Train':>8} {'Val':>8} {'Test':>8}")
    print("-" * 46)
    for cls in CLASS_NAMES:
        print(f"{cls:<20} {stats['train'][cls]:>8} "
              f"{stats['val'][cls]:>8} {stats['test'][cls]:>8}")
    if missing:
        print(f"\n⚠️  {missing} file tidak ditemukan di {RAW_IMAGES_DIR}")

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
    val_test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

    train_gen = train_datagen.flow_from_directory(
        os.path.join(processed_dir, "train"),
        target_size=img_size,
        batch_size=batch_size,
        class_mode="categorical",
        classes=CLASS_NAMES,
        shuffle=True,
    )
    val_gen = val_test_datagen.flow_from_directory(
        os.path.join(processed_dir, "val"),
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

if __name__ == "__main__":
    build_processed_dataset()
    print("\n📦 Mengecek generator...")
    train_gen, val_gen, test_gen = get_generators()
    print(f"Train batches : {len(train_gen)}")
    print(f"Val batches   : {len(val_gen)}")
    print(f"Test batches  : {len(test_gen)}")
    print(f"Class indices : {train_gen.class_indices}")