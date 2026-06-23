"""
train_mobilenetv2.py
Training model klasifikasi tipe kulit (dry / oil)
menggunakan arsitektur MobileNetV2 sebagai backbone.

Dataset: skin_dataset/ (3 kelas: acne, dry, oil)

Referensi:
  https://github.com/MdAliAhnaf/Skin_Type_Classification-Recommendation
  notebook: skin_classify_MobileNetV2.ipynb

Arsitektur head (dari notebook):
  MobileNetV2 base → AveragePooling2D(7,7) → Flatten → Dense(256, relu)
  → Dropout(0.25) → Dense(3, softmax)

Evaluasi di notebook: accuracy ~94%
"""

import os
import sys
sys.stdout.reconfigure(line_buffering=True)

import tensorflow as tf
from keras.applications import MobileNetV2
from keras.models import Model
from keras.layers import Dense, AveragePooling2D, Dropout, Flatten
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping

# Resolve project root (train_mobilenetv2.py lives in <project>/model/oiliness/)
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

# Import generator dari preprocess_mobilenetv2.py
from preprocess_mobilenetv2 import get_generators

# 1. KONFIGURASI
DATASET_DIR = os.path.join(BASE_DIR, 'skin_dataset')
MODEL_NAME  = os.path.join(BASE_DIR, 'model', 'oiliness', 'mobilenetv2_skin_type.h5')
EPOCHS      = 30
INIT_LR     = 1e-4
BATCH_SIZE  = 64    # Sesuai notebook referensi
IMG_SIZE    = (224, 224)
NUM_CLASSES = 2     # dry, oil

# 2. MUAT DATA
print("=" * 60)
print("  TRAINING MODEL KLASIFIKASI KULIT (dry / oil)")
print("  Arsitektur: MobileNetV2")
print("=" * 60)
print(f"\nDataset : {DATASET_DIR}")
print(f"Output  : {MODEL_NAME}\n")

train_gen, val_gen, test_gen = get_generators(
    dataset_dir=DATASET_DIR,
    img_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

print(f"\nTrain   : {train_gen.samples} gambar ({len(train_gen)} batch)")
print(f"Valid   : {val_gen.samples} gambar ({len(val_gen)} batch)")
print(f"Test    : {test_gen.samples} gambar ({len(test_gen)} batch)")
print(f"Kelas   : {list(train_gen.class_indices.keys())}")

# 3. BANGUN ARSITEKTUR MobileNetV2
print("\n--- Membangun Arsitektur MobileNetV2 ---")

# Load MobileNetV2 sebagai base model (tanpa top/classification layer)
base_model = MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)
)

# Freeze seluruh layer base model
for layer in base_model.layers:
    layer.trainable = False

# Bangun head model sesuai notebook MobileNetV2
head = base_model.output
head = AveragePooling2D(pool_size=(7, 7))(head)
head = Flatten(name="flatten")(head)
head = Dense(256, activation="relu")(head)
head = Dropout(0.25)(head)
predictions = Dense(NUM_CLASSES, activation="softmax")(head)

model = Model(inputs=base_model.input, outputs=predictions)

# Compile dengan Adam (sesuai notebook)
opt = Adam(learning_rate=INIT_LR)

model.compile(
    optimizer=opt,
    loss='categorical_crossentropy',
    metrics=[
        'accuracy',
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall')
    ]
)

model.summary()

# 4. CALLBACKS
checkpoint = ModelCheckpoint(
    MODEL_NAME,
    monitor='val_accuracy',
    save_best_only=True,
    mode='max',
    verbose=1
)

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=7,
    restore_best_weights=True,
    verbose=1
)

# 5. TRAINING (Base model frozen, hanya head yang dilatih)
print(f"\n--- Training MobileNetV2 ({EPOCHS} epoch) ---")
print("Base MobileNetV2 di-freeze, hanya melatih head classifier.\n")

history = model.fit(
    train_gen,
    steps_per_epoch=len(train_gen),
    validation_data=val_gen,
    validation_steps=len(val_gen),
    epochs=EPOCHS,
    callbacks=[checkpoint, early_stop]
)

# 6. EVALUASI DI DATA TEST
print("\n--- Evaluasi Model di Data Test ---")
test_loss, test_acc, test_prec, test_rec = model.evaluate(test_gen)
print(f"\nTest Accuracy : {test_acc:.4f}")
print(f"Test Precision: {test_prec:.4f}")
print(f"Test Recall   : {test_rec:.4f}")
print(f"Test Loss     : {test_loss:.4f}")

# F1 Score
if test_prec + test_rec > 0:
    f1 = 2 * (test_prec * test_rec) / (test_prec + test_rec)
    print(f"Test F1-Score : {f1:.4f}")

print(f"\n[OK] Model berhasil disimpan sebagai: {MODEL_NAME}")

# 7. SIMPAN HISTORY TRAINING
try:
    import json
    history_path = os.path.join(
        BASE_DIR, 'model', 'oiliness', 'mobilenetv2_training_history.json'
    )
    with open(history_path, 'w') as f:
        hist_data = {}
        for key, values in history.history.items():
            hist_data[key] = [float(v) for v in values]
        json.dump(hist_data, f, indent=2)
    print(f"[OK] Training history disimpan di: {history_path}")
except Exception as e:
    print(f"[!] Gagal menyimpan history: {e}")
