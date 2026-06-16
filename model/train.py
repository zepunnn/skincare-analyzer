import tensorflow as tf
from keras.applications import ResNet50
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping

# 1. Muat Generator Data
from preprocess import get_generators
train_gen, val_gen, test_gen = get_generators()

# 2. Bangun Arsitektur ResNet-50
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(4, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

#'categorical_crossentropy' karena class_mode="categorical"
model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss='categorical_crossentropy', 
    metrics=['accuracy', tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='recall')]
)

# 3. Callbacks untuk menyimpan model terbaik
checkpoint = ModelCheckpoint(
    'resnet50_acne_analyzer.h5', 
    monitor='val_loss', 
    save_best_only=True, 
    mode='min', 
    verbose=1
)
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# 4. Tahap Training (Warm-up)
print("\n--- Memulai Tahap Warm-up ---")
EPOCHS_WARMUP = 10
history_warmup = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS_WARMUP,
    callbacks=[checkpoint, early_stop]
)

# 5. Tahap Fine-Tuning
print("\n--- Memulai Tahap Fine-Tuning ---")
base_model.trainable = True

model.compile(
    optimizer=Adam(learning_rate=1e-5),
    loss='categorical_crossentropy',
    metrics=['accuracy', tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='recall')]
)

EPOCHS_FINETUNE = 20
history_finetune = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS_WARMUP + EPOCHS_FINETUNE,
    initial_epoch=history_warmup.epoch[-1],
    callbacks=[checkpoint, early_stop]
)

# 6. Evaluasi di Data Test
print("\n--- Evaluasi Model ---")
test_loss, test_acc, test_prec, test_rec = model.evaluate(test_gen)
print(f"Test Accuracy : {test_acc:.4f}")
print(f"Test Precision: {test_prec:.4f}")
print(f"Test Recall   : {test_rec:.4f}")