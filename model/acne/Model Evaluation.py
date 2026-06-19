import os
import numpy as np
import tensorflow as tf
from keras.models import load_model
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from model.acne.preprocess import get_generators, CLASS_NAMES

# 1. Konfigurasi
MODEL_PATH = 'resnet50_acne_analyzer.h5'


print("Memuat model...")
model = load_model(MODEL_PATH, compile=False) 
print("Model berhasil dimuat!\n")

def evaluate_test_set():
    """
    Fungsi untuk mengevaluasi seluruh data pada folder test.
    Menghasilkan Classification Report dan Confusion Matrix.
    """
    print("--- Evaluasi Seluruh Data Test ---")
    _, _, test_gen = get_generators()
    
    
    print("Memproses prediksi pada data test...")
    # Lakukan prediksi
    predictions = model.predict(test_gen, verbose=1)
    
    # Ambil index kelas dengan probabilitas tertinggi
    y_pred = np.argmax(predictions, axis=1)
    y_true = test_gen.classes
    
    # Dapatkan nama kelas yang benar dari generator
    labels = list(test_gen.class_indices.keys())
    
    print("\n[ Classification Report ]")
    print(classification_report(y_true, y_pred, target_names=labels))
    
    # Membuat visualisasi Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels)
    plt.title('Confusion Matrix - Skin Analyzer (Acne Severity)')
    plt.ylabel('Label Asli (True Label)')
    plt.xlabel('Prediksi Model (Predicted Label)')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    print("Visualisasi Confusion Matrix telah disimpan sebagai 'confusion_matrix.png'.")
    plt.show()

def predict_single_image(img_path):
    """
    Fungsi simulasi untuk mendeteksi satu gambar jerawat baru.
    """
    print(f"\n--- Deteksi Gambar Tunggal: {img_path} ---")
    if not os.path.exists(img_path):
        print("File gambar tidak ditemukan!")
        return

    # Muat gambar dan ubah ukuran ke 224x224
    img = image.load_img(img_path)
    
    # Ubah gambar ke array numpy
    img_array = image.img_to_array(img)
    
    # Tambahkan dimensi batch (dari (224, 224, 3) menjadi (1, 224, 224, 3))
    img_array = np.expand_dims(img_array, axis=0)
    
    # Aplikasikan preprocess khusus ResNet50
    img_array = preprocess_input(img_array)
    
    # Lakukan prediksi
    preds = model.predict(img_array, verbose=0)
    pred_idx = np.argmax(preds[0])
    confidence = preds[0][pred_idx] * 100
    
    pred_class = CLASS_NAMES[pred_idx]
    
    print(f"Hasil Analisis  : {pred_class}")
    print(f"Tingkat Keyakinan (Confidence): {confidence:.2f}%")
    
    # Tampilkan probabilitas untuk semua kelas
    print("Rincian Probabilitas:")
    for i, cls in enumerate(CLASS_NAMES):
        print(f" - {cls:<15}: {preds[0][i]*100:.2f}%")

if __name__ == "__main__":
    evaluate_test_set()
    