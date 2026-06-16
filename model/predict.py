import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.preprocessing.image import load_img, img_to_array
from keras.applications.resnet50 import preprocess_input
from keras.models import load_model

# 1. KONFIGURASI & VARIABEL GLOBAL
IMG_SIZE = (224, 224)
CLASS_NAMES = ["Normal", "Mild_Acne", "Moderate_Acne", "Severe_Acne"]
MODEL_PATH = 'resnet50_acne_analyzer.h5'

# 2. INISIALISASI MODEL
print("Menyiapkan sistem... Memuat model medis...")
try:
    model = load_model(MODEL_PATH, compile=False)
    print("Model berhasil dimuat!\n")
except Exception as e:
    print(f" Gagal memuat model. Pastikan file '{MODEL_PATH}' ada di folder yang sama.")
    print(f"Detail error: {e}")
    exit()

# 3. FUNGSI UTAMA (BATCH PROCESSING)
def predict_folder(folder_path, output_dir="hasil_deteksi"):
    """
    Membaca semua gambar di dalam folder, memprediksi tingkat keparahan jerawat,
    lalu membandingkannya dengan True Label yang diambil dari nama folder.
    """
    print(f"Memulai Batch Processing di Folder: {folder_path}")
    
    if not os.path.exists(folder_path):
        print(f"❌ Folder tidak ditemukan: {folder_path}")
        return
        
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # Mengambil nama kelas asli (True Label) dari nama folder terakhir
    true_class = os.path.basename(os.path.normpath(folder_path))
    print(f" True Label untuk batch ini diatur sebagai: '{true_class}'")
        
    image_paths = []
    for ext in ('*.jpg', '*.jpeg', '*.png', '*.JPG', '*.PNG'):
        image_paths.extend(glob.glob(os.path.join(folder_path, ext)))
        
    if not image_paths:
        print("❌ Tidak ada file gambar di dalam folder tersebut.")
        return

    print(f"Ditemukan {len(image_paths)} gambar. Mulai memproses...\n")

    for img_path in image_paths:
        filename = os.path.basename(img_path)
        print(f"Menganalisis: {filename}...")

        # A. Pemrosesan Gambar
        img = load_img(img_path, target_size=IMG_SIZE)
        img_array = img_to_array(img)
        img_array_expanded = np.expand_dims(img_array, axis=0)
        img_ready = preprocess_input(img_array_expanded)
        
        # B. Prediksi Model
        preds = model.predict(img_ready, verbose=0)[0]
        pred_idx = np.argmax(preds)
        confidence = preds[pred_idx] * 100
        pred_class = CLASS_NAMES[pred_idx]
        
        # C. Cek Kebenaran Prediksi & Visualisasi
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(img)
        ax.axis('off')
        
        # Tentukan warna teks (Hijau jika benar, Merah jika salah)
        is_correct = (pred_class == true_class)
        title_color = 'darkgreen' if is_correct else 'darkred'
        
        # Tampilkan True Label, Prediksi, dan Confidence
        title_text = (f"True Label: {true_class}\n"
                      f"Prediksi: {pred_class}\n"
                      f"Confidence: {confidence:.2f}%")
        
        ax.set_title(title_text, fontsize=14, fontweight='bold', color=title_color)
        
        plt.tight_layout()
        
        # D. Simpan dan Bersihkan Memori
        save_path = os.path.join(output_dir, f"result_{filename}")
        plt.savefig(save_path)
        plt.close(fig) 

    print(f"\nSelesai! Silakan buka folder '{output_dir}' untuk melihat hasilnya.")

# 4. BLOK EKSEKUSI
if __name__ == "__main__":
    # Sesuaikan dengan lokasi foldermu
    target_folder = "dataset/processed/test/Moderate_Acne"
    predict_folder(target_folder, output_dir="hasil_deteksi_moderate")