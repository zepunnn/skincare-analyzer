"""
predict.py - Evaluasi Dual-Model (Jerawat + Tipe Kulit)
Menggunakan 2 model terpisah:
  1. resnet50_acne_analyzer.h5         -> Deteksi Jerawat (4 kelas)
  2. mobilenetv2_skin_type.h5 (utama)  -> Deteksi Tipe Kulit (3 kelas)
     resnet50_skin_type.h5 (fallback)
"""
import os
import sys
import glob
sys.stdout.reconfigure(line_buffering=True)

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.preprocessing.image import load_img, img_to_array
from keras.applications.resnet50 import preprocess_input as resnet_preprocess
from keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess
from keras.models import load_model
from collections import defaultdict

# Resolve project root (predict.py lives in <project>/model/oiliness/)
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

# 1. KONFIGURASI
IMG_SIZE = (224, 224)

ACNE_CLASSES = ["Normal", "Mild_Acne", "Moderate_Acne", "Severe_Acne"]
SKIN_CLASSES = ["Dry", "Normal", "Oily"]

ACNE_MODEL_PATH = os.path.join(BASE_DIR, 'resnet50_acne_analyzer.h5')
SKIN_MODEL_PATH_MOBILENET = os.path.join(BASE_DIR, 'model', 'oiliness', 'mobilenetv2_skin_type.h5')
SKIN_MODEL_PATH_RESNET    = os.path.join(BASE_DIR, 'model', 'oiliness', 'resnet50_skin_type.h5')

ACNE_TEST_DIR = os.path.join(BASE_DIR, 'dataset', 'processed', 'test')
SKIN_TEST_DIR = os.path.join(BASE_DIR, 'dataset', 'oiliness', 'processed', 'test')


# 2. MUAT KEDUA MODEL
def load_model_safe(path, name):
    print(f"  Memuat model {name}...", end=" ", flush=True)
    if not os.path.exists(path):
        print(f"GAGAL!\n  [X] File tidak ditemukan: {path}")
        return None
    try:
        m = load_model(path, compile=False)
        print("OK!")
        return m
    except Exception as e:
        print(f"GAGAL!\n  [X] Error: {e}")
        return None


print("=" * 55)
print("  EVALUASI DUAL-MODEL (Jerawat + Tipe Kulit)")
print("=" * 55)
print()

acne_model = load_model_safe(ACNE_MODEL_PATH, "Jerawat")

# Coba load MobileNetV2 dulu, kalau gagal fallback ke ResNet50
skin_model = load_model_safe(SKIN_MODEL_PATH_MOBILENET, "Tipe Kulit (MobileNetV2)")
skin_model_type = 'mobilenetv2'
if skin_model is None:
    skin_model = load_model_safe(SKIN_MODEL_PATH_RESNET, "Tipe Kulit (ResNet50)")
    skin_model_type = 'resnet50'

if acne_model is None and skin_model is None:
    print("\n[X] Tidak ada model yang berhasil dimuat. Keluar.")
    exit()


# 3. FUNGSI PREDIKSI
def predict_single(img_path, model, class_names, preprocess_fn=resnet_preprocess):
    """Prediksi satu gambar menggunakan model tertentu."""
    img = load_img(img_path, target_size=IMG_SIZE)
    img_array = np.expand_dims(img_to_array(img), axis=0)
    img_ready = preprocess_fn(img_array)

    preds = model.predict(img_ready, verbose=0)

    # Handle both dict output (multi-task) and array output (single-task)
    if isinstance(preds, dict):
        preds = list(preds.values())[0]

    preds = preds[0]  # Ambil batch pertama
    pred_idx = np.argmax(preds)

    return {
        'img': img,
        'pred_class': class_names[pred_idx],
        'confidence': preds[pred_idx] * 100,
        'all_probs': preds,
    }


# 4. EVALUASI FOLDER TEST
def evaluate_test_folder(test_dir, model, class_names, task_label, output_dir, preprocess_fn=resnet_preprocess):
    """
    Evaluasi seluruh folder test untuk satu task.
    Menghitung akurasi per kelas dan total, serta menyimpan visualisasi.
    """
    print(f"\n{'=' * 60}")
    print(f"  EVALUASI {task_label.upper()} -- Folder: {test_dir}")
    print(f"{'=' * 60}")

    if not os.path.exists(test_dir):
        print(f"  [X] Folder tidak ditemukan: {test_dir}")
        return

    os.makedirs(output_dir, exist_ok=True)

    # Statistik
    stats = defaultdict(lambda: {'correct': 0, 'total': 0})
    total_correct = 0
    total_images = 0

    # Iterasi setiap subfolder kelas
    for true_class in sorted(os.listdir(test_dir)):
        class_folder = os.path.join(test_dir, true_class)
        if not os.path.isdir(class_folder):
            continue

        # Kumpulkan gambar
        image_paths = []
        for ext in ('*.jpg', '*.jpeg', '*.png', '*.JPG', '*.PNG', '*.webp'):
            image_paths.extend(glob.glob(os.path.join(class_folder, ext)))

        if not image_paths:
            print(f"  [!] {true_class}: Tidak ada gambar")
            continue

        print(f"\n  [>] Kelas: {true_class} ({len(image_paths)} gambar)")

        class_output = os.path.join(output_dir, true_class)
        os.makedirs(class_output, exist_ok=True)

        for img_path in image_paths:
            filename = os.path.basename(img_path)
            result = predict_single(img_path, model, class_names, preprocess_fn=preprocess_fn)

            pred_class = result['pred_class']
            confidence = result['confidence']

            is_correct = (pred_class == true_class)
            stats[true_class]['total'] += 1
            total_images += 1
            if is_correct:
                stats[true_class]['correct'] += 1
                total_correct += 1

            # Visualisasi
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.imshow(result['img'])
            ax.axis('off')

            title_color = 'darkgreen' if is_correct else 'darkred'
            status = "[OK]" if is_correct else "[X]"

            title_text = (
                f"[{task_label}] {status}\n"
                f"True: {true_class}\n"
                f"Pred: {pred_class} ({confidence:.1f}%)"
            )
            ax.set_title(title_text, fontsize=13, fontweight='bold', color=title_color)
            plt.tight_layout()

            save_path = os.path.join(class_output, f"pred_{filename}")
            plt.savefig(save_path, dpi=100)
            plt.close(fig)

        class_acc = (stats[true_class]['correct'] / stats[true_class]['total']) * 100
        print(f"     -> Akurasi: {stats[true_class]['correct']}/{stats[true_class]['total']} "
              f"({class_acc:.1f}%)")

    # Ringkasan
    if total_images > 0:
        overall_acc = (total_correct / total_images) * 100
        print(f"\n{'-' * 60}")
        print(f"  [#] RINGKASAN EVALUASI {task_label.upper()}")
        print(f"{'-' * 60}")
        print(f"  {'Kelas':<20} {'Benar':>8} {'Total':>8} {'Akurasi':>10}")
        print(f"  {'-' * 46}")
        for cls in class_names:
            if cls in stats:
                s = stats[cls]
                acc = (s['correct'] / s['total']) * 100 if s['total'] > 0 else 0
                print(f"  {cls:<20} {s['correct']:>8} {s['total']:>8} {acc:>9.1f}%")
        print(f"  {'-' * 46}")
        print(f"  {'TOTAL':<20} {total_correct:>8} {total_images:>8} {overall_acc:>9.1f}%")
        print(f"{'-' * 60}")

        # Simpan ringkasan ke file teks
        summary_path = os.path.join(output_dir, "ringkasan_evaluasi.txt")
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(f"RINGKASAN EVALUASI {task_label.upper()}\n")
            f.write(f"{'=' * 50}\n")
            f.write(f"{'Kelas':<20} {'Benar':>8} {'Total':>8} {'Akurasi':>10}\n")
            f.write(f"{'-' * 50}\n")
            for cls in class_names:
                if cls in stats:
                    s = stats[cls]
                    acc = (s['correct'] / s['total']) * 100 if s['total'] > 0 else 0
                    f.write(f"{cls:<20} {s['correct']:>8} {s['total']:>8} {acc:>9.1f}%\n")
            f.write(f"{'-' * 50}\n")
            f.write(f"{'TOTAL':<20} {total_correct:>8} {total_images:>8} {overall_acc:>9.1f}%\n")
        print(f"\n  [S] Ringkasan disimpan ke: {summary_path}")
    else:
        print("  [!] Tidak ada gambar yang diproses.")

    print(f"  [I] Visualisasi disimpan di: {output_dir}\n")
    return {'total_correct': total_correct, 'total_images': total_images}


# 5. BLOK EKSEKUSI
if __name__ == "__main__":
    output_base = os.path.join(BASE_DIR, 'model', 'oiliness', 'hasil_prediksi')

    # Evaluasi model jerawat
    if acne_model is not None:
        evaluate_test_folder(
            ACNE_TEST_DIR, acne_model, ACNE_CLASSES,
            task_label="Jerawat",
            output_dir=os.path.join(output_base, 'acne')
        )
    else:
        print("\n[!] Model jerawat tidak tersedia, skip evaluasi jerawat.")

    # Evaluasi model tipe kulit
    if skin_model is not None:
        # Pilih preprocess function yang sesuai dengan model
        skin_preprocess = mobilenet_preprocess if skin_model_type == 'mobilenetv2' else resnet_preprocess
        evaluate_test_folder(
            SKIN_TEST_DIR, skin_model, SKIN_CLASSES,
            task_label=f"Tipe Kulit ({skin_model_type})",
            output_dir=os.path.join(output_base, 'skin'),
            preprocess_fn=skin_preprocess
        )
    else:
        print("\n[!] Model tipe kulit tidak tersedia, skip evaluasi tipe kulit.")

    print("\n[OK] Semua evaluasi selesai!")
