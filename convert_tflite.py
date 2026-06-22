import tensorflow as tf
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

acne_h5 = os.path.join(BASE_DIR, 'resnet50_acne_analyzer.h5')
skin_h5 = os.path.join(BASE_DIR, 'model', 'oiliness', 'mobilenetv2_skin_type.h5')

acne_tflite = os.path.join(BASE_DIR, 'resnet50_acne_analyzer.tflite')
skin_tflite = os.path.join(BASE_DIR, 'model', 'oiliness', 'mobilenetv2_skin_type.tflite')

def convert_model(h5_path, tflite_path):
    if not os.path.exists(h5_path):
        print(f"[!] File tidak ditemukan: {h5_path}")
        return
        
    print(f"\nMemuat model: {h5_path}")
    model = tf.keras.models.load_model(h5_path, compile=False)
    
    print("Mengonversi ke TFLite...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    # Optional: Optimize for size
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    
    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)
        
    print(f"[OK] Model disimpan sebagai: {tflite_path}")

print("=== KONVERSI MODEL H5 KE TFLITE ===")
convert_model(acne_h5, acne_tflite)
convert_model(skin_h5, skin_tflite)
print("\nSemua konversi selesai.")
