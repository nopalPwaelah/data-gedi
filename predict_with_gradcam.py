import os
import sys
import argparse
import random
import glob
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2

# ==========================================
# 1. KONFIGURASI
# ==========================================
CLASS_NAMES = ['brain_glioma', 'brain_menin', 'brain_tumor']
CLASS_DISPLAY = {
    'brain_glioma': 'Glioma (Tumor Ganas Glia)',
    'brain_menin': 'Meningioma (Tumor Selaput Otak)',
    'brain_tumor': 'Pituitary Tumor (Tumor Kelenjar Hipofisis)'
}
MODEL_PATH = os.path.join("outputs", "best_brain_tumor_model.pth")
IMG_SIZE = 224

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# 2. IMPLEMENTASI GRAD-CAM (NATIVE PYTORCH)
# ==========================================
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Pasang hook untuk merekam aktivasi dan gradien
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output.detach()

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate_heatmap(self, input_tensor, class_idx=None):
        self.model.zero_grad()
        output = self.model(input_tensor)

        if class_idx is None:
            class_idx = torch.argmax(output, dim=1).item()

        # Backward untuk kelas target
        target_score = output[0, class_idx]
        target_score.backward()

        # Global average pooling pada gradien
        weights = torch.mean(self.gradients, dim=[2, 3], keepdim=True)
        # Weighted combination of activation maps
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        cam = torch.relu(cam)  # Hanya ambil pengaruh positif

        # Normalisasi ke [0, 1]
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam) + 1e-8)
        return cam, output

# ==========================================
# 3. MEMUAT MODEL
# ==========================================
def load_trained_model():
    try:
        weights = models.EfficientNet_B0_Weights.DEFAULT
        model = models.efficientnet_b0(weights=weights)
    except AttributeError:
        model = models.efficientnet_b0(pretrained=True)

    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, len(CLASS_NAMES))
    )

    if os.path.exists(MODEL_PATH):
        print(f"[*] Memuat bobot model terlatih dari: {MODEL_PATH}")
        try:
            model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
        except TypeError:
            model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    else:
        print(f"[!] PERINGATAN: File '{MODEL_PATH}' belum ditemukan. Menggunakan bobot awal untuk demonstrasi.")
    
    model = model.to(device)
    model.eval()
    return model

# ==========================================
# 4. FUNGSI PREDIKSI & VISUALISASI
# ==========================================
def predict_and_explain(image_path, model, output_result_path="outputs/prediction_result.png"):
    print(f"\n[*] Menganalisis scan MRI: {image_path}")
    raw_img = Image.open(image_path).convert("RGB")
    orig_np = np.array(raw_img)

    # Preprocessing
    preprocess = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ])
    input_tensor = preprocess(raw_img).unsqueeze(0).to(device)

    # Target layer untuk Grad-CAM pada EfficientNet-B0 (layer konvolusi terakhir)
    target_layer = model.features[-1]
    grad_cam = GradCAM(model, target_layer)

    # Dapatkan heatmap dan output logits
    heatmap, logits = grad_cam.generate_heatmap(input_tensor)
    probs = torch.softmax(logits, dim=1).squeeze().detach().cpu().numpy()
    pred_idx = np.argmax(probs)
    pred_class = CLASS_NAMES[pred_idx]
    pred_confidence = probs[pred_idx] * 100

    # Print hasil ke terminal
    print("\n" + "="*45)
    print("              HASIL PREDIKSI")
    print("="*45)
    print(f"Prediksi Tumor    : {CLASS_DISPLAY[pred_class]}")
    print(f"Confidence Score  : {pred_confidence:.2f}%")
    print("-" * 45)
    print("Rincian Probabilitas:")
    for i, cname in enumerate(CLASS_NAMES):
        bar = "#" * int(probs[i] * 30)
        print(f"  {cname:<12} : {probs[i]*100:5.1f}% | {bar}")
    print("="*45)

    # Resize heatmap agar pas dengan ukuran asli gambar MRI
    heatmap_resized = cv2.resize(heatmap, (orig_np.shape[1], orig_np.shape[0]))
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    # Overlay (tumpuk) heatmap di atas citra MRI asli
    overlay = np.uint8(0.6 * orig_np + 0.4 * heatmap_colored)

    # Buat figure perbandingan
    plt.figure(figsize=(12, 5))

    # Gambar Asli
    plt.subplot(1, 2, 1)
    plt.imshow(orig_np)
    plt.title(f"Scan MRI Asli\nFile: {os.path.basename(image_path)}", fontsize=11)
    plt.axis('off')

    # Grad-CAM Overlay
    plt.subplot(1, 2, 2)
    plt.imshow(overlay)
    plt.title(f"Deteksi AI (Grad-CAM)\nPrediksi: {pred_class} ({pred_confidence:.1f}%)", 
              fontsize=11, color="darkred" if pred_confidence > 70 else "black")
    plt.axis('off')

    os.makedirs(os.path.dirname(output_result_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_result_path, dpi=300)
    plt.close()

    print(f"[+] Visualisasi Grad-CAM berhasil disimpan di: {output_result_path}\n")

# ==========================================
# 5. ENTRY POINT
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deteksi Kanker Otak dengan Grad-CAM Explainable AI")
    parser.add_argument("--image", type=str, default="", help="Path ke file citra scan MRI")
    args = parser.parse_args()

    # Jika user tidak memasukkan path gambar, ambil 1 contoh acak dari dataset
    image_path = args.image
    if not image_path or not os.path.exists(image_path):
        sample_pool = glob.glob(os.path.join("Multi Cancer", "Multi Cancer", "Brain Cancer", "*", "*.jpg"))
        if sample_pool:
            image_path = random.choice(sample_pool)
            print(f"[*] Tidak ada file gambar yang dispesifikasikan. Menggunakan contoh acak:\n    {image_path}")
        else:
            print("[!] Error: Tidak ada dataset ditemukan di folder 'Multi Cancer'.")
            sys.exit(1)

    model = load_trained_model()
    predict_and_explain(image_path, model)
