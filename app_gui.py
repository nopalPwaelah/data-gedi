import os
import random
import glob
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms

# ==========================================
# KONFIGURASI MODEL & KELAS
# ==========================================
CLASS_NAMES = ['brain_glioma', 'brain_menin', 'brain_tumor']
CLASS_DISPLAY = {
    'brain_glioma': 'Glioma (Tumor Ganas Jaringan Glia)',
    'brain_menin': 'Meningioma (Tumor Selaput Otak)',
    'brain_tumor': 'Pituitary Tumor (Tumor Kelenjar Hipofisis)'
}
CLASS_DESC = {
    'brain_glioma': 'Tumor otak yang berkembang dari sel glia. Sebagian besar bersifat ganas dan memerlukan penanganan cepat.',
    'brain_menin': 'Tumor yang tumbuh pada selaput meninges pelindung otak. Sebagian besar jinak namun dapat menekan jaringan otak.',
    'brain_tumor': 'Tumor pada kelenjar pituitari di dasar otak yang mengatur produksi hormon tubuh.'
}

MODEL_PATH = os.path.join("outputs", "best_brain_tumor_model.pth")
IMG_SIZE = 224
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# GRAD-CAM IMPLEMENTATION
# ==========================================
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
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

        target_score = output[0, class_idx]
        target_score.backward()

        weights = torch.mean(self.gradients, dim=[2, 3], keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        cam = torch.relu(cam)
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam) + 1e-8)
        return cam, output

# ==========================================
# LOAD MODEL
# ==========================================
def load_model():
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
        try:
            model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
        except TypeError:
            model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model = model.to(device)
    model.eval()
    return model

# ==========================================
# GUI APPLICATION
# ==========================================
class BrainTumorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Brain Tumor Detection & Localization (AI + Grad-CAM)")
        self.root.geometry("1100x750")
        self.root.minsize(950, 700)
        self.root.configure(bg="#1e1e2e")

        # Load AI Model
        self.model = load_model()
        self.grad_cam = GradCAM(self.model, self.model.features[-1])

        self.current_image_path = None
        self.build_ui()

    def build_ui(self):
        # Header Frame
        header = tk.Frame(self.root, bg="#2b2b3d", height=70)
        header.pack(fill=tk.X, side=tk.TOP)

        title_lbl = tk.Label(
            header, text="🩺 Brain Tumor AI Diagnostic Dashboard", 
            font=("Segoe UI", 18, "bold"), fg="#ffffff", bg="#2b2b3d"
        )
        title_lbl.pack(side=tk.LEFT, padx=25, pady=15)

        sub_lbl = tk.Label(
            header, text=f"EfficientNet-B0 + Explainable AI | Device: {device.type.upper()}", 
            font=("Segoe UI", 10), fg="#a6adc8", bg="#2b2b3d"
        )
        sub_lbl.pack(side=tk.RIGHT, padx=25, pady=15)

        # Control Panel (Buttons)
        btn_frame = tk.Frame(self.root, bg="#1e1e2e")
        btn_frame.pack(fill=tk.X, padx=25, pady=15)

        self.btn_browse = tk.Button(
            btn_frame, text="📁 Buka File Scan MRI", font=("Segoe UI", 11, "bold"),
            bg="#89b4fa", fg="#11111b", activebackground="#b4befe",
            padx=18, pady=8, relief=tk.FLAT, cursor="hand2", command=self.browse_file
        )
        self.btn_browse.pack(side=tk.LEFT, padx=(0, 10))

        self.btn_random = tk.Button(
            btn_frame, text="🎲 Uji Contoh Acak dari Dataset", font=("Segoe UI", 11, "bold"),
            bg="#a6e3a1", fg="#11111b", activebackground="#94e2d5",
            padx=18, pady=8, relief=tk.FLAT, cursor="hand2", command=self.load_random_sample
        )
        self.btn_random.pack(side=tk.LEFT, padx=5)

        self.path_lbl = tk.Label(
            btn_frame, text="Silakan pilih gambar MRI untuk mulai menganalisis...", 
            font=("Segoe UI", 9, "italic"), fg="#a6adc8", bg="#1e1e2e", wraplength=450, justify=tk.LEFT
        )
        self.path_lbl.pack(side=tk.LEFT, padx=15)

        # Main Content Area (Images & Analysis)
        content_frame = tk.Frame(self.root, bg="#1e1e2e")
        content_frame.pack(fill=tk.BOTH, expand=True, padx=25, pady=(0, 20))

        # Left: Original Image & Grad-CAM Image
        visual_frame = tk.LabelFrame(
            content_frame, text=" Visualisasi Medis ", font=("Segoe UI", 11, "bold"),
            fg="#cdd6f4", bg="#181825", bd=1, relief=tk.SOLID
        )
        visual_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 15))

        imgs_container = tk.Frame(visual_frame, bg="#181825")
        imgs_container.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)

        # Original Image Card
        card_left = tk.Frame(imgs_container, bg="#11111b", bd=1, relief=tk.SOLID)
        card_left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)

        tk.Label(card_left, text="Scan MRI Asli", font=("Segoe UI", 11, "bold"), 
                 fg="#89dceb", bg="#11111b", pady=8).pack()
        self.lbl_orig_img = tk.Label(card_left, bg="#11111b", text="Belum ada gambar", fg="#6c7086")
        self.lbl_orig_img.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Grad-CAM Heatmap Card
        card_right = tk.Frame(imgs_container, bg="#11111b", bd=1, relief=tk.SOLID)
        card_right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)

        tk.Label(card_right, text="Deteksi Massa Tumor (Grad-CAM AI)", font=("Segoe UI", 11, "bold"), 
                 fg="#f38ba8", bg="#11111b", pady=8).pack()
        self.lbl_cam_img = tk.Label(card_right, bg="#11111b", text="Belum ada analisis", fg="#6c7086")
        self.lbl_cam_img.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Right: Prediction Results & Probabilities
        results_frame = tk.LabelFrame(
            content_frame, text=" Hasil Diagnosa AI ", font=("Segoe UI", 11, "bold"),
            fg="#cdd6f4", bg="#181825", bd=1, relief=tk.SOLID, width=380
        )
        results_frame.pack(side=tk.RIGHT, fill=tk.BOTH, padx=(0, 0))
        results_frame.pack_propagate(False)

        res_inner = tk.Frame(results_frame, bg="#181825", padx=20, pady=20)
        res_inner.pack(fill=tk.BOTH, expand=True)

        tk.Label(res_inner, text="Hasil Prediksi:", font=("Segoe UI", 10), fg="#a6adc8", bg="#181825").pack(anchor=tk.W)
        self.lbl_pred_class = tk.Label(
            res_inner, text="-", font=("Segoe UI", 16, "bold"), 
            fg="#fab387", bg="#181825", wraplength=340, justify=tk.LEFT
        )
        self.lbl_pred_class.pack(anchor=tk.W, pady=(2, 10))

        tk.Label(res_inner, text="Tingkat Keyakinan (Confidence):", font=("Segoe UI", 10), fg="#a6adc8", bg="#181825").pack(anchor=tk.W)
        self.lbl_confidence = tk.Label(
            res_inner, text="0.00%", font=("Segoe UI", 22, "bold"), 
            fg="#a6e3a1", bg="#181825"
        )
        self.lbl_confidence.pack(anchor=tk.W, pady=(0, 15))

        tk.Label(res_inner, text="Probabilitas Semua Kelas:", font=("Segoe UI", 10, "bold"), fg="#cdd6f4", bg="#181825").pack(anchor=tk.W, pady=(5, 5))

        self.prob_bars = {}
        for cname in CLASS_NAMES:
            p_frame = tk.Frame(res_inner, bg="#181825")
            p_frame.pack(fill=tk.X, pady=4)

            name_lbl = tk.Label(p_frame, text=cname.replace("brain_", "").capitalize(), font=("Segoe UI", 9), fg="#bac2de", bg="#181825", width=10, anchor=tk.W)
            name_lbl.pack(side=tk.LEFT)

            progress = ttk.Progressbar(p_frame, orient=tk.HORIZONTAL, length=160, mode='determinate')
            progress.pack(side=tk.LEFT, padx=8)

            val_lbl = tk.Label(p_frame, text="0.0%", font=("Segoe UI", 9, "bold"), fg="#cdd6f4", bg="#181825", width=6, anchor=tk.E)
            val_lbl.pack(side=tk.LEFT)

            self.prob_bars[cname] = (progress, val_lbl)

        # Tumor description note
        tk.Label(res_inner, text="Keterangan Medis:", font=("Segoe UI", 10, "bold"), fg="#cdd6f4", bg="#181825").pack(anchor=tk.W, pady=(20, 5))
        self.lbl_desc = tk.Label(
            res_inner, text="Pilih scan MRI untuk melihat informasi klinis tumor.", 
            font=("Segoe UI", 9), fg="#9399b2", bg="#181825", wraplength=330, justify=tk.LEFT
        )
        self.lbl_desc.pack(anchor=tk.W)

    def browse_file(self):
        initial_dir = os.path.join(os.getcwd(), "Multi Cancer", "Multi Cancer", "Brain Cancer")
        if not os.path.exists(initial_dir):
            initial_dir = os.getcwd()

        file_path = filedialog.askopenfilename(
            initialdir=initial_dir,
            title="Pilih Citra Scan MRI",
            filetypes=[("Image files", "*.jpg;*.jpeg;*.png")]
        )
        if file_path:
            self.process_image(file_path)

    def load_random_sample(self):
        sample_pool = glob.glob(os.path.join("Multi Cancer", "Multi Cancer", "Brain Cancer", "*", "*.jpg"))
        if sample_pool:
            sample_path = random.choice(sample_pool)
            self.process_image(sample_path)
        else:
            messagebox.showerror("Error", "Dataset tidak ditemukan di folder Multi Cancer.")

    def process_image(self, file_path):
        self.current_image_path = file_path
        self.path_lbl.config(text=os.path.basename(file_path))

        raw_img = Image.open(file_path).convert("RGB")
        orig_np = np.array(raw_img)

        # Preprocessing
        preprocess = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225])
        ])
        input_tensor = preprocess(raw_img).unsqueeze(0).to(device)

        # Grad-CAM Inference
        heatmap, logits = self.grad_cam.generate_heatmap(input_tensor)
        probs = torch.softmax(logits, dim=1).squeeze().detach().cpu().numpy()
        pred_idx = np.argmax(probs)
        pred_class = CLASS_NAMES[pred_idx]
        pred_conf = probs[pred_idx] * 100

        # Update Text UI
        self.lbl_pred_class.config(text=CLASS_DISPLAY[pred_class])
        self.lbl_confidence.config(text=f"{pred_conf:.2f}%")
        self.lbl_desc.config(text=CLASS_DESC[pred_class])

        for i, cname in enumerate(CLASS_NAMES):
            prog, val = self.prob_bars[cname]
            prog['value'] = probs[i] * 100
            val.config(text=f"{probs[i]*100:.1f}%")

        # Colorize Heatmap and Blend
        heatmap_resized = cv2.resize(heatmap, (orig_np.shape[1], orig_np.shape[0]))
        heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        overlay_np = np.uint8(0.6 * orig_np + 0.4 * heatmap_colored)

        # Render to Tkinter Labels
        target_size = (340, 340)
        img_orig_disp = ImageTk.PhotoImage(raw_img.resize(target_size, Image.Resampling.LANCZOS))
        img_cam_disp = ImageTk.PhotoImage(Image.fromarray(overlay_np).resize(target_size, Image.Resampling.LANCZOS))

        self.lbl_orig_img.config(image=img_orig_disp, text="")
        self.lbl_orig_img.image = img_orig_disp

        self.lbl_cam_img.config(image=img_cam_disp, text="")
        self.lbl_cam_img.image = img_cam_disp

if __name__ == "__main__":
    root = tk.Tk()
    app = BrainTumorApp(root)
    root.mainloop()
