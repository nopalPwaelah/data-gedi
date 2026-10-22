import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt

# ==========================================
# 1. KONFIGURASI & HYPERPARAMETERS
# ==========================================
DATA_DIR = os.path.join("Multi Cancer", "Multi Cancer", "Brain Cancer")
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODEL_SAVE_PATH = os.path.join(OUTPUT_DIR, "best_brain_tumor_model.pth")
BATCH_SIZE = 32
NUM_EPOCHS = 10
LEARNING_RATE = 1e-4
NUM_CLASSES = 3
IMG_SIZE = 224

# Cek ketersediaan GPU (NVIDIA RTX 4060)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[*] Menggunakan perangkat: {device}")
if torch.cuda.is_available():
    print(f"[*] Nama GPU: {torch.cuda.get_device_name(0)}")

# ==========================================
# 2. DATA AUGMENTASI & TRANSFORMASI
# ==========================================
train_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=10),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225])
])

eval_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ==========================================
# 3. LOAD DATASET & SPLIT TRAIN/TEST
# ==========================================
print(f"\n[*] Memuat data dari folder: {DATA_DIR}")
full_dataset = datasets.ImageFolder(root=DATA_DIR)
class_names = full_dataset.classes
print(f"[*] Kelas yang terdeteksi ({len(class_names)}): {class_names}")

# Wrapper untuk menerapkan transformasi berbeda pada tiap subset.
class DatasetFromSubset(torch.utils.data.Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.subset)

total_size = len(full_dataset)
train_size = int(0.85 * total_size)
test_size = total_size - train_size
train_subset, test_subset = random_split(
    full_dataset,
    [train_size, test_size],
    generator=torch.Generator().manual_seed(42)
)

train_dataset = DatasetFromSubset(train_subset, transform=train_transforms)
test_dataset = DatasetFromSubset(test_subset, transform=eval_transforms)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

print(f"[*] Jumlah Data Train : {len(train_dataset)}")
print(f"[*] Jumlah Data Test  : {len(test_dataset)}")

# ==========================================
# 4. MEMBUAT MODEL TRANSFER LEARNING
# ==========================================
print("\n[*] Menginisialisasi arsitektur EfficientNet-B0 (Pretrained)...")
try:
    weights = models.EfficientNet_B0_Weights.DEFAULT
    model = models.efficientnet_b0(weights=weights)
except AttributeError:
    model = models.efficientnet_b0(pretrained=True)

in_features = model.classifier[1].in_features
model.classifier[1] = nn.Sequential(
    nn.Dropout(p=0.3),
    nn.Linear(in_features, NUM_CLASSES)
)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-2)
scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

# ==========================================
# 5. TRAINING LOOP
# ==========================================
def train_model():
    history = {
        'train_loss': [],
        'train_acc': []
    }

    start_time = time.time()
    print("\n" + "="*50)
    print("           MEMULAI PROSES TRAINING")
    print("="*50)

    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch + 1}/{NUM_EPOCHS}")
        print("-" * 25)

        model.train()
        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in train_loader:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_train_loss = running_loss / len(train_dataset)
        epoch_train_acc = (running_corrects.double() / len(train_dataset)).item()

        history['train_loss'].append(epoch_train_loss)
        history['train_acc'].append(epoch_train_acc)

        print(f"Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc*100:.2f}%")

    time_elapsed = time.time() - start_time
    print(f"\n[*] Training selesai dalam {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"[*] Model disimpan di: {MODEL_SAVE_PATH}")
    return model, history

# ==========================================
# 6. VISUALISASI TRAINING
# ==========================================
def plot_training_history(history):
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.title('Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot([acc * 100 for acc in history['train_acc']], label='Train Accuracy')
    plt.title('Accuracy Curve (%)')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    curves_path = os.path.join(OUTPUT_DIR, "training_curves.png")
    plt.savefig(curves_path, dpi=300)
    plt.close()
    print(f"[*] Grafik training disimpan di: {curves_path}")

def evaluate_test_set(model):
    model.eval()
    correct = 0
    confusion = torch.zeros((NUM_CLASSES, NUM_CLASSES), dtype=torch.int64)

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device, non_blocking=True)
            outputs = model(inputs)
            predictions = outputs.argmax(dim=1).cpu()

            correct += (predictions == labels).sum().item()
            for actual, predicted in zip(labels, predictions):
                confusion[actual, predicted] += 1

    accuracy = correct / len(test_dataset)
    print(f"\n[*] Akurasi Test: {accuracy * 100:.2f}%")
    print("[*] Confusion Matrix (baris=label asli, kolom=prediksi):")
    print(confusion.numpy())

    matrix_path = os.path.join(OUTPUT_DIR, "confusion_matrix.png")
    plt.figure(figsize=(7, 6))
    plt.imshow(confusion.numpy(), interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion Matrix - Test Set")
    plt.colorbar()
    tick_marks = range(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    plt.xlabel("Prediksi")
    plt.ylabel("Label Sebenarnya")
    plt.tight_layout()
    plt.savefig(matrix_path, dpi=300)
    plt.close()
    print(f"[*] Confusion matrix disimpan di: {matrix_path}")

if __name__ == "__main__":
    trained_model, history = train_model()
    plot_training_history(history)
    evaluate_test_set(trained_model)
