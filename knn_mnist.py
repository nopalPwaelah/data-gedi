import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from torchvision import datasets


DATA_DIR = os.path.join("data", "mnist")
OUTPUT_DIR = os.path.join("outputs", "knn_mnist")
TRAIN_SAMPLES = 6000
TEST_SAMPLES = 1000
HISTOGRAM_BINS = 16
K_NEIGHBORS = 3
RANDOM_STATE = 42
CLASS_NAMES = [str(digit) for digit in range(10)]

os.makedirs(OUTPUT_DIR, exist_ok=True)


def select_samples(images, labels, sample_count, rng):
    """Select a reproducible sample while keeping the original images and labels paired."""
    indices = rng.choice(len(images), size=sample_count, replace=False)
    return images[indices], labels[indices]


def extract_histogram_features(images, bins=HISTOGRAM_BINS):
    """Represent each grayscale image by a normalized intensity histogram."""
    features = np.empty((len(images), bins), dtype=np.float32)
    for index, image in enumerate(images):
        histogram, _ = np.histogram(image, bins=bins, range=(0, 256))
        features[index] = histogram / image.size
    return features


def save_sample_images(images, labels):
    figure, axes = plt.subplots(5, 5, figsize=(8, 8))
    for index, axis in enumerate(axes.flat):
        axis.imshow(images[index], cmap="gray", vmin=0, vmax=255)
        axis.set_title(f"Label: {labels[index]}")
        axis.axis("off")

    figure.tight_layout()
    path = os.path.join(OUTPUT_DIR, "mnist_samples.png")
    figure.savefig(path, dpi=150)
    plt.close(figure)
    print(f"[*] Contoh gambar disimpan: {path}")


def evaluate_classifier(name, features_train, labels_train, features_test, labels_test):
    classifier = KNeighborsClassifier(
        n_neighbors=K_NEIGHBORS,
        metric="euclidean",
        n_jobs=-1,
    )

    start_time = time.time()
    classifier.fit(features_train, labels_train)
    predictions = classifier.predict(features_test)
    elapsed = time.time() - start_time

    accuracy = accuracy_score(labels_test, predictions)
    print(f"\n=== {name} ===")
    print(f"Akurasi: {accuracy * 100:.2f}%")
    print(f"Waktu fit + prediksi: {elapsed:.2f} detik")
    print(classification_report(
        labels_test,
        predictions,
        labels=np.arange(10),
        target_names=CLASS_NAMES,
        zero_division=0,
    ))

    return classifier, predictions, accuracy


def save_comparison(histogram_accuracy, pixel_accuracy, labels_test, histogram_predictions, pixel_predictions):
    figure, axes = plt.subplots(1, 3, figsize=(17, 5))

    axes[0].bar(
        ["Histogram", "Raw Pixels"],
        [histogram_accuracy * 100, pixel_accuracy * 100],
        color=["#2a9d8f", "#e76f51"],
    )
    axes[0].set_title("Perbandingan Akurasi KNN")
    axes[0].set_ylabel("Akurasi (%)")
    axes[0].set_ylim(0, 100)

    for axis, predictions, title in [
        (axes[1], histogram_predictions, "Confusion Matrix: Histogram"),
        (axes[2], pixel_predictions, "Confusion Matrix: Raw Pixels"),
    ]:
        matrix = confusion_matrix(labels_test, predictions, labels=np.arange(10))
        axis.imshow(matrix, cmap="Blues")
        axis.set_title(title)
        axis.set_xlabel("Prediksi")
        axis.set_ylabel("Label Asli")
        axis.set_xticks(range(10))
        axis.set_yticks(range(10))

    figure.tight_layout()
    path = os.path.join(OUTPUT_DIR, "knn_comparison.png")
    figure.savefig(path, dpi=160)
    plt.close(figure)
    print(f"[*] Grafik perbandingan disimpan: {path}")


def predict_single_image(image, true_label, histogram_model, pixel_model):
    histogram_features = extract_histogram_features(image[np.newaxis, ...])
    pixel_features = image.reshape(1, -1).astype(np.float32) / 255.0

    histogram_prediction = int(histogram_model.predict(histogram_features)[0])
    pixel_prediction = int(pixel_model.predict(pixel_features)[0])

    print("\n=== Prediksi Satu Gambar ===")
    print(f"Label sebenarnya: {true_label}")
    print(f"Prediksi histogram: {histogram_prediction}")
    print(f"Prediksi raw pixels: {pixel_prediction}")

    figure, axis = plt.subplots(figsize=(4, 4))
    axis.imshow(image, cmap="gray", vmin=0, vmax=255)
    axis.set_title(
        f"Asli: {true_label} | Histogram: {histogram_prediction}\n"
        f"Raw pixels: {pixel_prediction}"
    )
    axis.axis("off")
    figure.tight_layout()
    path = os.path.join(OUTPUT_DIR, "single_image_prediction.png")
    figure.savefig(path, dpi=150)
    plt.close(figure)
    print(f"[*] Visualisasi prediksi disimpan: {path}")


def main():
    print("[*] Memuat dataset MNIST (akan diunduh otomatis jika belum tersedia)...")
    train_dataset = datasets.MNIST(root=DATA_DIR, train=True, download=True)
    test_dataset = datasets.MNIST(root=DATA_DIR, train=False, download=True)

    rng = np.random.default_rng(RANDOM_STATE)
    train_images, train_labels = select_samples(
        train_dataset.data.numpy(), train_dataset.targets.numpy(), TRAIN_SAMPLES, rng
    )
    test_images, test_labels = select_samples(
        test_dataset.data.numpy(), test_dataset.targets.numpy(), TEST_SAMPLES, rng
    )
    print(f"[*] Sampel yang dipakai: {len(train_images)} train, {len(test_images)} test")

    save_sample_images(train_images, train_labels)

    print("[*] Mengekstrak histogram intensitas...")
    histogram_train = extract_histogram_features(train_images)
    histogram_test = extract_histogram_features(test_images)

    print("[*] Menyiapkan fitur raw pixels...")
    pixel_train = train_images.reshape(len(train_images), -1).astype(np.float32) / 255.0
    pixel_test = test_images.reshape(len(test_images), -1).astype(np.float32) / 255.0

    histogram_model, histogram_predictions, histogram_accuracy = evaluate_classifier(
        "KNN dengan Histogram Intensitas",
        histogram_train,
        train_labels,
        histogram_test,
        test_labels,
    )
    pixel_model, pixel_predictions, pixel_accuracy = evaluate_classifier(
        "KNN dengan Raw Pixels",
        pixel_train,
        train_labels,
        pixel_test,
        test_labels,
    )

    save_comparison(
        histogram_accuracy,
        pixel_accuracy,
        test_labels,
        histogram_predictions,
        pixel_predictions,
    )
    predict_single_image(
        test_images[0],
        int(test_labels[0]),
        histogram_model,
        pixel_model,
    )


if __name__ == "__main__":
    main()
