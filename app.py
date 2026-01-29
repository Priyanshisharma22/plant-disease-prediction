import os
import zipfile
import numpy as np
import matplotlib.pyplot as plt
import itertools
import subprocess
import re

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import confusion_matrix, classification_report


# =====================================================
# CONFIG
# =====================================================
ZIP_PATH = r"C:\plant disease\archive.zip"
EXTRACT_DIR = r"C:\plant disease\dataset"
TEST_IMAGE_PATH = r"C:\plant disease\test_leaf.jpg"

IMG_SIZE = (224, 224)
BATCH_SIZE = 8
EPOCHS = 15

# ✅ fix saving error by using weights checkpoint only
BEST_WEIGHTS_PATH = "best_weights.weights.h5"
FINAL_MODEL_PATH = "plant_disease_model.h5"


# =====================================================
# CUDA CHECK (NVIDIA CUDA 13.0)
# =====================================================
def run_cmd(cmd):
    try:
        out = subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT, text=True)
        return out.strip()
    except Exception:
        return None


def print_cuda_info():
    print("\n================ CUDA INFO ================")
    cuda_path = os.environ.get("CUDA_PATH")
    print("CUDA_PATH:", cuda_path)

    nvcc_out = run_cmd("nvcc --version")
    print("\n--- nvcc --version ---")
    print(nvcc_out if nvcc_out else "❌ nvcc not found")

    if nvcc_out:
        m = re.search(r"release\s+([\d.]+)", nvcc_out)
        if m:
            print("✅ CUDA Toolkit Version:", m.group(1))

    smi_out = run_cmd("nvidia-smi")
    print("\n--- nvidia-smi ---")
    print(smi_out if smi_out else "❌ nvidia-smi not found")

    if smi_out:
        m = re.search(r"CUDA Version:\s*([\d.]+)", smi_out)
        if m:
            print("✅ NVIDIA Driver Supported CUDA Version:", m.group(1))

    print("===========================================\n")


# =====================================================
# TF GPU CHECK
# =====================================================
def setup_gpu():
    print("\n================ TF GPU INFO ================")
    print("TensorFlow Version:", tf.__version__)

    gpus = tf.config.list_physical_devices("GPU")
    print("GPUs detected by TensorFlow:", gpus)

    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("✅ GPU memory growth enabled.")
        except Exception as e:
            print("⚠ GPU memory growth error:", e)
    else:
        print("❌ TensorFlow cannot use GPU on this system.")
        print("➡ CUDA 13.0 exists, but TF (Windows pip) cannot use CUDA 13.")
    print("=============================================\n")


# =====================================================
# EXTRACT ZIP
# =====================================================
def extract_zip(zip_path, extract_dir):
    if not os.path.exists(zip_path):
        raise FileNotFoundError(f"ZIP not found: {zip_path}")

    os.makedirs(extract_dir, exist_ok=True)

    if len(os.listdir(extract_dir)) == 0:
        print(f"\n✅ Extracting dataset...\nFrom: {zip_path}\nTo:   {extract_dir}")
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(extract_dir)
        print("✅ Extraction complete!")
    else:
        print("\n✅ Dataset already extracted. Skipping extraction.")


# =====================================================
# FIND DATASET FOLDER (FIXED FOR PlantVillage nesting)
# =====================================================
def find_dataset_root(extract_dir):
    # 1) Most common expected path
    pv1 = os.path.join(extract_dir, "PlantVillage")
    if os.path.isdir(pv1):
        # if inside PlantVillage there is another PlantVillage folder
        pv2 = os.path.join(pv1, "PlantVillage")
        if os.path.isdir(pv2):
            return pv2
        return pv1

    # 2) Auto-detect fallback
    for root, dirs, _ in os.walk(extract_dir):
        img_class_dirs = []
        for d in dirs:
            full = os.path.join(root, d)
            if os.path.isdir(full):
                for f in os.listdir(full):
                    if f.lower().endswith((".jpg", ".jpeg", ".png")):
                        img_class_dirs.append(d)
                        break
        if len(img_class_dirs) >= 5:
            return root

    return None


# =====================================================
# PLOTS
# =====================================================
def plot_history(history):
    acc = history.history.get("accuracy", [])
    val_acc = history.history.get("val_accuracy", [])
    loss = history.history.get("loss", [])
    val_loss = history.history.get("val_loss", [])

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(acc, label="Train Accuracy")
    plt.plot(val_acc, label="Val Accuracy")
    plt.title("Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(loss, label="Train Loss")
    plt.plot(val_loss, label="Val Loss")
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(cm, classes):
    plt.figure(figsize=(10, 10))
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix")
    plt.colorbar()

    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(
            j, i, cm[i, j],
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black"
        )

    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.show()


# =====================================================
# MAIN
# =====================================================
def main():
    # 1) CUDA INFO
    print_cuda_info()

    # 2) GPU INFO
    setup_gpu()

    # 3) Dataset
    extract_zip(ZIP_PATH, EXTRACT_DIR)
    dataset_dir = find_dataset_root(EXTRACT_DIR)

    if dataset_dir is None:
        raise RuntimeError("❌ Dataset folder not found!")

    print("✅ Dataset path:", dataset_dir)

    # 4) Load Dataset
    print("\n✅ Loading dataset...")
    train_ds = tf.keras.utils.image_dataset_from_directory(
        dataset_dir,
        validation_split=0.2,
        subset="training",
        seed=42,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE
    )

    val_test_ds = tf.keras.utils.image_dataset_from_directory(
        dataset_dir,
        validation_split=0.2,
        subset="validation",
        seed=42,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE
    )

    class_names = train_ds.class_names
    num_classes = len(class_names)

    print("\n✅ Classes Found:", num_classes)
    print("✅ Class Names:", class_names)

    # 5) Split into Val & Test
    val_test_batches = tf.data.experimental.cardinality(val_test_ds).numpy()
    test_ds = val_test_ds.take(val_test_batches // 2)
    val_ds = val_test_ds.skip(val_test_batches // 2)

    # ✅ SAFE pipeline (no cache)
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.shuffle(1000).prefetch(AUTOTUNE)
    val_ds = val_ds.prefetch(AUTOTUNE)
    test_ds = test_ds.prefetch(AUTOTUNE)

    # 6) Model
    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
    ])

    print("\n✅ Building model...")
    base_model = MobileNetV2(
        input_shape=IMG_SIZE + (3,),
        include_top=False,
        weights="imagenet"
    )
    base_model.trainable = False

    inputs = layers.Input(shape=IMG_SIZE + (3,))
    x = data_augmentation(inputs)
    x = preprocess_input(x)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    model.summary()

    # ✅ FIXED callbacks (no keras save crash)
    callbacks = [
        EarlyStopping(patience=3, restore_best_weights=True),
        ModelCheckpoint(BEST_WEIGHTS_PATH, save_best_only=True, save_weights_only=True),
    ]

    # 7) Train
    device = "/GPU:0" if tf.config.list_physical_devices("GPU") else "/CPU:0"
    print("\n✅ Training device:", device)

    with tf.device(device):
        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=EPOCHS,
            callbacks=callbacks
        )

    # Load best weights
    if os.path.exists(BEST_WEIGHTS_PATH):
        model.load_weights(BEST_WEIGHTS_PATH)
        print("\n✅ Loaded best weights:", BEST_WEIGHTS_PATH)

    # Save final model
    model.save(FINAL_MODEL_PATH)
    print("\n✅ Final model saved:", FINAL_MODEL_PATH)

    plot_history(history)

    # 8) Evaluate
    print("\n✅ Evaluating on test set...")
    test_loss, test_acc = model.evaluate(test_ds)
    print("✅ Test Accuracy:", round(test_acc * 100, 2), "%")

    # 9) Confusion Matrix + Report
    print("\n✅ Confusion Matrix + Report...")

    y_true, y_pred = [], []
    for images, labels in test_ds:
        preds = model.predict(images, verbose=0)
        preds = np.argmax(preds, axis=1)
        y_true.extend(labels.numpy())
        y_pred.extend(preds)

    cm = confusion_matrix(y_true, y_pred)

    print("\n✅ Classification Report:\n")
    print(classification_report(y_true, y_pred, target_names=class_names))

    plot_confusion_matrix(cm, class_names)

    # 10) Predict one image
    print("\n✅ Predicting test image...")

    def predict_leaf(img_path):
        img = tf.keras.preprocessing.image.load_img(img_path, target_size=IMG_SIZE)
        x = tf.keras.preprocessing.image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)

        pred = model.predict(x, verbose=0)[0]
        idx = np.argmax(pred)
        return class_names[idx], float(pred[idx])

    if os.path.exists(TEST_IMAGE_PATH):
        label, conf = predict_leaf(TEST_IMAGE_PATH)
        print("✅ Prediction:", label)
        print("✅ Confidence:", round(conf * 100, 2), "%")
    else:
        print("⚠ Test image not found:", TEST_IMAGE_PATH)


if __name__ == "__main__":
    main()
