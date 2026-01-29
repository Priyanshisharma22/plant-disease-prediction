import os
import numpy as np
import tensorflow as tf
from PIL import Image

# ===============================
# CONFIG (CHANGE THESE)
# ===============================
MODEL_PATH = r"C:\plant disease\plant_disease_model.h5"
IMAGE_PATH = r"C:\plant disease\de.webp"   # ✅ your test image

IMG_SIZE = (224, 224)


# ===============================
# Helper: convert webp to jpg
# ===============================
def convert_webp_to_jpg(webp_path):
    jpg_path = webp_path.rsplit(".", 1)[0] + ".jpg"
    img = Image.open(webp_path).convert("RGB")
    img.save(jpg_path, "JPEG")
    return jpg_path


# ===============================
# Load model
# ===============================
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"❌ Model not found: {MODEL_PATH}")

model = tf.keras.models.load_model(MODEL_PATH)
print("✅ Model loaded successfully!")


# ===============================
# Get class names from model (if available)
# ===============================
# If you saved class names separately, load them.
# Otherwise, here we manually define them same as dataset folder.
CLASS_NAMES = [
    "Pepper__bell___Bacterial_spot",
    "Pepper__bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Tomato_Bacterial_spot",
    "Tomato_Early_blight",
    "Tomato_Late_blight",
    "Tomato_Leaf_Mold",
    "Tomato_Septoria_leaf_spot",
    "Tomato_Spider_mites_Two_spotted_spider_mite",
    "Tomato__Target_Spot",
    "Tomato__Tomato_YellowLeaf__Curl_Virus",
    "Tomato__Tomato_mosaic_virus",
    "Tomato_healthy"
]

# ⚠ If your dataset had 16 classes (including extra), adjust here
# Print model output classes:
print("✅ Model output units:", model.output_shape[-1])


# ===============================
# Prediction Function
# ===============================
def predict_image(img_path):
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"❌ Image not found: {img_path}")

    # ✅ Convert WEBP to JPG
    if img_path.lower().endswith(".webp"):
        print("⚠ WEBP image detected. Converting to JPG...")
        img_path = convert_webp_to_jpg(img_path)
        print("✅ Converted:", img_path)

    # Load and preprocess image
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=IMG_SIZE)
    x = tf.keras.preprocessing.image.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    # MobileNetV2 preprocess
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)

    # Predict
    probs = model.predict(x, verbose=0)[0]
    idx = int(np.argmax(probs))
    conf = float(probs[idx])

    # label
    if idx < len(CLASS_NAMES):
        label = CLASS_NAMES[idx]
    else:
        label = f"Class_{idx}"

    return label, conf


# ===============================
# Run Prediction
# ===============================
label, confidence = predict_image(IMAGE_PATH)

print("\n✅ Prediction Result")
print("Predicted Class:", label)
print("Confidence:", round(confidence * 100, 2), "%")
