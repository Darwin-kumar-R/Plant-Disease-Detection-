from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
import io

app = Flask(__name__)

# Load the model
model = tf.keras.models.load_model("keras_model.h5", compile=False)

# Load labels
class_names = open("labels.txt", "r").readlines()

# Preprocess the image
def preprocess_image(image):
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    return normalized_image_array.reshape(1, 224, 224, 3)

@app.route("/predict", methods=["POST"])
def predict():
    # Debug: Check if request contains image
    if "image" not in request.files:
        print("🚨 No image found in request!")
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    print(f"✅ Received image: {file.filename}")

    # Open and preprocess image
    image = Image.open(io.BytesIO(file.read())).convert("RGB")
    processed_image = preprocess_image(image)

    # Make prediction
    prediction = model.predict(processed_image)
    index = np.argmax(prediction)

    response = {
        "class": class_names[index].strip(),
        "confidence": float(prediction[0][index])
    }

    print(f"✅ Prediction: {response}")  # Debugging output

    return jsonify(response)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)