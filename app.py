# -*- coding: utf-8 -*-
"""
Created on Fri Sep 19 12:40:02 2025

@author: ambag
"""

from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import cv2

app = Flask(__name__)

# Model load
model = tf.keras.models.load_model("cnn8grps_rad1_model.h5")

# Labels (adjust according to your training)
labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']

@app.route("/")
def home():
    return "Sign Language Recognition API is running!"

@app.route("/predict", methods=["POST"])
def predict():
    file = request.files['file']
    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    img = cv2.resize(img, (64, 64))  # resize as per training
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    pred = model.predict(img)
    class_id = np.argmax(pred)
    return jsonify({"prediction": labels[class_id]})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
