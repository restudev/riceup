from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import joblib
from skimage.feature import graycomatrix, graycoprops
from PIL import Image
import io
from io import BytesIO
import base64

app = Flask(__name__)

# Fungsi ekstraksi fitur GLCM
def extract_glcm_features(image_rgb):
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
    glcm = graycomatrix(gray, distances=[1, 2],
                        angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
                        levels=256, symmetric=True, normed=True)
    features = []
    properties = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']
    for prop in properties:
        features.extend(graycoprops(glcm, prop).ravel())
    return features

# Load model, scaler, dan label encoder
knn_model = joblib.load('model/knn_model.pkl')
scaler = joblib.load('model/scaler.pkl')
label_encoder = joblib.load('model/label_encoder.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    prediction_result = None
    image = None

    try:
        if 'image' in request.files:
            file = request.files['image']
            image = Image.open(file).convert('RGB')
        elif 'camera_image' in request.form:
            data_url = request.form['camera_image']
            header, encoded = data_url.split(",", 1)
            image_data = base64.b64decode(encoded)
            image = Image.open(BytesIO(image_data)).convert('RGB')
    except:
        return jsonify({'error': 'Gagal membaca gambar'}), 400

    if image:
        try:
            image_np = np.array(image)
            features = extract_glcm_features(image_np)
            features_scaled = scaler.transform([features])
            prediction = knn_model.predict(features_scaled)
            result = label_encoder.inverse_transform(prediction)[0]
            return jsonify({'prediction': result})
        except Exception as e:
            return jsonify({'error': 'Gagal memproses gambar'}), 500

    return jsonify({'error': 'Tidak ada gambar valid'}), 400

if __name__ == '__main__':
    app.run(debug=True)
