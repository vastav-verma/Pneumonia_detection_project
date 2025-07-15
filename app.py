from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os

app = Flask(__name__)
model = load_model("pneumonia_model.h5")  # Ensure model.h5 is in your project folder

IMG_SIZE = (150, 150)

def preprocess_image(image_path):
    img = load_img(image_path, target_size=IMG_SIZE)
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    return img / 255.0

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return "No image uploaded", 400
    file = request.files['image']
    if file.filename == '':
        return "No selected file", 400

    filepath = os.path.join('static', file.filename)
    file.save(filepath)

    img = preprocess_image(filepath)
    pred = model.predict(img)[0][0]

    if pred > 0.5:
        prediction = "Pneumonia"
        advice = """
        <ul>
            <li>Consult a doctor immediately for proper diagnosis and treatment.</li>
            <li>Take prescribed antibiotics or antivirals as directed.</li>
            <li>Stay hydrated and rest as much as possible.</li>
            <li>Avoid smoking and second-hand smoke.</li>
            <li>Use a humidifier to keep your airways moist.</li>
        </ul>
        """
    else:
        prediction = "Normal"
        advice = "<p>Your chest X-ray appears normal. Maintain a healthy lifestyle and regular checkups!</p>"

    return render_template('index.html', prediction=prediction, image_path=filepath, advice=advice)

if __name__ == '__main__':
    app.run(debug=True)
