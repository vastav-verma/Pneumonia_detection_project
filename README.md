# Pneumonia_detection_project
This project is a deep learning-based web application that detects pneumonia from chest X-ray images using a Convolutional Neural Network (CNN) model trained from scratch.

🔍 Features
Custom CNN architecture trained on a labeled dataset (NORMAL vs PNEUMONIA)
Data augmentation and early stopping to reduce overfitting
Streamlit-based interactive web app for real-time predictions
Flask backend for deployment with HTML/CSS frontend
Displays helpful medical suggestions based on the prediction

🧠 Model Highlights
Input Size: 150x150 RGB images
Layers: Conv2D, MaxPooling2D, Flatten, Dense, Dropout
Loss Function: Binary Crossentropy
Optimizer: Adam
Metrics: Accuracy

📊 Evaluation
Classification Report and Confusion Matrix on test set

📦 Tech Stack
Python, TensorFlow, Keras
Streamlit & Flask
HTML, CSS
Matplotlib, NumPy, PIL

🚀 How to Use
Upload a chest X-ray image via the Streamlit web interface.
The model will predict if the patient has pneumonia or not.
Based on the result, health suggestions will be displayed.
