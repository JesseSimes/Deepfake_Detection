import os
import cv2
from tensorflow.keras.models import load_model
import numpy as np
import tkinter as tk
from tkinter import filedialog

# Define model path
model_path = r"E:\Deepfake detection\Updated model\deepfake_detection_updated_v2.h5"

# Check if model file exists before loading
if not os.path.exists(model_path):
    print("Model file not found! Check the directory where it was saved.")
    exit()

print("Model file found! Loading model...")
model = load_model(model_path)
print("Model loaded successfully!")

# Function to preprocess the image
def preprocess_image(image_path):
    try:
        img = cv2.imread(image_path)
        if img is None:
            print("Error: Unable to read the image file!")
            return None
        img = cv2.resize(img, (128, 128))  # Resize to match model input
        img = img.astype('float32') / 255.0  # Normalize pixel values
        img = np.expand_dims(img, axis=0)  # Add batch dimension
        return img
    except Exception as e:
        print(f"Error processing image: {e}")
        return None

# Open file dialog for image selection
root = tk.Tk()
root.withdraw()  # Hide Tkinter main window
image_path = filedialog.askopenfilename(title="Select an Image", filetypes=[("Image Files", "*.jpg;*.png;*.jpeg")])

# Check if user selected an image
if not image_path:
    print("No image selected!")
    exit()

print(f"Selected Image: {image_path}")

# Preprocess the image
processed_img = preprocess_image(image_path)

# Ensure image processing was successful
if processed_img is None:
    print("Image preprocessing failed! Please try another image.")
    exit()

# Predict using the model
prediction = model.predict(processed_img)[0][0]  # Extract single value

# Interpret the result (Ensure label assignment is correct)
if prediction > 0.5:
    print("Fake Image Detected!")
else:
    print("Real Image!")