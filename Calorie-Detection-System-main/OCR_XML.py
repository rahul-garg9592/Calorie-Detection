import sys
import os
import torch
import cv2
import pytesseract
import easyocr
import numpy as np
import matplotlib.pyplot as plt
from flask import Flask, render_template, request, jsonify
yolov5_code_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'yolov5'))
utils_path = os.path.join(yolov5_code_path, 'utils')
sys.path.insert(0, yolov5_code_path)
sys.path.insert(0, utils_path)
from yolov5.utils.dataloaders import letterbox
from yolov5.utils.general import TryExcept, emojis
import re

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploaded_images'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Import the YOLOv5 modules
from models.common import DetectMultiBackend # type: ignore
# Load the YOLOv5 model using DetectMultiBackend
model_path = os.path.join(yolov5_code_path, 'runs/train/calories_detection_fresh3/weights/best.pt')
model = DetectMultiBackend(model_path)
print("YOLOv5 model loaded successfully!")

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'])
print("EasyOCR initialized successfully!")

# Preprocess image for better OCR accuracy
def preprocess_image(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply adaptive thresholding for better text visibility
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    
    # Sharpen the image
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    sharpened = cv2.filter2D(thresh, -1, kernel)
    
    return sharpened

# Function to extract calorie value using regex
def extract_calorie_value(text):
    # Enhanced regex pattern to capture different variations
    pattern = (
        r'(?:Calories|Calorie|CALORIES|calories|Cal|CAL|Energy|ENERGY|energy)\s*[:\-]?\s*(\d+)'  # Matches "Calories 200", "Energy 150", etc.
        r'|\b(\d+)\s*(Cal|CAL|kcal|KCAL|KJ|kj|kjoule|kilojoule)'  # Matches "200 Cal", "150 kcal", "300 KJ"
    )

    # Search for the pattern in the text
    match = re.search(pattern, text, re.IGNORECASE)
    if match:
        # Return the first matched group with a numeric value
        return match.group(1) if match.group(1) else match.group(2)

    return "Calorie or energy value not detected"


# Main function to perform detection and OCR
def detect_calories(image_path):

    # Load the image using OpenCV
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to load image from {image_path}")
        return
    
    # Preprocess the image for OCR
    processed_image = preprocess_image(image)

    # Resize the image using letterbox to maintain aspect ratio
    resized_image = letterbox(image, new_shape=(640, 640), auto=False)[0]

    # Convert the image to a tensor
    image_tensor = torch.from_numpy(resized_image).permute(2, 0, 1).unsqueeze(0).float() / 255.0

    # Ensure the tensor is on the correct device (CPU in this case)
    image_tensor = image_tensor.to(torch.device("cpu"))

    # Perform YOLOv5 inference with the preprocessed image tensor
    results = model(image_tensor)
    detections = results[0].cpu().numpy()
    if len(detections) > 0:
        print("YOLOv5 detected bounding boxes.")
        for detection in detections:
            # Flatten the detection array to ensure it is 1D
            detection = detection.flatten()

            # Extract bounding box coordinates and other details
            xmin, ymin, xmax, ymax = map(int, detection[:4])
            confidence = float(detection[4])
            class_id = int(detection[5])

            # Print the extracted values for debugging
            print(f"Detection: xmin={xmin}, ymin={ymin}, xmax={xmax}, ymax={ymax}, confidence={confidence}, class_id={class_id}")

            # Crop the detected region
            cropped_image = image[ymin:ymax, xmin:xmax]
            cropped_image = preprocess_image(cropped_image)

            # Apply OCR using EasyOCR
            ocr_results = reader.readtext(cropped_image)
            for (_, text, ocr_confidence) in ocr_results:
                calorie_value = extract_calorie_value(text)
                if calorie_value:
                    print(f"Detected Calorie Value: {calorie_value} (Confidence: {ocr_confidence})")
                    return calorie_value



    calorie_value = None

    # If YOLOv5 detects any object, apply OCR to the detected region
    if detections.size == 0:
        print("YOLOv5 detected bounding boxes.")
        for index, row in detections.iterrows():
            xmin, ymin, xmax, ymax = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
            
            # Crop the detected region
            cropped_image = image[ymin:ymax, xmin:xmax]
            cropped_image = preprocess_image(cropped_image)
            
            # Apply OCR using EasyOCR
            ocr_results = reader.readtext(cropped_image)
            for (_, text, confidence) in ocr_results:
                calorie_value = extract_calorie_value(text)
                if calorie_value:
                    print(f"Detected Calorie Value: {calorie_value} (Confidence: {confidence})")
                    return calorie_value

    # If no detection from YOLOv5, apply OCR on the entire image
    if not calorie_value:
        print("No bounding box detected. Using OCR fallback on the entire image.")
        text = pytesseract.image_to_string(processed_image, config='--psm 6')
        calorie_value = extract_calorie_value(text)
        print(f"OCR Fallback Detected Calorie Value: {calorie_value}")

    return calorie_value
# Test the function with a sample image


# Flask route for the web interface
@app.route('/')
def index():
    return render_template('index.html')

# Flask route for handling image upload and calorie detection
@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)

    calorie_value = detect_calories(file_path)
    return jsonify({'calorie_value': calorie_value})


def process_main_folder(input_folder, output_file):
    # Check if the input folder exists
    if not os.path.exists(input_folder):
        print(f"Error: Input folder not found: {input_folder}")
        return

    # Create or overwrite the output file
    with open(output_file, 'w') as file:
        # List all files in the main folder
        for filename in os.listdir(input_folder):
            image_path = os.path.join(input_folder, filename)

            # Check if it is a file (not a subfolder) and an image
            if os.path.isfile(image_path) and filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                print(f"Processing: {image_path}")

                # Detect calorie value for the current image
                calorie_value = detect_calories(image_path)

                # Write the result to the output file
                file.write(f"{filename}: {calorie_value}\n")
                print(f"Detected Calorie Value for {filename}: {calorie_value}")

    print(f"Results saved to {output_file}")

# Main block to process only the main images folder
# if __name__ == "__main__":
#     detect_calories("../test_images/img0001.png")

if __name__ == '__main__':
    app.run(debug=True)




