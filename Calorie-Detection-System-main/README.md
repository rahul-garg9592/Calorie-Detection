# 🍎 Calorie Detection System

This project is a **Calorie Detection System** designed to identify and extract calorie values from food labels using a combination of deep learning and Optical Character Recognition (OCR) techniques. The application utilizes **YOLOv5** for object detection and **EasyOCR** for text recognition, offering a robust and efficient solution for real-time calorie information extraction.

## 📝 Project Motivation

With the increasing focus on health and nutrition, people are becoming more conscious about their daily calorie intake. However, manually checking the calorie information on food packaging can be time-consuming and inconvenient. This project aims to simplify this process by providing a **real-time solution** that automatically detects and reads the calorie values from food labels using an image.

## ⚙️ Key Features

- **Object Detection**: Uses **YOLOv5** to accurately detect the food label area on the packaging.
- **Text Recognition**: Employs **EasyOCR** to read the calorie information from the detected label.
- **User-Friendly Interface**: A web-based application built with **Flask** allows users to upload images and get instant calorie information.
- **Fallback OCR**: If no bounding box is detected, the system applies OCR on the entire image to ensure calorie values are extracted.

## 🧠 Methodology

1. **Preprocessing**: The input image is processed using techniques like grayscale conversion, Gaussian blur, and adaptive thresholding to enhance text visibility.
2. **Object Detection with YOLOv5**: The preprocessed image is passed through a trained YOLOv5 model to detect the bounding box around the calorie information on the label.
3. **Text Extraction with EasyOCR**: The detected region is then cropped and fed to EasyOCR for text recognition.
4. **Calorie Value Extraction**: A regular expression (regex) pattern is applied to identify and extract the calorie value from the recognized text.
5. **Result Display**: The extracted calorie value is displayed on the web interface for the user.

## 🛠️ Technologies Used

- **Python**: The core programming language used for development.
- **YOLOv5**: A state-of-the-art object detection model for detecting the label area on food packaging.
- **EasyOCR**: An OCR library for extracting text from images.
- **OpenCV**: For image preprocessing and manipulation.
- **Flask**: A lightweight web framework for creating the user interface.
- **HTML, CSS, JavaScript**: For building the frontend of the web application.

## 📊 Evaluation Metrics

- **Precision**: Measures the accuracy of the detected labels (TP / (TP + FP)).
- **Recall**: Measures the completeness of the detected labels (TP / (TP + FN)).
- **F1-Score**: Harmonic mean of precision and recall, providing a single performance metric.

## 📈 Results

The model achieves a **high precision and recall**, indicating its robustness in detecting calorie values across various types of food packaging.

## 🌟 Future Enhancements

- **Mobile App Integration**: Extend the functionality to a mobile app for on-the-go calorie detection.
- **Support for Multiple Languages**: Incorporate additional OCR languages to support a wider range of food labels.
- **Enhanced UI/UX**: Improve the user interface with better design and real-time feedback.


## 🧑‍🏫 Under the Supervision of

- **Dr. Raman Kumar Goyal**, Department of Computer Science and Engineering, Thapar University, Patiala

## 📚 Course Code

- **UML501: Machine Learning**

## 🏫 College

- **Thapar University, Patiala**
