# **Brain Tumor MRI Classifier**

This repository contains the code for an end-to-end deep learning project that trains a model to classify brain tumor types from MRI scans and deploys it as an interactive web application using Streamlit. The link to test it out is mri-scan.streamlit.app



## 📋 Project Overview

<img src="Te-gl_0010.jpg">

The goal of this project is to build a robust classification model to distinguish between four categories of brain MRI:

*Glioma*

*Meningioma*

*Pituitary Tumor*

*No Tumor*

The project uses a transfer learning approach with a pre-trained EfficientNetB0 model and provides a simple web interface for real-time inference.

## 📖 Dataset

This model was trained on the Brain Tumor MRI Dataset from Kaggle.

**Total Images**: 7,023

**Training Set**: 5,712 images

**Test Set**: 1,311 images

**Classes**: glioma, meningioma, pituitary, notumor


## 🤖 Model Architecture

**Model**: EfficientNetB0 (pre-trained on ImageNet).

**Technique**: Transfer learning with fine-tuning.

**Classification Head**:

GlobalAveragePooling2D

Dense(128, activation='relu')

Dropout(0.5)

Dense(4, activation='softmax')

## 🚀 How to Run the App

*Clone the repository*:

```bash
git clone https://github.com/Jaykay73/MRI-Scan.git
cd MRI-Scan

```



*Install dependencies*:
It's recommended to use a virtual environment.

pip install -r requirements.txt



(You will need to create a requirements.txt file. See below.)

Download the Model:
Ensure your trained model file, efficientnet_best_model.keras, is in the root of the project directory.

Run the Streamlit app:
```bash
streamlit run app.py
```


The application will open in your browser at http://localhost:8501.

## 📦 Requirements

Create a requirements.txt file with the following content:

```bash

streamlit
tensorflow
numpy
Pillow  # PIL

```



