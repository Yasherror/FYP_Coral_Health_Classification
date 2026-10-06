# 🪸 Coral Health Classification

A Streamlit dashboard for classifying coral health status using machine learning.

## 📋 Overview

This project is a Final Year Project (FYP) that uses image classification techniques to assess coral reef health. The dashboard provides an interactive interface for uploading coral images and receiving health predictions.

## ✨ Features

- Upload coral images for classification
- Real-time prediction with confidence scores
- Visual dashboard for coral health analytics
- Preprocessed dataset pipeline

## 🛠️ Tech Stack

- **Frontend:** Streamlit
- **Language:** Python
- **ML Framework:** PyTorch **
- **Data Processing:** NumPy, Pandas, OpenCV

## 📁 Project Structure

## 🚀 Getting Started

### Prerequisites

- Python 3.9+
- pip

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Yasherror/FYP_Coral_Health_Classification.git
   cd FYP_Coral_Health_Classification


   Create a virtual environment (recommended)

bash
python -m venv venv

# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
Install dependencies

bash
pip install -r requirements.txt
Run the Streamlit app

bash
streamlit run app.py
Open your browser at http://localhost:8501

🎯 Usage
Launch the Streamlit dashboard using the command above

Upload a coral image (JPG/PNG) using the file uploader

Wait for the model to process the image

View the predicted health class and confidence score

Explore the visualizations for more insights

📊 Dataset
The processed coral image dataset is not included in this repository due to GitHub's file size limits (the ZIP file exceeds 100 MB).

Source: https://github.com/XL-SHAO/CoralConditionDataset

Size: 19974

Classes: Healthy, Compromised, Dead]


🧠 Model
Architecture: ResNet50, EfficientNet, VGG16, MobileNet-V3

Training Framework: Pytorch

Accuracy: 86%

Input Size: 224x224x3

Classes: 3 classes
