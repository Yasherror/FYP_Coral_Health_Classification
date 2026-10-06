# 🪸 Coral Health Classification


A Streamlit dashboard for classifying coral health status using deep learning. This Final Year Project (FYP) applies image classification techniques to assess coral reef health from uploaded images, providing researchers and conservationists with a fast, automated monitoring tool.

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Usage](#-usage)
- [Dataset](#-dataset)
- [Model](#-model)
- [Results](#-results)
- [Live Demo](#-live-demo)
- [Author](#-author)
- [Acknowledgements](#-acknowledgements)
- [License](#-license)

---

## 🔍 Overview

Coral reefs are critical to marine ecosystems but are increasingly threatened by climate change, pollution, and disease. This project provides an automated tool to classify coral health from images, helping researchers and conservationists monitor reef conditions more efficiently.

The system uses multiple pretrained deep learning models to analyze uploaded coral images and predict their health status across three categories: **Healthy**, **Compromised**, and **Dead**.

---

## ✨ Features

- 📤 Upload coral images through an interactive Streamlit interface
- 🤖 Real-time classification with confidence scores
- 🧠 Multiple model architectures (ResNet50, EfficientNet, VGG16, MobileNet-V3)
- 📊 Visual dashboard showing prediction breakdowns
- 🖼️ Image preview before and after classification
- ⚡ Fast inference pipeline optimized for real-time use
- 📈 Preprocessed dataset pipeline for consistent inputs

---

## 🛠️ Tech Stack

| Category | Technology |
|----------|-----------|
| Frontend | Streamlit |
| Language | Python 3.9+ |
| ML Framework | PyTorch |
| Model Architectures | ResNet50, EfficientNet, VGG16, MobileNet-V3 |
| Image Processing | OpenCV, Pillow |
| Data Handling | NumPy, Pandas |
| Visualization | Matplotlib, Plotly |

---

## 📁 Project Structure

```
FYP_Coral_Health_Classification/
├── src/
│   ├── app.py                              # Main Streamlit application
│   ├── requirements.txt                    # Python dependencies
│   ├── components/
│   │   ├── data/                           # Data files (large ZIP excluded)
│   │   └── ...                             # Other UI components
│   ├── models/                             # Trained model weights (.pth)
│   ├── utils/                              # Helper functions & preprocessing
│   └── assets/                             # Images, icons, styles
├── .gitignore
└── README.md
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.9 or higher
- pip package manager
- Git

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Yasherror/FYP_Coral_Health_Classification.git
   cd FYP_Coral_Health_Classification
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv

   # Windows
   venv\Scripts\activate

   # macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Streamlit app**
   ```bash
   streamlit run app.py
   ```

5. **Open your browser** at `http://localhost:8501`

---

## 🎯 Usage

1. Launch the Streamlit dashboard using the command above
2. Upload a coral image (JPG/PNG) using the file uploader
3. Select the model architecture you want to use *(if applicable)*
4. Wait for the model to process the image
5. View the predicted health class and confidence score
6. Explore the visualizations for deeper insights

---

## 📊 Dataset

The processed coral image dataset is **not included** in this repository due to GitHub's file size limits (the ZIP file exceeds 100 MB).

| Property | Details |
|----------|---------|
| **Source** | [CoralConditionDataset](https://github.com/XL-SHAO/CoralConditionDataset) |
| **Total Images** | 19,974 |
| **Classes** | Healthy, Compromised, Dead |
| **Input Size** | 224 × 224 × 3 |

To reproduce the training pipeline, download the dataset from the source above and run the preprocessing scripts in `src/utils/`.

---

## 🧠 Model

Four pretrained architectures were trained and evaluated for this project. The best-performing model is used in the deployed dashboard.

| Property | Details |
|----------|---------|
| **Architectures Tested** | ResNet50, EfficientNet, VGG16, MobileNet-V3 |
| **Training Framework** | PyTorch |
| **Input Size** | 224 × 224 × 3 |
| **Number of Classes** | 3 (Healthy, Compromised, Dead) |
| **Best Accuracy** | **86%** |
| **Transfer Learning** | Yes (ImageNet pretrained weights) |

### Training Highlights

- Transfer learning with ImageNet-pretrained weights
- Data augmentation (flips, rotations, color jitter)
- Stratified train/validation/test split
- Early stopping and learning rate scheduling

---

## 📈 Results

| Model | Accuracy |
|-------|----------|
| ResNet50 | *[fill in]* |
| EfficientNet | *[fill in]* |
| VGG16 | *[fill in]* |
| MobileNet-V3 | *[fill in]* |
| **Best Model** | **86%** |

*[Add a confusion matrix or sample predictions image here for extra polish]*

---

## 🌐 Live Demo

🔗 *[Add your Streamlit Cloud link here after deployment]*

---

## 👤 Author

**Yashreen**
- GitHub: [@Yasherror](https://github.com/Yasherror)
- Institution: *Asia Pacific University of Technology & Innovation (APU)*

---

## 🙏 Acknowledgements

- [CoralConditionDataset](https://github.com/XL-SHAO/CoralConditionDataset) by XL-SHAO for providing the coral image dataset
- *Asia Pacific University of Technology & Innovation (APU)*
- The open-source PyTorch and Streamlit communities

---

## 📄 License

This project is developed as part of an academic Final Year Project. All rights reserved.

---

⭐ If you find this project useful, please consider giving it a star!
