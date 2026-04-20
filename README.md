<div align="center">
  <h1>🧠 Fetal Brain Abnormality Detection using Ultrasound Images</h1>
  <p><strong>Deep Learning Application for Fetal Brain Anomaly Detection</strong></p>

  <p>
    <img alt="Python" src="https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54" />
    <img alt="Flask" src="https://img.shields.io/badge/flask-%23000.svg?style=for-the-badge&logo=flask&logoColor=white" />
    <img alt="TensorFlow" src="https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white" />
    <img alt="Keras" src="https://img.shields.io/badge/Keras-%23D00000.svg?style=for-the-badge&logo=Keras&logoColor=white" />
    <img alt="HTML5" src="https://img.shields.io/badge/html5-%23E34F26.svg?style=for-the-badge&logo=html5&logoColor=white" />
    <img alt="CSS3" src="https://img.shields.io/badge/css3-%231572B6.svg?style=for-the-badge&logo=css3&logoColor=white" />
  </p>
</div>

---

## 📖 Overview

The **Fetal Brain Abnormality Detection using Ultrasound Images** application leverages deep learning architectures to analyze and classify fetal brain ultrasound images into **16 distinct categories**. Designed with a sophisticated, glassmorphism-inspired dark mode interface, the application bridges the gap between advanced medical AI and intuitive user experience. 

It provides detailed predictions, confidence scores, model comparative analysis, and downloadable clinical reports to assist in automated fetal anomaly detection.

---

## ✨ Key Features

- **🖼️ High-Fidelity Image Processing:** Upload high-resolution diagnostic fetal brain ultrasound images for robust classification.
- **🧠 Advanced Deep Learning Architectures:** Integrated with multiple pre-trained cutting-edge CNN algorithms for accurate prediction (e.g., Xception, Separable CNNs).
- **🔒 Secure Authentication:** A streamlined, Firebase-driven user authentication flow (Sign-In / Sign-Up) protecting patient data and workflows.
- **📈 Comprehensive Visual Insights:** Dynamic radial progress rings for class probabilities, interactive data visualizations, and comparative model analysis metrics.
- **📄 Extensible Report Generation:** Seamlessly generate and download comprehensive diagnostic outcome reports based on your analysis.

---

## 🛠️ Technology Stack

| Architecture Layer | Tools & Technologies |
| :--- | :--- |
| **Frontend UI/UX** | HTML5, CSS3, Vanilla JavaScript, FontAwesome |
| **Styling Concept** | Glassmorphism, CSS Variables, Ambient Gradients |
| **Backend API** | Python, Flask, Werkzeug |
| **AI/Deep Learning** | TensorFlow, Keras, NumPy |
| **Image Pipeline** | OpenCV (Headless), Pillow (PIL) |
| **Auth & Routing** | Session-based Auth, Multi-view Application Logic |

---

## 🚀 Getting Started

Follow these steps to deploy and run the classification engine locally.

### 1. Clone the Repository
```bash
git clone https://github.com/appu-ui/Fetal-brain-abnormality-detection-using-ultrasound-images.git
cd Fetal-brain-abnormality-detection-using-ultrasound-images
```

### 2. Environment Setup
It is recommended to run this project in an isolated virtual environment.
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

### 3. Install Required Dependencies
Ensure you have Python 3.8+ installed, then run:
```bash
pip install -r requirements.txt
```
*(Requires `Flask`, `tensorflow`, `Pillow`, `opencv-python-headless`, and `numpy`).*

### 4. Run the Application Server
Start the Flask application using:
```bash
python app.py
```
By default, the server will be available at: `http://localhost:5000`

---

## 💡 Usage Guide

1. **Authentication:** Create an account or log in to the secure dashboard.
2. **Setup Analysis:** 
    - Click **"Launch Application"** to enter the workspace.
    - Drag and drop or browse to select a fetal brain ultrasound image (`.jpg`, `.png`, `.jpeg`).
    - Select your preferred backend **Deep Learning Model** from the dropdown menu.
3. **Execution:** Click **"Analyze Image"**. The Flask backend will map the image tensor through the specified model layer weights and return the predictive probabilities.
4. **Insights:** Review the categorized findings, dominant confidence metrics via radial gauges, and clinical metadata on the result panel.

---

## 📜 Repository Structure

```tree
├── app.py                # Main Flask Server & Route Controller
├── util.py               # Image processing and Model loading utilities
├── requirements.txt      # Python dependencies
├── README.md             # Project documentation
├── models/               # Contains pre-trained TensorFlow/Keras .h5 files
├── static/
│   ├── main.css          # Core UI styling (Glassmorphism & animations)
│   ├── auth.css          # Authentication screen specific styles
│   └── uploads/          # Temporary storage for uploaded images
└── templates/            # HTML templates
    ├── index.html        # Main prediction dashboard UI
    └── auth.html         # Login / Registration screens
```

---

<div align="center">
  <sub>Built with ❤️ for Medical AI Advancement.</sub>
</div>
