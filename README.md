# 🌿 AgroVision AI — Plant Disease Detector

<div align="center">

![AgroVision AI](https://img.shields.io/badge/AgroVision-AI-4ade80?style=for-the-badge&logo=leaflet&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Accuracy](https://img.shields.io/badge/Val%20Accuracy-94.99%25-22c55e?style=for-the-badge)

**AI-powered plant disease detection with complete treatment & prevention guidance**

[🚀 Live Demo](#deployment) · [📋 Features](#features) · [🛠️ Setup](#setup) · [📊 Model](#model)

</div>

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🔬 **Disease Detection** | Identifies 38 plant diseases across 14 crop species |
| 💊 **Treatment Steps** | Step-by-step treatment instructions for each disease |
| 🛡️ **Prevention Tips** | Disease-specific prevention recommendations |
| 🔍 **Symptom Analysis** | Detailed symptom descriptions for each condition |
| 📊 **Confidence Score** | AI confidence meter with top-3 predictions |
| 🎨 **Premium UI** | Dark glassmorphism design with smooth animations |

---

## 🌱 Supported Crops & Diseases

| Crop | Diseases Detected |
|------|-------------------|
| 🍎 Apple | Apple Scab, Black Rot, Cedar Apple Rust, Healthy |
| 🌽 Corn (Maize) | Gray Leaf Spot, Common Rust, Northern Leaf Blight, Healthy |
| 🍇 Grape | Black Rot, Esca (Black Measles), Leaf Blight, Healthy |
| 🍅 Tomato | Bacterial Spot, Early Blight, Late Blight, Leaf Mold, Septoria Leaf Spot, Spider Mites, Target Spot, Yellow Leaf Curl Virus, Mosaic Virus, Healthy |
| 🥔 Potato | Early Blight, Late Blight, Healthy |
| 🍑 Peach | Bacterial Spot, Healthy |
| 🍒 Cherry | Powdery Mildew, Healthy |
| 🍊 Orange | Citrus Greening (HLB) |
| 🫑 Bell Pepper | Bacterial Spot, Healthy |
| 🍓 Strawberry | Leaf Scorch, Healthy |
| 🫐 Blueberry | Healthy |
| 🫐 Raspberry | Healthy |
| 🌿 Soybean | Healthy |
| 🥦 Squash | Powdery Mildew |

---

## 🤖 Model

- **Architecture**: EfficientNet-B0 (Transfer Learning)
- **Dataset**: PlantVillage (~87,000 images)
- **Training Accuracy**: 97.8%
- **Validation Accuracy**: 94.99%
- **Classes**: 38 (14 crops, various disease conditions)
- **Input Size**: 512×512 pixels
- **Epochs**: 50

---

## 🚀 Setup

### Prerequisites
- Python 3.9+
- pip

### Installation

```bash
# Clone the repository
git clone https://github.com/saqib54/AgroVision-AI.git
cd AgroVision-AI

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app/app.py
```

The app will open at `http://localhost:8501`

### Model File
The model file (`plant_disease_model.h5`) is required. Place it in the root directory or set the `MODEL_PATH` environment variable:

```bash
export MODEL_PATH=/path/to/your/plant_disease_model.h5
```

---

## 🌐 Deployment

### Streamlit Cloud (Recommended — Free)

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub account
4. Select this repo and set **Main file path** to `app/app.py`
5. Click **Deploy**

> **Note**: Upload the `plant_disease_model.h5` model file to the repo or use Git LFS for large files.

---

## 📁 Project Structure

```
AgroVision-AI/
├── app/
│   └── app.py              # Main Streamlit application
├── src/
│   ├── model.py            # Model architecture
│   ├── predict.py          # Prediction utilities
│   ├── preprocess.py       # Image preprocessing
│   └── train.py            # Training script
├── notebook/               # Jupyter notebooks
├── .streamlit/
│   └── config.toml         # Streamlit configuration
├── plant_disease_model.h5  # Trained model (required)
├── requirements.txt        # Python dependencies
└── README.md
```

---

## 🎨 UI Preview

The app features:
- 🌑 **Dark glassmorphism** design
- 🟢 **Color-coded severity** indicators (Healthy → Critical)
- 📊 **AI confidence bar** with animated fill
- 💊 **Numbered treatment steps** with highlighted cards
- 🛡️ **Prevention tips** with visual bullets
- 📱 **Responsive layout** for mobile and desktop

---

## 🤝 Contributing

Pull requests are welcome! For major changes, please open an issue first.

---

## 📄 License

MIT License — feel free to use and modify.

---

<div align="center">
Made with ❤️ using TensorFlow &amp; Streamlit
</div>
