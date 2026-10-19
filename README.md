# 🌿 AgroVision AI — Plant Disease Detector

<div align="center">

![AgroVision AI](https://img.shields.io/badge/AgroVision-AI-4ade80?style=for-the-badge&logo=leaflet&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.19-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.51-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Accuracy](https://img.shields.io/badge/Val%20Accuracy-94.99%25-22c55e?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-blue?style=for-the-badge)

**AI-powered plant disease detection with complete treatment & prevention guidance**

[🚀 Live Demo](#deployment) · [📋 Features](#-features) · [🛠️ Setup](#-setup--installation) · [🤖 AI Model](#-ai-methodology--model) · [👥 Team](#-team-contributions)

</div>

---
## screenshoots
<img width="1265" height="611" alt="Screenshot 2026-10-19 170912" src="https://github.com/user-attachments/assets/c25cff2b-0245-40c0-94d0-4bb5a1c7560c" />
<img width="1266" height="555" alt="Screenshot 2026-10-19 170928" src="https://github.com/user-attachments/assets/b1ed557a-5abd-48e3-8112-982a19bad748" />
<img width="1157" height="545" alt="Screenshot 2026-10-19 170940" src="https://github.com/user-attachments/assets/6b502bbb-9e33-4719-8b02-9e06ede40363" />
<img width="897" height="546" alt="Screenshot 2026-10-19 170956" src="https://github.com/user-attachments/assets/c19a854f-787a-4121-b71c-daf7a786d75d" />
<img width="1151" height="536" alt="Screenshot 2026-10-19 171017" src="https://github.com/user-attachments/assets/9ec6e156-c3b9-455c-a80a-48cee5b67871" />

## 🧩 Problem Statement & Solution

<table>
<tr>
<td width="50%">

### ❗ The Problem

> Every year, **plant diseases destroy 20–40% of global crop yield**, costing farmers billions. In developing countries, most smallholder farmers:
> - **Cannot afford** expert agronomists
> - **Lack access** to timely disease diagnosis
> - **Lose entire harvests** due to late or wrong treatment
> - Rely on **guesswork** when identifying diseases

This leads to food insecurity, financial ruin, and unnecessary pesticide overuse.

</td>
<td width="50%">

### ✅ Our Solution — AgroVision AI

> A **free, instant, AI-powered** plant disease diagnostic tool that:
> - Detects **38 plant diseases** across **14 crop species** from a single photo
> - Provides **step-by-step treatment plans** in seconds
> - Offers **prevention strategies** to protect future crops
> - Works on **any device** with a browser — no installation needed
> - Achieves **94.99% validation accuracy** using deep learning

**Empowering every farmer with a plant doctor in their pocket.** 🌾

</td>
</tr>
</table>

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🔬 **Disease Detection** | Identifies 38 plant diseases across 14 crop species |
| 💊 **Treatment Steps** | Step-by-step treatment instructions for each disease |
| 🛡️ **Prevention Tips** | Disease-specific prevention recommendations |
| 🔍 **Symptom Analysis** | Detailed symptom descriptions for each condition |
| 📊 **Confidence Score** | AI confidence meter with top-3 predictions |
| 🌍 **Multi-language** | Severity labels in English & Urdu |
| 🎨 **Premium UI** | Dark glassmorphism design with smooth animations |
| 📱 **Responsive** | Works on mobile, tablet, and desktop |

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

## 🛠️ Technology Stack

<div align="center">

| Layer | Technology | Purpose |
|-------|-----------|---------|
| 🧠 **AI / ML** | TensorFlow 2.19 + tf-Keras 2.19 | Model training & inference |
| 🏗️ **Model Architecture** | EfficientNet-B0 (Transfer Learning) | Image classification backbone |
| 🖼️ **Image Processing** | Pillow (PIL) + NumPy | Preprocessing & array ops |
| 🌐 **Web Framework** | Streamlit 1.51 | Interactive UI & deployment |
| 📦 **Data** | PlantVillage Dataset (~87,000 images) | Model training data |
| 🐍 **Language** | Python 3.11 | Core programming language |
| 💾 **Model Format** | HDF5 (.h5) | Saved model weights |
| 🎨 **Styling** | Custom CSS + Google Fonts (Inter) | Glassmorphism dark UI |

</div>

### Dependency Versions

```
tensorflow-cpu==2.19.0
tf-keras==2.19.0
protobuf>=5.29.6,<6.0.0
streamlit>=1.28.0
numpy>=1.23.5,<2.0
Pillow>=9.5.0
h5py>=3.9.0
```

---

## 🚀 Setup & Installation

### Prerequisites

- **Python 3.9+** (tested on Python 3.11)
- **pip** package manager
- **~200MB** disk space (for dependencies)
- **plant_disease_model.h5** model file (16.6 MB)

### Step-by-Step Installation

```bash
# 1. Clone the repository
git clone https://github.com/saqib54/AgroVision-AI.git
cd AgroVision-AI

# 2. (Recommended) Create a virtual environment
python -m venv .venv

# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
streamlit run streamlit_app.py
```

The app will open automatically at **`http://localhost:8501`**

### Model File Setup

The trained model (`plant_disease_model.h5`) must be in the **root directory**:

```
AgroVision-AI/
└── plant_disease_model.h5   ← place here
```

Or set a custom path via environment variable:

```bash
# Windows (PowerShell)
$env:MODEL_PATH = "C:\path\to\plant_disease_model.h5"

# macOS/Linux
export MODEL_PATH=/path/to/plant_disease_model.h5
```

### Troubleshooting

| Error | Fix |
|-------|-----|
| `VersionError: Protobuf` mismatch | `pip install "protobuf>=5.29.6,<6.0.0" "tf-keras==2.19.0"` |
| `Port 8501 already in use` | Run `streamlit run streamlit_app.py --server.port 8502` |
| `Model file not found` | Ensure `plant_disease_model.h5` is in the root directory |
| `Module not found` | Re-run `pip install -r requirements.txt` |

---

## 🤖 AI Methodology & Model

### Architecture Overview

```
Input Image (any size)
        ↓
  Image Preprocessing
  • Resize → 512×512 px
  • Normalize → [0, 1]
  • Add batch dimension
        ↓
  EfficientNet-B0 Backbone
  (Pre-trained on ImageNet)
        ↓
  Custom Classification Head
  • Global Average Pooling
  • Dense layers + Dropout
  • Softmax (38 classes)
        ↓
  Top-3 Predictions + Confidence
        ↓
  Disease Knowledge Lookup
  (Symptoms, Treatment, Prevention)
```

### Model Specifications

| Parameter | Value |
|-----------|-------|
| **Architecture** | EfficientNet-B0 (Transfer Learning) |
| **Dataset** | PlantVillage (~87,000 labeled images) |
| **Classes** | 38 (14 crop species, various conditions) |
| **Input Size** | 512 × 512 pixels |
| **Training Accuracy** | **97.8%** |
| **Validation Accuracy** | **94.99%** |
| **Epochs** | 50 |
| **Model File Size** | 16.6 MB |
| **Model Format** | HDF5 (.h5) via tf-Keras |
| **Inference Time** | ~0.3s per image (CPU) |

### Why EfficientNet-B0?

> EfficientNet-B0 was chosen because it provides an excellent **accuracy-to-efficiency trade-off**:
> - Scales depth, width, and resolution simultaneously using **compound scaling**
> - **8× smaller** than ResNet-50 with comparable or better accuracy
> - Pre-trained on ImageNet → excellent low-level feature extraction (edges, textures, colors)
> - Ideal for plant leaf pattern recognition

### Training Methodology

1. **Transfer Learning** — EfficientNet-B0 weights pre-trained on ImageNet
2. **Fine-tuning** — All layers unfrozen and fine-tuned on PlantVillage
3. **Data Augmentation** — Rotation, flipping, zoom, color jitter
4. **Optimizer** — Adam with learning rate scheduling
5. **Loss** — Categorical Cross-Entropy
6. **Regularization** — FixedDropout layers to prevent overfitting

### API & Model Usage

```python
import tf_keras as keras
import numpy as np
from PIL import Image

# Load model (cached)
model = keras.models.load_model('plant_disease_model.h5',
    custom_objects={'FixedDropout': FixedDropout})

# Preprocess image
img = Image.open('leaf.jpg').resize((512, 512))
img_array = np.array(img).astype('float32') / 255.0
img_array = np.expand_dims(img_array, axis=0)

# Predict
predictions = model.predict(img_array)
top_class = CLASS_NAMES[np.argmax(predictions)]
confidence = float(np.max(predictions)) * 100

print(f"Disease: {top_class} ({confidence:.1f}% confidence)")
```

---

## 🖼️ Product Screenshots

### 🏠 Main Interface
> Dark glassmorphism UI with animated upload zone and hero section

```
┌─────────────────────────────────────────────┐
│  🌿 AgroVision AI                           │
│  Plant Disease Detector                     │
│                                             │
│  ┌─────────────────────────────────────┐   │
│  │  📸 Drop your plant leaf image here │   │
│  │         or click to browse          │   │
│  └─────────────────────────────────────┘   │
│                                             │
│  ✅ 94.99% Accuracy  🌿 38 Diseases         │
│  🔬 14 Crops         ⚡ Instant Results     │
└─────────────────────────────────────────────┘
```

### 🔍 Detection Results
> After upload: AI confidence meter, top-3 predictions, disease info

```
┌─────────────────────────────────────────────┐
│  🍅 Tomato — Late Blight                    │
│  Severity: 🔴 CRITICAL                      │
│                                             │
│  AI Confidence  ████████████░░  87.3%       │
│                                             │
│  📋 Symptoms                                │
│  Water-soaked dark lesions, white mold...   │
│                                             │
│  💊 Treatment Steps                         │
│  1. ⚠️ Treat immediately — spreads fast    │
│  2. Apply mancozeb fungicide               │
│  3. Remove ALL infected material            │
│                                             │
│  🛡️ Prevention Tips                        │
│  • Plant resistant varieties               │
│  • Avoid overhead irrigation               │
└─────────────────────────────────────────────┘
```

### 📊 Top-3 Predictions Panel
> Ranked confidence scores for the top 3 most likely diseases

```
┌─────────────────────────────────────────────┐
│  🏆 Top Predictions                         │
│                                             │
│  #1 Tomato Late Blight      ████ 87.3%     │
│  #2 Tomato Early Blight     █░░░  8.1%     │
│  #3 Potato Late Blight      ░░░░  3.2%     │
└─────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
AgroVision-AI/
├── app/
│   └── app.py              # Main Streamlit application (1374 lines)
│                           # Contains: CSS, model loading, UI, prediction logic
├── src/
│   ├── model.py            # EfficientNet model architecture definition
│   ├── predict.py          # Prediction pipeline utilities
│   ├── preprocess.py       # Image preprocessing functions
│   └── train.py            # Model training script
├── notebook/               # Jupyter notebooks (EDA, training experiments)
├── .streamlit/
│   └── config.toml         # Streamlit theme & server configuration
├── plant_disease_model.h5  # Trained model weights (16.6 MB, required)
├── streamlit_app.py        # Entry point for Streamlit Cloud deployment
├── requirements.txt        # Python dependencies with version pins
├── runtime.txt             # Python runtime version for deployment
└── README.md               # This file
```

---

## 👥 Team Contributions

<table>
<tr>
<th>👤 Team Member</th>
<th>🎯 Role</th>
<th>📋 Contributions</th>
</tr>
<tr>
<td><b>Saqib Azair</b><br/><i>Team Lead & AI Engineer</i></td>
<td>ML / Backend</td>
<td>
• Designed & trained EfficientNet-B0 model<br/>
• Built model training pipeline (<code>src/train.py</code>)<br/>
• Achieved 94.99% validation accuracy<br/>
• Integrated tf-Keras model loading with custom FixedDropout layer<br/>
• Wrote prediction & preprocessing utilities
</td>
</tr>
<tr>
<td><b>ShahBano</b><br/><i>Ducomentation</i></td>
<td>UI / UX</td>
<td>
• Built responsive layout for mobile & desktop& documents<br/>
</td>
</tr>
<tr>

---

## 🌐 Deployment

### Streamlit Cloud (Recommended — Free)

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub account
4. Select this repo → set **Main file path** to `streamlit_app.py`
5. Add `MODEL_PATH` as a secret environment variable
6. Click **Deploy** ✅

> **Note**: Upload `plant_disease_model.h5` to GitHub or use Git LFS for large files.

### Local Development

```bash
streamlit run streamlit_app.py --server.port 8501
```

---

## 🤝 Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss what you would like to change.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📄 License

MIT License — free to use, modify, and distribute.

---

<div align="center">

**AgroVision AI** — Empowering farmers with intelligent crop protection 🌾

Made with ❤️ using TensorFlow & Streamlit

![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=flat&logo=tensorflow&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)

</div>
