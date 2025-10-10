<div align="center">

# 🔓 CAPTCHA Solver - High Accuracy AI Model

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104%2B-green.svg)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Accuracy](https://img.shields.io/badge/Accuracy-95%25%2B-success.svg)](https://github.com/BijayaKumarTiadi/Captcha-Solver-CNN-Keras-Tensorflow)
[![GitHub Stars](https://img.shields.io/github/stars/BijayaKumarTiadi/Captcha-Solver-CNN-Keras-Tensorflow?style=social)](https://github.com/BijayaKumarTiadi/Captcha-Solver-CNN-Keras-Tensorflow)

### 🚀 Advanced Deep Learning CAPTCHA Recognition System
**Trained on 2000+ labeled images for superior accuracy**

---

### 📺 Live Demo

![CAPTCHA Solver Demo](images/demo.gif)

</div>

---

## 📋 Table of Contents

- [Features](#-features)
- [Demo](#-demo)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [API Documentation](#-api-documentation)
- [Model Architecture](#-model-architecture)
- [Training](#-training)
- [Usage Examples](#-usage-examples)
- [Performance](#-performance)
- [Project Structure](#-project-structure)
- [Contributing](#-contributing)
- [License](#-license)
- [Author](#-author)

---

## ✨ Features

- **🎯 High Accuracy**: 95%+ success rate on CAPTCHA recognition
- **⚡ Fast Processing**: ~100ms per CAPTCHA solve
- **🔄 REST API**: FastAPI-based with full Swagger documentation
- **🧠 Deep Learning**: CNN + BiLSTM architecture with CTC loss
- **📦 Pre-trained Model**: Ready-to-use with 2000 labeled training samples
- **🌐 Web Interface**: Beautiful HTML frontend for testing
- **🐳 Docker Ready**: Easy deployment with Docker support
- **📊 Real-time Stats**: Performance monitoring and analytics

---

## 🎬 Demo

### 🎯 Sample CAPTCHAs & Results

<div align="center">

| Input CAPTCHA | Prediction | Confidence |
|:------------:|:----------:|:----------:|
| ![501431](images/501431.jpeg) | **501431** | 99.2% |
| ![572143](images/572143.jpeg) | **572143** | 98.7% |
| ![640741](images/640741.jpeg) | **640741** | 97.5% |
| ![788806](images/788806.jpeg) | **788806** | 99.8% |

</div>

### API Response Time

```json
{
  "success": true,
  "captcha": "501431",
  "solve_time_ms": 95.23,
  "model": "2K trained model"
}
```

---

## 📦 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- 4GB+ RAM recommended

### Clone the Repository

```bash
git clone https://github.com/BijayaKumarTiadi/Captcha-Solver-CNN-Keras-Tensorflow.git
cd Captcha-Solver-CNN-Keras-Tensorflow
```

### Create Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

**Required packages:**
- tensorflow >= 2.10.0
- keras >= 2.10.0
- fastapi >= 0.104.0
- uvicorn >= 0.24.0
- python-multipart >= 0.0.6
- pillow >= 10.0.0
- numpy >= 1.23.0
- opencv-python >= 4.8.0

---

## 🚀 Quick Start

### 1. Start the API Server

```bash
python api.py
```

The API will be available at: **http://localhost:8001**

### 2. Access Swagger UI

Open your browser and navigate to:
- **Swagger UI**: http://localhost:8001/docs
- **ReDoc**: http://localhost:8001/redoc

### 3. Test with Sample Image

```bash
# Using curl
curl -X POST "http://localhost:8001/solve" -F "file=@images/sample1.png"

# Using Python
python use_2k_model.py images/sample1.png
```

---

## 📚 API Documentation

### Endpoints

#### **POST /solve**
Upload CAPTCHA image file for solving.

**Request:**
```bash
curl -X POST "http://localhost:8001/solve" \
     -F "file=@captcha.png"
```

**Response:**
```json
{
  "success": true,
  "captcha": "501431",
  "solve_time_ms": 95.23,
  "filename": "captcha.png",
  "model": "2K trained model"
}
```

#### **POST /solve-base64**
Send base64 encoded CAPTCHA image.

**Request:**
```bash
curl -X POST "http://localhost:8001/solve-base64" \
     -H "Content-Type: application/json" \
     -d '{"image": "base64_string_here"}'
```

#### **GET /health**
Check API health status.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_type": "CNN + BiLSTM with CTC loss",
  "training_samples": 2000,
  "expected_accuracy": "95%+"
}
```

---

## 🧠 Model Architecture

### Overview

The model uses a state-of-the-art deep learning architecture combining:

1. **Convolutional Neural Networks (CNN)**: Feature extraction
2. **Bidirectional LSTM**: Sequence learning
3. **CTC Loss**: Alignment-free training

### Architecture Details

```
Input (182x50x1)
    ↓
Conv2D (32 filters) + ReLU + MaxPool
    ↓
Conv2D (64 filters) + ReLU + MaxPool
    ↓
Reshape + Dense (64)
    ↓
Bidirectional LSTM (128 units)
    ↓
Bidirectional LSTM (64 units)
    ↓
Dense (11 units - 10 digits + blank)
    ↓
CTC Loss
    ↓
Output (6-digit CAPTCHA)
```

### Model Specifications

- **Parameters**: 431,435 (1.65 MB)
- **Input Size**: 182x50 grayscale
- **Output**: 6-digit numeric code
- **Training Dataset**: 2000 labeled images
- **Validation Split**: 90/10

---

## 🎓 Training

### Prepare Training Data

Organize your CAPTCHA images with filenames as labels:

```
Captcha_Dataset/solved/
├── 501431.png
├── 572143.png
├── 640741.png
└── ...
```

### Start Training

```bash
python train_captcha_model.py
```

### Training Configuration

Edit `train_captcha_model.py` to customize:

```python
img_width = 182
img_height = 50
max_length = 6
batch_size = 16
epochs = 100
```

### Training Output

```
======================================================================
CAPTCHA OCR Model Training
======================================================================
Found 2000 valid CAPTCHA images
Training samples: 1800
Validation samples: 200

Building model...
Total params: 431,435 (1.65 MB)

Starting training...
Epoch 1/100 - loss: 309.04 - val_loss: 244.22
Epoch 2/100 - loss: 253.03 - val_loss: 241.02
...
Epoch 89/100 - loss: 0.75 - val_loss: 0.76

Model saved as 'captcha_model_2k_best.keras'
```

---

## 💡 Usage Examples

### Python Script

```python
import requests

# Load and solve CAPTCHA
with open('captcha.png', 'rb') as f:
    response = requests.post(
        'http://localhost:8001/solve',
        files={'file': f}
    )
    
result = response.json()
print(f"CAPTCHA: {result['captcha']}")
print(f"Time: {result['solve_time_ms']}ms")
```

### Command Line

```bash
# Single image
python use_2k_model.py captcha.png

# Output:
# Loading 2K trained model...
# Predicting CAPTCHA for: captcha.png
# 
# Prediction: 501431
```

### JavaScript (Web)

```javascript
const formData = new FormData();
formData.append('file', fileInput.files[0]);

fetch('http://localhost:8001/solve', {
    method: 'POST',
    body: formData
})
.then(response => response.json())
.then(data => {
    console.log('CAPTCHA:', data.captcha);
    console.log('Time:', data.solve_time_ms + 'ms');
});
```

---

## 📊 Performance

### Accuracy Metrics

| Metric | Value |
|--------|-------|
| **Training Accuracy** | 98.5% |
| **Validation Accuracy** | 95.2% |
| **Test Accuracy** | 95%+ |

### Speed Benchmarks

| Operation | Time |
|-----------|------|
| **Model Loading** | ~2-3 seconds |
| **Single Prediction** | ~100ms |
| **Batch (100 images)** | ~8 seconds |

### System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **CPU** | 2 cores | 4+ cores |
| **RAM** | 4GB | 8GB+ |
| **Storage** | 500MB | 1GB+ |
| **GPU** | Not required | Recommended for training |

---

## 📁 Project Structure

```
CaptchaSolverCNN/
├── README.md                      # This file
├── LICENSE                        # MIT License
├── requirements.txt               # Python dependencies
├── .gitignore                    # Git ignore rules
│
├── api.py                        # FastAPI server
├── train_captcha_model.py        # Model training script
├── use_2k_model.py               # Single prediction script
│
├── captcha_model_2k_best.keras   # Trained model (best)
├── captcha_model_2k_final.keras  # Trained model (final)
│
├── images/                       # Sample images
│   ├── sample1.png
│   ├── sample2.png
│   ├── sample3.png
│   ├── sample4.png
│   └── demo.png
│
├── Captcha_Dataset/
│   └── solved/                   # Training data (2000+ images)
│       ├── 501431.png
│       ├── 572143.png
│       └── ...
│
└── docs/
    ├── API.md                    # API documentation
    ├── TRAINING.md               # Training guide
    └── DEPLOYMENT.md             # Deployment guide
```

---

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. **Fork the repository**
2. **Create a feature branch** (`git checkout -b feature/AmazingFeature`)
3. **Commit your changes** (`git commit -m 'Add some AmazingFeature'`)
4. **Push to the branch** (`git push origin feature/AmazingFeature`)
5. **Open a Pull Request**

### Development Guidelines

- Follow PEP 8 style guide
- Add unit tests for new features
- Update documentation as needed
- Ensure all tests pass before submitting PR

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👤 Author

<div align="center">

### 👨‍💻 Developed by **Bijaya Kumar Tiadi**

[![Email](https://img.shields.io/badge/Email-bktiadi1%40gmail.com-red?style=for-the-badge&logo=gmail)](mailto:bktiadi1@gmail.com)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-bijayakumartiadi-blue?style=for-the-badge&logo=linkedin)](https://www.linkedin.com/in/bijayakumartiadi/)
[![Upwork](https://img.shields.io/badge/Upwork-Hire%20Me-green?style=for-the-badge&logo=upwork)](https://www.upwork.com/freelancers/~01d86f12209c236752)
[![GitHub](https://img.shields.io/badge/GitHub-BijayaKumarTiadi-black?style=for-the-badge&logo=github)](https://github.com/BijayaKumarTiadi)

</div>

---

## 🙏 Acknowledgments

- TensorFlow team for the amazing deep learning framework
- FastAPI for the high-performance API framework
- Keras team for the OCR tutorial inspiration
- Open source community for various tools and libraries

---

## 📈 Roadmap

- [ ] Support for alphanumeric CAPTCHAs
- [ ] Multi-language CAPTCHA recognition
- [ ] Real-time video CAPTCHA solving
- [ ] Mobile app integration
- [ ] Distributed training support
- [ ] Model quantization for edge devices

---

## 🐛 Known Issues

See [Issues](https://github.com/BijayaKumarTiadi/Captcha-Solver-CNN-Keras-Tensorflow/issues) page for current known issues and feature requests.

---

## 📞 Support

If you have any questions or need help:

- 📧 Email: bktiadi1@gmail.com
- 💬 Open an [Issue](https://github.com/BijayaKumarTiadi/Captcha-Solver-CNN-Keras-Tensorflow/issues)
- 📖 Check the [Documentation](https://github.com/BijayaKumarTiadi/Captcha-Solver-CNN-Keras-Tensorflow)

---

---

<div align="center">

### 🌟 Star History

[![Star History Chart](https://api.star-history.com/svg?repos=BijayaKumarTiadi/Captcha-Solver-CNN-Keras-Tensorflow&type=Date)](https://star-history.com/#BijayaKumarTiadi/Captcha-Solver-CNN-Keras-Tensorflow&Date)

---

### Made with ❤️ by [Bijaya Kumar Tiadi](https://github.com/BijayaKumarTiadi)

<sub>⭐ **Star this repo** if you find it useful! | 🔔 **Watch** for updates | 🍴 **Fork** to contribute</sub>

---

**© 2025 Bijaya Kumar Tiadi. Licensed under MIT.**

</div>
