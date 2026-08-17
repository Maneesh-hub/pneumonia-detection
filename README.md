Markdown
# Chest X-Ray Pneumonia Inference Server

[![GitHub Repository](https://img.shields.io/badge/GitHub-Repository-blue?logo=github)](https://github.com/Maneesh-hub/pneumonia-detection)

A high-performance asynchronous FastAPI inference backend powering a PyTorch (ResNet18) Deep CNN model for real-time pneumonia detection from chest radiograph images, accompanied by a lightweight Streamlit diagnostic client.

---

## Local Setup & Execution Guide

Follow these steps to clone the repository, set up the environment, and launch both the backend server and frontend client locally.

### 1. Clone the Repository

```bash
git clone [https://github.com/Maneesh-hub/pneumonia-detection.git](https://github.com/Maneesh-hub/pneumonia-detection.git)
cd pneumonia-detection
2. Set Up Virtual Environment
Windows (PowerShell):

PowerShell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
macOS / Linux:

Bash
python3 -m venv .venv
source .venv/bin/activate
3. Install Dependencies
Bash
pip install -r requirements.txt
How to Run the System
Running the complete application requires two open terminal windows.

Step 1: Launch the FastAPI Backend (Terminal 1)
Execute the backend Uvicorn server from the root directory:

Bash
python -m uvicorn src.main:app --reload --host 0.0.0.0 --port 8000
Local API Base URL: http://127.0.0.1:8000

Interactive OpenAPI/Swagger Interface: http://127.0.0.1:8000/docs

Step 2: Launch the Streamlit Client UI (Terminal 2)
In a second terminal window, launch the Streamlit dashboard client:

Bash
streamlit run app.py
Local Dashboard UI: http://localhost:8501

Repository Directory Structure
Plaintext
pneumonia-detection/
├── app.py                  # Streamlit visual frontend client
├── requirements.txt        # Managed Python dependencies
├── README.md               # Setup & project documentation
├── .gitignore              # Git ignore configuration
├── models/
│   └── pneumonia_cnn.pt    # Serialized PyTorch model checkpoint
├── data/
│   ├── test/               # Test set radiograph images
│   ├── train/              # Training set radiograph images
│   └── val/                # Validation set radiograph images
└── src/
    ├── dataset.py          # PyTorch Dataset & DataLoader creation script
    ├── evaluate.py         # Model evaluation metrics script
    ├── main.py             # FastAPI backend server & API endpoints
    ├── model.py            # Neural network architecture definition
    └── train.py            # Model training & checkpointing script
API Documentation & Endpoint Specification
1. GET /health
Checks server status, verifies model memory availability, and identifies the active hardware device (cpu / cuda).

Response (200 OK):

JSON
{
  "status": "healthy",
  "model_loaded": true,
  "device": "cpu"
}
2. POST /predict
Processes an uploaded chest radiograph image through PyTorch preprocessing transforms (224x224 resize, tensor conversion, ImageNet mean/std normalization) and returns binary classification predictions, class probabilities, and a clinical triage flag.

Request Format: multipart/form-data

Body Parameter: file (.jpg, .jpeg, .png)

Response (200 OK):

JSON
{
  "filename": "person78_bacteria_386.jpeg",
  "prediction": "PNEUMONIA",
  "confidence": 0.9988,
  "flag_for_review": true,
  "class_probabilities": {
    "NORMAL": 0.0012,
    "PNEUMONIA": 0.9988
  }
}
Model Specifications
Backbone Architecture: Fine-tuned ResNet18 Deep Convolutional Neural Network

Framework: PyTorch & Torchvision

Input Resolution: 224 x 224 pixels

ImageNet Normalization:

Mean: [0.485, 0.456, 0.406]

Std: [0.229, 0.224, 0.225]

Evaluation Focus: High Sensitivity / Recall to minimize false-negative clinical misclassifications.