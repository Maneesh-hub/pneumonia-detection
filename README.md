# Chest X-Ray Pneumonia Detection System

[![GitHub Repository](https://img.shields.io/badge/GitHub-Repository-blue?logo=github)](https://github.com/Maneesh-hub/pneumonia-detection)

An end-to-end medical AI diagnostic tool that processes uploaded chest radiograph images through a fine-tuned **ResNet18** Deep Convolutional Neural Network (PyTorch) to return real-time diagnostic predictions, confidence scores, and clinical triage alerts. The application consists of a low-latency **FastAPI** backend served alongside an interactive **Streamlit** user interface.

---

## Live Deployment Links

* **GitHub Repository:** [https://github.com/Maneesh-hub/pneumonia-detection](https://github.com/Maneesh-hub/pneumonia-detection)
* **Frontend Web UI (Streamlit):** `https://pneumonia-detection-mfyhgav44utq9jzgywiubm.streamlit.app`
* **Backend API (Render):** `https://pneumonia-detection-api.onrender.com`
* **Interactive API Docs (Swagger UI):** `https://pneumonia-detection-api.onrender.com/docs`

---

## Key Features

* **ResNet18 CNN Architecture:** Fine-tuned deep network leveraging transfer learning for high-accuracy chest X-ray image classification.
* **FastAPI REST Backend:** Asynchronous API service (`POST /predict`) built for scalable, low-latency model inference.
* **Streamlit Interactive UI:** Drag-and-drop web dashboard for doctors and developers to visualize diagnosis, confidence percentages, and probability distributions.
* **Clinical Triage Flagging:** Automatically tags positive or uncertain detections with `flag_for_review: true` for prioritized secondary medical review.

---

## System Architecture

```text
[ User / Web Dashboard (Streamlit) ]
                 │
                 │ HTTP POST (Multipart Image Bytes)
                 ▼
     [ FastAPI Backend Engine ]
                 │
                 ├──> Preprocessing (224x224 Resize, ImageNet Normalization)
                 │
                 ├──> PyTorch ResNet18 Model Inference (models/pneumonia_cnn.pt)
                 │
                 └──> Confidence & Clinical Triage Evaluation
                 │
                 ▼
    [ Structured JSON Response ]
Repository Directory StructurePlaintextpneumonia-detection/
├── app.py                   # Streamlit web frontend application
├── requirements.txt         # Python dependencies
├── README.md                # Project documentation
├── models/
│   └── pneumonia_cnn.pt     # Saved PyTorch ResNet18 model weights
└── src/
    ├── main.py              # FastAPI server implementation
    ├── model.py             # ResNet18 neural network architecture
    └── evaluate.py          # Model evaluation metrics script
Local Setup & Installation1. Clone the RepositoryBashgit clone [https://github.com/Maneesh-hub/pneumonia-detection.git](https://github.com/Maneesh-hub/pneumonia-detection.git)
cd pneumonia-detection
2. Set Up Virtual EnvironmentWindows (PowerShell):PowerShellpython -m venv .venv
.\.venv\Scripts\Activate.ps1
macOS / Linux:Bashpython3 -m venv .venv
source .venv/bin/activate
3. Install Package DependenciesBashpip install -r requirements.txt
How to Run LocallyRunning the full application locally requires two separate terminal windows:Step 1: Launch the FastAPI Backend (Terminal 1)Bashpython -m uvicorn src.main:app --reload --port 8000
Access local API docs at: http://127.0.0.1:8000/docsStep 2: Launch the Streamlit Frontend (Terminal 2)Bashpython -m streamlit run app.py
Access local dashboard UI at: http://localhost:8501API Documentation & UsageEndpoints OverviewMethodEndpointDescriptionGET/healthHealth check endpoint returning model loading statusPOST/predictPrimary inference endpoint for processing image filesSample POST /predict Payload & ResponseRequestContent-Type: multipart/form-dataBody: file (Image in .jpg, .jpeg, or .png format)Response (200 OK)JSON{
  "filename": "person78_bacteria_386.jpeg",
  "prediction": "PNEUMONIA",
  "confidence": 0.9988,
  "flag_for_review": true,
  "class_probabilities": {
    "NORMAL": 0.0012,
    "PNEUMONIA": 0.9988
  }
}
Model & Training DetailsBackbone: ResNet18 (Deep Residual Learning)Framework: PyTorch & TorchvisionInput Resolution: 224 × 224 pixelsNormalization Parameters: Mean [0.485, 0.456, 0.406], Std [0.229, 0.224, 0.225]