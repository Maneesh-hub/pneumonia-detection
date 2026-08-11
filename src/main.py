# import io
# import torch
# import torch.nn.functional as F
# from PIL import Image
# from fastapi import FastAPI, File, UploadFile, HTTPException
# from torchvision import transforms

# # Import your PyTorch model architecture
# from src.model import PneumoniaCNN  # Or: from .model import PneumoniaCNN

# app = FastAPI(title="Pneumonia Detection API")

# # 1. Setup Device & Load Model
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = PneumoniaCNN()

# # Load saved weights if you have trained your model (update path if needed)
# model.load_state_dict(torch.load("models/pneumonia_cnn.pt", map_location=device))
# model.to(device)
# model.eval()

# # Define image preprocessing matching your training pipeline
# transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.Grayscale(num_output_channels=1),  # Set to 3 if using RGB
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485], std=[0.229])
# ])

# # ==========================================
# # 1. Health Check Endpoint
# # ==========================================
# @app.get("/health")
# def health_check():
#     return {
#         "status": "healthy",
#         "model_loaded": model is not None,
#         "device": str(device)
#     }

# # ==========================================
# # 2. Inference Endpoint
# # ==========================================
# @app.post("/predict")
# async def predict(file: UploadFile = File(...)):
#     # Validate file type
#     if not file.content_type.startswith("image/"):
#         raise HTTPException(status_code=400, detail="File uploaded is not an image.")

#     try:
#         # Read image file bytes
#         contents = await file.read()
#         image = Image.open(io.BytesIO(contents)).convert("RGB")
        
#         # Preprocess image and add batch dimension [1, C, H, W]
#         tensor_image = transform(image).unsqueeze(0).to(device)

#         # Run inference
#         with torch.no_grad():
#             outputs = model(tensor_image)
#             probabilities = F.softmax(outputs, dim=1)[0]

#         # Extract probabilities (assuming index 0 = NORMAL, index 1 = PNEUMONIA)
#         normal_prob = float(probabilities[0])
#         pneumonia_prob = float(probabilities[1])

#         # Determine prediction
#         if pneumonia_prob > normal_prob:
#             prediction = "PNEUMONIA"
#             confidence = pneumonia_prob
#         else:
#             prediction = "NORMAL"
#             confidence = normal_prob

#         # Flag for human review if model confidence is under 90% (0.90)
#         flag_for_review = confidence < 0.90

#         return {
#             "filename": file.filename,
#             "prediction": prediction,
#             "confidence": round(confidence, 4),
#             "flag_for_review": flag_for_review,
#             "class_probabilities": {
#                 "NORMAL": round(normal_prob, 4),
#                 "PNEUMONIA": round(pneumonia_prob, 4)
#             }
#         }

#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")
import io
import torch
import torch.nn.functional as F
from PIL import Image
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from torchvision import transforms

from src.model import PneumoniaCNN

# ==========================================
# 1. Configuration & Model Path
# ==========================================
MODEL_PATH = "./models/pneumonia_cnn.pt"  # <-- Change this path if your file has a different name
CLASS_NAMES = ["NORMAL", "PNEUMONIA"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_container = {}

# Match preprocessing to training specs
inference_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Loads model ONCE when server starts
@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"Loading model from {MODEL_PATH} onto {DEVICE}...")
    model = PneumoniaCNN(num_classes=len(CLASS_NAMES))
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        model.to(DEVICE)
        model.eval()
        model_container["model"] = model
        print("Model successfully loaded!")
    except Exception as e:
        print(f"Warning: Could not load model file from {MODEL_PATH}: {e}")
        model_container["model"] = None
    
    yield
    model_container.clear()

app = FastAPI(
    title="Chest X-Ray Pneumonia Detection API",
    lifespan=lifespan
)

class PredictionResponse(BaseModel):
    filename: str
    prediction: str
    confidence: float
    flag_for_review: bool
    class_probabilities: dict

@app.get("/health")
def health_check():
    is_ready = model_container.get("model") is not None
    return {
        "status": "healthy" if is_ready else "unhealthy",
        "model_loaded": is_ready,
        "device": str(DEVICE)
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    if model_container.get("model") is None:
        raise HTTPException(status_code=500, detail="Model file is not loaded on server.")

    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file.")

    tensor = inference_transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model_container["model"](tensor)
        probabilities = F.softmax(logits, dim=1)[0]

    predicted_idx = torch.argmax(probabilities).item()
    predicted_class = CLASS_NAMES[predicted_idx]
    confidence = float(probabilities[predicted_idx])

    prob_dict = {CLASS_NAMES[i]: round(float(probabilities[i]), 4) for i in range(len(CLASS_NAMES))}
    flag_for_review = predicted_class == "PNEUMONIA" or confidence < 0.90

    return PredictionResponse(
        filename=file.filename,
        prediction=predicted_class,
        confidence=round(confidence, 4),
        flag_for_review=flag_for_review,
        class_probabilities=prob_dict
    )