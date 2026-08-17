import io
import torch
import torch.nn.functional as F
from PIL import Image
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from torchvision import transforms

from src.model import PneumoniaCNN

#1.Configuration & Model Path

MODEL_PATH="./models/pneumonia_cnn.pt"
CLASS_NAMES=["NORMAL","PNEUMONIA"]
DEVICE=torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_container={}
 
# Match preprocessing to training specs
inference_transform=transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(),transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])])

# Loads model ONCE when server starts
@asynccontextmanager
async def lifespan(app:FastAPI):
    print(f"Loading model from {MODEL_PATH} onto {DEVICE}...")
    model=PneumoniaCNN(num_classes=len(CLASS_NAMES))
    try:
        model.load_state_dict(torch.load(MODEL_PATH,map_location=DEVICE))
        model.to(DEVICE)
        model.eval()
        model_container["model"]=model
        print("Model successfully loaded!")
    except Exception as e:
        print(f"Warning: Could not load model file from {MODEL_PATH}: {e}")
        model_container["model"]=None
    
    yield
    model_container.clear()

app=FastAPI(
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
    is_ready=model_container.get("model") is not None
    return {
        "status": "healthy" if is_ready else "unhealthy",
        "model_loaded": is_ready,
        "device": str(DEVICE)
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile=File(...)):
    if model_container.get("model") is None:
        raise HTTPException(status_code=500, detail="Model file is not loaded on server.")

    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

    try:
        contents=await file.read()
        image=Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400,detail="Invalid image file.")

    tensor=inference_transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits=model_container["model"](tensor)
        probabilities=F.softmax(logits,dim=1)[0]

    predicted_idx=torch.argmax(probabilities).item()
    predicted_class=CLASS_NAMES[predicted_idx]
    confidence=float(probabilities[predicted_idx])

    prob_dict={CLASS_NAMES[i]: round(float(probabilities[i]), 4) for i in range(len(CLASS_NAMES))}
    flag_for_review=predicted_class=="PNEUMONIA" or confidence < 0.90

    return PredictionResponse(
        filename=file.filename,
        prediction=predicted_class,
        confidence=round(confidence, 4),
        flag_for_review=flag_for_review,
        class_probabilities=prob_dict
    )