import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

from src.model import PneumoniaCNN

# 1. Setup Device & Model
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "./models/pneumonia_cnn.pt"

model = PneumoniaCNN(num_classes=2)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# 2. Load Test Data
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Ensure this points to your actual test data directory
test_dataset = datasets.ImageFolder(root="data/test", transform=test_transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 3. Collect Predictions
all_preds = []
all_labels = []

print("Evaluating model on test dataset...")
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(DEVICE)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())

# 4. Print Results
class_names = test_dataset.classes  # ['NORMAL', 'PNEUMONIA']

print("\n================ CLASSIFICATION REPORT ================")
print(classification_report(all_labels, all_preds, target_names=class_names, digits=4))

print("\n================ CONFUSION MATRIX ================")
cm = confusion_matrix(all_labels, all_preds)
print(f"True Negatives (Correct NORMAL):   {cm[0][0]}")
print(f"False Positives (Wrong PNEUMONIA): {cm[0][1]}")
print(f"False Negatives (Missed Sick):     {cm[1][0]}")
print(f"True Positives (Correct PNEUMONIA): {cm[1][1]}")