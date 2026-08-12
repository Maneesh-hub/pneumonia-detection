import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

class PneumoniaCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(PneumoniaCNN, self).__init__()

        self.model = resnet18(weights=ResNet18_Weights.DEFAULT)
        
        for param in self.model.parameters():
            param.requires_grad = False
            
        in_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        return self.model(x)