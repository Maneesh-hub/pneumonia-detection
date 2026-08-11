import torch
import torch.nn as nn

class PneumoniaCNN(nn.Module):
    def __init__(self, num_classes: int = 2):
        super(PneumoniaCNN, self).__init__()
        
        # Feature Extractor Block
        self.features=nn.Sequential(
            
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # Output: 32 x 112 x 112
            
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # Output: 64 x 56 x 56
            
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # Output: 128 x 28 x 28

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2)  # Output: 256 x 14 x 14
        )
        
        # Classification Head
        self.classifier=nn.Sequential(nn.Flatten(),nn.Linear(256*14*14,128),nn.ReLU(),nn.Dropout(p=0.5),nn.Linear(128,num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x=self.features(x)
        logits=self.classifier(x)
        return logits