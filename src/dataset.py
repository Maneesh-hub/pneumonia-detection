import os
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Standard dimensions & ImageNet normalization parameters
IMAGE_SIZE=(224,224)
BATCH_SIZE=32

def get_data_transforms():
    """Returns train and evaluation image transformations.
    Train split includes data augmentation to prevent overfitting."""

    train_transform=transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.1,contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])
    ])

    eval_transform=transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])
    ])

    return train_transform,eval_transform

def create_dataloaders(data_dir: str,batch_size: int=BATCH_SIZE):
    """Builds PyTorch DataLoaders for Train, Validation, and Test splits."""
    train_transform,eval_transform=get_data_transforms()

    train_dataset=datasets.ImageFolder(os.path.join(data_dir,'train'),transform=train_transform)

    val_dataset=datasets.ImageFolder(os.path.join(data_dir, 'val'),transform=eval_transform)

    test_dataset=datasets.ImageFolder(os.path.join(data_dir, 'test'),transform=eval_transform)

    train_loader=DataLoader(train_dataset,batch_size=batch_size,shuffle=True,num_workers=0)
    val_loader=DataLoader(val_dataset,batch_size=batch_size,shuffle=False,num_workers=0)
    test_loader=DataLoader(test_dataset,batch_size=batch_size,shuffle=False,num_workers=0)

    return train_loader,val_loader,test_loader,train_dataset.classes