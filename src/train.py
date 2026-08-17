import os
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import precision_recall_fscore_support
from dataset import create_dataloaders
from model import PneumoniaCNN

def evaluate_model(model,dataloader,criterion,device):
    model.eval()
    running_loss=0.0
    all_preds=[]
    all_targets=[]

    with torch.no_grad():
        for images,labels in dataloader:
            images,labels=images.to(device),labels.to(device)
            outputs=model(images)
            loss=criterion(outputs,labels)
            
            running_loss+=loss.item()*images.size(0)
            preds=torch.argmax(outputs,dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())

    total_loss=running_loss/len(dataloader.dataset)
    precision,recall,f1, _ =precision_recall_fscore_support(all_targets,all_preds,average='binary',zero_division=0)
    accuracy=sum([p==t for p,t in zip(all_preds,all_targets)])/len(all_targets)

    return total_loss,accuracy,precision,recall,f1

def train():
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    data_dir="./data"
    model_save_path="./models/pneumonia_cnn.pt"
    os.makedirs("./models",exist_ok=True)

    train_loader,val_loader,test_loader,classes=create_dataloaders(data_dir)

    model=PneumoniaCNN(num_classes=len(classes)).to(device)
    criterion=nn.CrossEntropyLoss()
    optimizer=optim.Adam(model.parameters(),lr=0.0001,weight_decay=1e-4)

    epochs=5
    best_val_loss=float('inf') 

    for epoch in range(epochs):
        model.train()
        train_loss=0.0
        
        for images,labels in train_loader:
            images,labels=images.to(device),labels.to(device)
            
            optimizer.zero_grad()
            outputs=model(images)
            loss=criterion(outputs,labels)
            loss.backward()
            optimizer.step()
            
            train_loss+=loss.item()*images.size(0)

        epoch_train_loss=train_loss/len(train_loader.dataset)
        val_loss,val_acc,val_prec,val_rec,val_f1=evaluate_model(model,val_loader,criterion,device)

        print(f"Epoch [{epoch+1}/{epochs}] | "
              f"Train Loss: {epoch_train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val Acc: {val_acc:.4f} | "
              f"Val Recall: {val_rec:.4f}")


        if val_loss < best_val_loss:
            best_val_loss=val_loss
            torch.save(model.state_dict(),model_save_path)
            print(f"--> Saved best model checkpoint to {model_save_path}")

    model.load_state_dict(torch.load(model_save_path))
    test_loss,test_acc,test_prec,test_rec,test_f1=evaluate_model(model,test_loader,criterion,device)
    
    print("\n-----------Final Test Performance--------------")
    print(f"Test Accuracy  : {test_acc:.4f}")
    print(f"Test Precision : {test_prec:.4f}")
    print(f"Test Recall    : {test_rec:.4f}")
    print(f"Test F1-Score  : {test_f1:.4f}")
    print("--------------------------------------------------\n")

if __name__=='__main__':
    train() 