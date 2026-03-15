import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from dataset import *
from model import define_model
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights
from torch.optim import Adam
from sklearn.metrics import roc_curve, auc

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),  
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

ds_train = CustomImageDataset(x_train, y_train, transform = transform)
ds_test = CustomImageDataset(x_test, y_test, transform = transform)

train_dl = DataLoader(ds_train, batch_size=64, shuffle=True)
test_dl = DataLoader(ds_test, batch_size=64, shuffle=True)

model = define_model()
criterion = nn.BCEWithLogitsLoss()
optimizer = Adam(model.parameters(), lr = 0.001)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)

probs, labels = [], []

def train(model: resnet50, train_dl: DataLoader, criterion) -> int:
    #Trains one epoch
    running_loss, loss = 0, 0
    
    model.train()
    
    for i, data in enumerate(train_dl):
        optimizer.zero_grad()
        ip, lab = data
        ip, lab = ip.to(device), lab.to(device)
        op = model(ip)

        loss = criterion(op, lab)
        loss.backward()
        
        optimizer.step()
        
        running_loss += loss.item()
        if i % 100 == 0:
            print(f"Batch {i}, loss: {loss.item()}")
        
    return running_loss
    
def validate(model: resnet50, test_dl: DataLoader, criterion) -> int:
    model.eval()
    val_loss, correct, total = 0, 0, 0
    
    with torch.no_grad():
        for ip, lab in test_dl:
            ip, lab = ip.to(device), lab.to(device)            
            op = model(ip)
            loss = criterion(op, lab)
            
            labels.append(lab)
            probs.append(torch.sigmoid(op))
            
            val_loss += loss.item()
            
    return val_loss

for epoch in range(5):
    rl = train(model, train_dl, criterion)
    vl = validate(model, test_dl, criterion)
    print(f"The running loss is = {rl} and the val loss is = {vl}")
    torch.save(model.state_dict(), 'model.pth')

y = torch.cat(labels)
scores = torch.cat(probs)

y = y.detach().cpu().numpy()
scores = scores.detach().cpu().numpy()

fpr, tpr = {}, {}

for i in range(15):
    fpr[i], tpr[i], _ = roc_curve(y[:, i], scores[:, i])
    roc_auc = auc(fpr[i], tpr[i])
    plt.plot(fpr[i], tpr[i], label=f'{model.__class__.__name__} - Class {i} (AUC = {roc_auc:.2f})')

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves for ChestX-ray14 Disease Classification')
plt.legend(loc="lower right")
plt.show()




    