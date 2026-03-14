import torch
import torch.nn as nn
from dataset import *
from model import define_model
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights
from torch.optim import Adam
from opacus import PrivacyEngine

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

pe = PrivacyEngine()
model, optimizer, train_dl = pe.make_private(
    module = model,
    optimizer = optimizer,
    data_loader = train_dl,
    noise_multiplier = 1.1,
    max_grad_norm = 1.0,
    )

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
            
            val_loss += loss.item()
            
    return val_loss

for epoch in range(100):
    rl = train(model, train_dl, criterion)
    vl = validate(model, test_dl, criterion)
    print(f"The running loss is = {rl} and the val loss is = {vl}")
    torch.save(model.state_dict(), 'model.pth')
    
    
    
    
    
    
    
    
    
    
    
    