from dataset import CustomImageDataset
from model import define_model
from torchvision import datasets
from torch.utils.data import DataLoader
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

ds_train = CustomImageDataset(x_train, y_train)
ds_test = CustomImageDataset(x_test, y_test)

train_dl = DataLoader(ds_train, batch_size = 64, shuffle = True)
test_dl = DataLoader(ds_test, batch_size = 64, shuffle = True)

model = define_model()

criterion = nn.BCEWithLogitsLoss()