import pandas as pd
import torch
import os
from PIL import Image 
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
from sklearn.model_selection import train_test_split

df = pd.read_csv('Data_Entry_2017.csv')
x = df['Image Index']
y = df['Finding Labels']

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.75, random_state = 8)

class CustomImageDataset(Dataset):
    def __init__(self, filenames, labels, transform=None, target_transform=None):
        self.filenames = filenames
        self.img_dir = img_dir
        self.labels = labels
        self.transform = transform
        self.target_transform = target_transform
    
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.labels.iloc[idx])
        image = Image.open(img_path)
        label = self.labels.iloc[idx]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
