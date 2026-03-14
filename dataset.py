import pandas as pd
import torch
import os
from PIL import Image 
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer

df = pd.read_csv('Data_Entry_2017.csv')
x = df['Image Index']

mlb = MultiLabelBinarizer()
y = mlb.fit_transform(df['Finding Labels'].str.split('|'))

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.75, random_state = 8)
x_train = x_train.reset_index(drop=True)
x_test = x_test.reset_index(drop=True)

class CustomImageDataset(Dataset):
    def __init__(self, filenames, labels, transform=None, target_transform=None):
        self.filenames = filenames
        self.labels = labels
        self.transform = transform
        self.target_transform = target_transform
    
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        for i in ['images_001', 'images_002', 'images_003', 'images_004', 'images_005', 'images_006',
                  'images_007', 'images_008', 'images_009', 'images_010', 'images_011', 'images_012']:
            pth = os.path.join('assets', i, 'images', self.filenames[idx])
            if os.path.exists(pth):
                img_path = pth
                break
                
        image = Image.open(img_path)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
