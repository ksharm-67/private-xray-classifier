import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
from sklearn.model_selection import train_test_split

df = pd.read_csv('assets/Data_Entry_2017.csv')
x = df['Image Index']
y = df['Finding Labels']

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.75, random_state = 8)

