import pandas as pd
import torch
import os
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

def define_model() -> resnet50:
    model = resnet50(weights = 'DEFAULT')
    model.fc = nn.Linear(2048, 14)
    
    return model

    

