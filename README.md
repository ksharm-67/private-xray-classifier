# X-Ray Disease Classification Model

A deep learning model trained on the NIH ChestX-ray14 dataset to classify 15 thoracic diseases from chest X-ray images using a fine-tuned ResNet50 architecture.

### Dataset
The NIH ChestX-ray14 dataset contains 112,120 frontal-view chest X-ray images labeled with 15 disease findings. The dataset is publicly available on Kaggle.

### Model
ResNet50 pretrained on ImageNet, with the final fully connected layer replaced to output 15 classes. Trained using Binary Cross Entropy with Logits Loss to handle the multilabel classification problem.

### Results
Model performance is evaluated using AUC-ROC curves per disease class, which is the standard evaluation metric for this dataset.

## Setup

### Install dependencies:

```pip install torch torchvision pandas scikit-learn Pillow kagglehub```

### Download the dataset:
```
pythonimport kagglehub
path = kagglehub.dataset_download("nih-chest-xrays/data")
```

## Training
```
python train.py
```
Trains for 5 epochs with Adam optimizer at a learning rate of 0.001 and batch size of 64. Model weights are saved after each epoch to model.pth.

### Tech Stack

PyTorch
torchvision
scikit-learn
Pandas
PIL
