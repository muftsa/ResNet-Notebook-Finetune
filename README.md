# ResNet-Notebook-Finetune

Below is a Jupyter notebook for fine-tuning an OpenCV-based model using PyTorch. Since OpenCV does not have deep learning models suitable for direct fine-tuning, we'll use a PyTorch model (e.g., ResNet) that works well for fine-tuning with image data.

### Requirements
1. Install necessary libraries:
   ```bash
   pip install torch torchvision opencv-python
   ```

2. **Directory Structure**:
   Organize the data into `train` and `val` directories:
   ```
   data/
       train/
           class1/
               img1.jpg
               img2.jpg
           class2/
               img1.jpg
       val/
           class1/
               img1.jpg
           class2/
               img1.jpg
   ```

### Jupyter Notebook Code

This notebook:
1. Loads and preprocesses images.
2. Fine-tunes a pre-trained ResNet model on custom data.
3. Evaluates the model on the validation set.

```python
# Fine-tuning a ResNet model with PyTorch and OpenCV images

# Imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
import numpy as np
import cv2
import os

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Define paths for training and validation data
train_dir = 'data/train'
val_dir = 'data/val'

# Define image transformations
# These include resizing, normalization, and data augmentation for training
transform_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

transform_val = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load datasets
train_dataset = ImageFolder(root=train_dir, transform=transform_train)
val_dataset = ImageFolder(root=val_dir, transform=transform_val)

# Data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Load a pre-trained ResNet model and fine-tune the last layer
model = models.resnet18(pretrained=True)

# Modify the final layer for the number of classes in our dataset
num_classes = len(train_dataset.classes)
model.fc = nn.Linear(model.fc.in_features, num_classes)

# Move model to device
model = model.to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Training and validation functions
def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    running_loss = 0.0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
    
    epoch_loss = running_loss / len(loader.dataset)
    return epoch_loss

def validate(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()

    epoch_loss = running_loss / len(loader.dataset)
    accuracy = correct / len(loader.dataset)
    return epoch_loss, accuracy

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    train_loss = train_one_epoch(model, train_loader, optimizer, criterion)
    val_loss, val_accuracy = validate(model, val_loader, criterion)
    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

print("Training complete.")

# Save the fine-tuned model
torch.save(model.state_dict(), "fine_tuned_resnet.pth")
```

### Explanation

1. **Data Loading**:
   - `ImageFolder` organizes the images based on directory structure and applies the transformations (`transform_train` and `transform_val`) for data augmentation and normalization.

2. **Model Fine-Tuning**:
   - Loads a pre-trained ResNet-18 model and replaces its final layer to match the number of classes in your dataset.
   - Uses cross-entropy loss and SGD optimizer with momentum.

3. **Training and Validation**:
   - `train_one_epoch`: Runs one epoch of training, calculating loss and updating weights.
   - `validate`: Evaluates the model on the validation set and calculates accuracy.

4. **Training Loop**:
   - Loops through each epoch, calling the training and validation functions, and prints loss and accuracy.

5. **Model Saving**:
   - Saves the fine-tuned model to a file (`fine_tuned_resnet.pth`).

This notebook will allow you to fine-tune a ResNet model on your predefined training images, with the validation set providing accuracy monitoring. You can adjust the model and optimizer as needed based on your dataset and requirements.
