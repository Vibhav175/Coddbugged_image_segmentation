#!/usr/bin/env python
# coding: utf-8

# In[17]:


import torch
import torch.nn as nn
from torchvision.transforms import ToTensor
from PIL import Image

# Define your custom dataset class to load a single image
class CustomDataset:
    def __init__(self, image_path, transform=None):
        self.image_path = image_path
        self.transform = transform

    def __len__(self):
        return 1  # Only one image in the dataset

    def __getitem__(self, idx):
        image = Image.open(self.image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image

# Load a single image
image_path = "C:\\Users\\Hp\\Downloads\\B4120C1_img_01.jpeg"
transform = ToTensor()  # Convert image to tensor
dataset = CustomDataset(image_path, transform)

# Define the model architecture
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(64, 1, kernel_size=1),
            nn.Sigmoid()  # Sigmoid activation for binary segmentation
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# Create an instance of the UNet model
model = UNet()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)  # Move model to the appropriate device

# Check if model has trainable parameters
if sum(p.numel() for p in model.parameters() if p.requires_grad) == 0:
    raise ValueError("Model has no trainable parameters.")

# Define loss function and optimizer
criterion = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    for images in [dataset]:  # Process a single image
        optimizer.zero_grad()
        images = images.to(device)  # Move images to the appropriate device
        outputs = model(images.unsqueeze(0))  # Add a batch dimension
        loss = criterion(outputs, images.unsqueeze(0))  # Add a batch dimension
        loss.backward()
        optimizer.step()

