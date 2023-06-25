from typing import Any, Tuple
import torchvision
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import os
from PIL import Image




class ImageFolder(torchvision.datasets.ImageFolder):
    def __init__(self, image_pths, targets ,transform=None, img_size = None):
        self.image_paths = image_pths
        self.targets = targets
        self.transform = transform
        self.image_size = img_size

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        
        image = image.resize((self.image_size, self.image_size))
        if self.transform:
            image = self.transform(image)
        target = self.targets[idx]
        return image, target
    
    def __len__(self):
        return len(self.image_paths)