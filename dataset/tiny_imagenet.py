import os

import torch
import torchvision.transforms as T
from torchvision.datasets import ImageFolder

# Normalization statistics of ImageNet
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def get_transform():
    return T.Compose([
        T.Resize((224, 224)),  # Resize to fit the input dimensions of the network
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD),
    ])


def get_dataloaders(data_dir, batch_size=32, num_workers=2):
    # data_dir must contain train/ and val/ in ImageFolder format: root/{classX}/x001.jpg
    # (run utils/prepare_tiny_imagenet.py first)
    transform = get_transform()

    train_dataset = ImageFolder(root=os.path.join(data_dir, 'train'), transform=transform)
    val_dataset = ImageFolder(root=os.path.join(data_dir, 'val'), transform=transform)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader
