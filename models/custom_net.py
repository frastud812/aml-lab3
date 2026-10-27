import torch
from torch import nn

# Define the custom neural network
class CustomNet(nn.Module):
    def __init__(self):
        super(CustomNet, self).__init__()
        # Define layers of the neural network
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1, stride=1)

        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        # Add more layers...
        self.conv3 = nn.Conv2d(128, 256, kernel_size= 3, padding=1)

        self.fc1 = nn.Linear(256, 200) # 200 is the number of classes in TinyImageNet

        self.flatten = nn.Flatten()

        self.global_pool = nn.AdaptiveAvgPool2d((1,1))

        self.pool = nn.MaxPool2d(kernel_size = 2)
        self.pool1 = nn.MaxPool2d(kernel_size = 4)

    def forward(self, x):
        # Define forward pass

        x = self.pool1(self.conv1(x).relu()) # B x 64 x 56 x 56
        x = self.pool(self.conv2(x).relu()) # B x 128 x 28 x 28
        x = self.pool(self.conv3(x).relu()) # B x 256 x 14 x 14
        x = self.global_pool(x) # B x 256 x 1 x 1
        x = self.flatten(x) # B x 256
        x = self.fc1(x)  # B x 200

        return x