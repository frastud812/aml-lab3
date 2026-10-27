import argparse

import torch
from torch import nn

from dataset.tiny_imagenet import get_dataloaders
from models.custom_net import CustomNet
from utils.device import get_device


# Validation loop
def validate(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0

    correct, total = 0, 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(val_loader):
            inputs, targets = inputs.to(device), targets.to(device)

            # compute outputs and loss
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            val_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    val_loss = val_loss / len(val_loader)
    val_accuracy = 100. * correct / total

    print(f'Validation Loss: {val_loss:.6f} Acc: {val_accuracy:.2f}%')
    return val_loss, val_accuracy


if __name__ == '__main__':
    # Evaluate a saved checkpoint on the val split
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default='checkpoints/best.pth')
    parser.add_argument('--data-dir', default='data/tiny-imagenet-200')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--num-workers', type=int, default=2)
    args = parser.parse_args()

    device = get_device()
    _, val_loader = get_dataloaders(args.data_dir, args.batch_size, args.num_workers)

    model = CustomNet().to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))

    validate(model, val_loader, nn.CrossEntropyLoss(), device)
