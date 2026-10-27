import argparse
import os

import torch
from torch import nn

from dataset.tiny_imagenet import get_dataloaders
from eval import validate
from models.custom_net import CustomNet
from utils.device import get_device


def train(epoch, model, train_loader, criterion, optimizer, device, log_interval=100):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)

        # compute outputs and loss
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        #backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        # progress inside the epoch, so a long epoch does not look stuck
        if batch_idx % log_interval == 0:
            print(f'  [{batch_idx}/{len(train_loader)}] loss: {loss.item():.4f}')

    train_loss = running_loss / len(train_loader)
    train_accuracy = 100. * correct / total
    print(f'Train Epoch: {epoch} Loss: {train_loss:.6f} Acc: {train_accuracy:.2f}%')
    return train_loss, train_accuracy


def main(args):
    device = get_device()
    print(f'Using device: {device}')

    train_loader, val_loader = get_dataloaders(args.data_dir, args.batch_size, args.num_workers)

    model = CustomNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    if args.wandb:
        import wandb
        wandb.init(project=args.project, config=vars(args))

    best_acc = 0
    os.makedirs('checkpoints', exist_ok=True)

    # Run the training process for {num_epochs} epochs
    for epoch in range(1, args.epochs + 1):
        train_loss, train_accuracy = train(epoch, model, train_loader, criterion, optimizer, device)

        # At the end of each training iteration, perform a validation step
        val_loss, val_accuracy = validate(model, val_loader, criterion, device)

        if args.wandb:
            wandb.log({
                'epoch': epoch,
                'train/loss': train_loss,
                'train/accuracy': train_accuracy,
                'val/loss': val_loss,
                'val/accuracy': val_accuracy,
            })

        # Best validation accuracy: keep the weights of the best epoch
        if epoch == 1 or val_accuracy > best_acc:
            best_acc = val_accuracy
            torch.save(model.state_dict(), 'checkpoints/best.pth')

    print(f'Best validation accuracy: {best_acc:.2f}%')

    if args.wandb:
        wandb.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='data/tiny-imagenet-200')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--num-workers', type=int, default=2)
    parser.add_argument('--wandb', action='store_true', help='log metrics to Weights & Biases')
    parser.add_argument('--project', default='aml-lab3')
    main(parser.parse_args())
