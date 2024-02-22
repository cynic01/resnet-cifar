'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms

import time
import argparse
from tqdm import tqdm

from resnet18 import *


def train(net, dataloader, optimizer, criterion, device):
    net.train()
    train_loss = 0
    correct = 0
    total = 0

    # Performance measurement
    dataload_time = 0
    training_time = 0
    if device == 'cuda':
        torch.cuda.synchronize()
    dataload_start = time.perf_counter()

    for inputs, targets in (pbar := tqdm(dataloader)):
        if device == 'cuda':
            torch.cuda.synchronize()
        dataload_time += time.perf_counter() - dataload_start

        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()

        if device == 'cuda':
            torch.cuda.synchronize()
        training_start = time.perf_counter()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        if device == 'cuda':
            torch.cuda.synchronize()
        training_time += time.perf_counter() - training_start

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        pbar.set_description(f'Loss: {train_loss/(pbar.n+1):.3f} | Acc: {100*correct/total:.3f}% ({correct}/{total})')

        if device == 'cuda':
            torch.cuda.synchronize()
        dataload_start = time.perf_counter()
    
    return train_loss / len(dataloader), correct / total, dataload_time, training_time


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--epoch', default=5, type=int, help='number of epochs to train')
    parser.add_argument('--cuda', default=False, action='store_true', help='enable CUDA')
    parser.add_argument('--data_path', default='cifar10', help='name or path to training data')
    parser.add_argument('--num_workers', default=2, type=int, help='number of dataloader workers')
    parser.add_argument('--optimizer', default='sgd', choices=('sgd', 'nesterov', 'adagrad', 'adadelta', 'adam'), help='optimizer to use')
    args = parser.parse_args()

    device = 'cpu'
    if args.cuda:
        if torch.cuda.is_available():
            device = 'cuda'
            print('CUDA enabled')
        else:
            raise Exception('CUDA is not available')

    # Data
    transform_train = transforms.Compose([
        transforms.RandomCrop(size=32, padding=4),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=args.num_workers)

    # classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # Model
    net = ResNet18().to(device)

    criterion = nn.CrossEntropyLoss()
    if args.optimizer == 'sgd':
        optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    elif args.optimizer == 'nestorov':
        optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4, nesterov=True)
    elif args.optimizer == 'adagrad':
        optimizer = optim.Adagrad(net.parameters(), lr=args.lr, weight_decay=5e-4)
    elif args.optimizer == 'adadelta':
        optimizer = optim.Adadelta(net.parameters(), lr=args.lr, weight_decay=5e-4)
    elif args.optimizer == 'adam':
        optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=5e-4)

    # Train
    if device == 'cuda':
        train(net, trainloader, optimizer, criterion, device)
        torch.cuda.synchronize()
    
    epoch_times = []
    for epoch in range(args.epoch):
        print(f'\nEpoch: {epoch+1}')
        if device == 'cuda':
            torch.cuda.synchronize()
        epoch_start = time.perf_counter()

        loss, acc, dataload_time, training_time = train(net, trainloader, optimizer, criterion, device)

        if device == 'cuda':
            torch.cuda.synchronize()
        epoch_times.append(time.perf_counter() - epoch_start)

        print(f'Training loss: {loss}; Top-1 training accuracy: {acc}')
        print(f'Dataloading time: {dataload_time} sec; Training time: {training_time} sec; Epoch time: {epoch_times[-1]} sec')
    
    print(f'Average running time over {args.epoch} epochs: {sum(epoch_times) / args.epoch}')

