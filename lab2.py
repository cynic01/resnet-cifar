import subprocess

def c2():
    subprocess.run(['python3', 'train.py', '--cuda'])

def c3():
    for num_workers in range(0, 20+1, 4):
        subprocess.run(['python3', 'train.py', '--cuda', '--epoch', '1', '--num_workers', str(num_workers)])

def c4():
    subprocess.run(['python3', 'train.py', '--cuda', '--num_workers', "1"])
    subprocess.run(['python3', 'train.py', '--cuda', '--num_workers', "4"])

def c5():
    subprocess.run(['python3', 'train.py', '--cuda', '--num_workers', "4"])
    subprocess.run(['python3', 'train.py', '--num_workers', "4"])

def c6():
    for optimizer in ('sgd', 'nesterov', 'adagrad', 'adadelta', 'adam'):
        subprocess.run(['python3', 'train.py', '--cuda', '--num_workers', "4", '--optimizer', optimizer])

def c7():
    subprocess.run(['python3', 'train.py', '--cuda', '--no_bn', '--num_workers', "4"])

def q3():
    import torch, torchvision
    import resnet18

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Data
    transform_train = torchvision.transforms.Compose([
        torchvision.transforms.RandomCrop(size=32, padding=4),
        torchvision.transforms.RandomHorizontalFlip(p=0.5),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=4)

    # Model
    net = resnet18.ResNet18().to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)

    # Train
    net.train()
    inputs, targets = next(iter(trainloader))
    inputs, targets = inputs.to(device), targets.to(device)
    optimizer.zero_grad()
    outputs = net(inputs)
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()

    # Calculate Number of Trainable Parameters and Gradients
    num_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    num_grad = sum(p.grad.numel() for p in net.parameters() if p.requires_grad)
    print('Number of Trainable Parameters:', num_params)
    print('Number of Gradients:', num_grad)


if __name__ == '__main__':
    c2()
    c3()
    c4()
    c5()
    c6()
    c7()
    q3()