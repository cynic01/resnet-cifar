import subprocess

def c2():
    subprocess.run(['python3', 'train.py'])

def c3():
    for num_workers in range(0, 16+1, 4):
        subprocess.run(['python3', 'train.py', '--epoch', '1', '--num_workers', str(num_workers)])

def c4():
    subprocess.run(['python3', 'train.py', '--num_workers', "1"])
    subprocess.run(['python3', 'train.py', '--num_workers', "8"])

def c5():
    subprocess.run(['python3', 'train.py', '--cuda', '--num_workers', "8"])

def q3():
    import resnet18
    net = resnet18.ResNet18()


if __name__ == '__main__':
    # c2()
    c3()
    # c4()
    # c5()
    # q3()