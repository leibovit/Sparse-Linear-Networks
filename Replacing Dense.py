import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse
from tqdm import tqdm

import numpy as np

parser = argparse.ArgumentParser(description='Replacing Dense Linear Layers By The Proposed Architecture')
                                 
parser.add_argument('--lr', default=0.001, type=float, help='learning rate')

parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--batch-size', default=128, type=int,
                    metavar='N', help='mini-batch size (default: 128)')
                                 
parser.add_argument('--butterfly', default="True",type=str,
                    help='to use the butterfly replacement or not (default: "True")')  

parser.add_argument('--model', default='ResNet18', type=str,
                     help='model (default: ResNet18), other options are EfficientNet and PreActResNet')

parser.add_argument('--dataset', default='CIFAR10', type=str,
                    help='dataset (default: (CIFAR10), other options are CIFAR100')


args = parser.parse_args()

print (args)

if args.butterfly == 'True':

    args_butterfly = True
    print  ('using butterfly replacment')
    
else:
    
    args_butterfly = False  
    print (' NOT using butterfly replacment')
    
args_data = args.dataset
args_model = args.model
args_lr = args.lr
args_epochs = args.epochs


# Training

name = 'long'

def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    batch_idx = 0
    with tqdm(trainloader, unit="batch",position=0, leave=True) as tepoch:
        for inputs, targets in tepoch:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            batch_idx +=1 

            tepoch.set_postfix(loss=train_loss/(batch_idx), accuracy= 100.*correct/total, correct = correct, total = total)


def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    batch_idx = 0
    with torch.no_grad():
        with tqdm(testloader, unit="batch",position=0, leave=True) as tepoch:
            for inputs, targets in tepoch:
                inputs, targets = inputs.to(device), targets.to(device)
                
                outputs = net(inputs)
                loss = criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                batch_idx +=1 
                tepoch.set_postfix(loss=test_loss/(batch_idx), accuracy= 100.*correct/total, correct = correct, total = total)

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.pth' + name)
        best_acc = acc




device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print ('Using {} device'.format(device))
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch



# Data

if args_data == 'CIFAR100':
    
    print('==> Preparing dataset %s')
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    dataloader = torchvision.datasets.CIFAR100
    num_classes = 100
    trainset = dataloader(root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, num_workers=4)
    testset = dataloader(root='./data', train=False, download=False, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=1000, shuffle=False, num_workers=4)

if args_data == 'CIFAR10':

    print('==> Preparing data..')
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=128, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=100, shuffle=False, num_workers=2)

    num_classes = 10
    classes = ('plane', 'car', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck')



import ResNet, EfficientNet, PreActResNet

# Model
print('==> Building model..')

if args_model == 'ResNet18':
    net = ResNet.ResNet18(butterfly_layer = args_butterfly,num_classes=num_classes).to(device);
    
if args_model == 'EfficientNet':
    net = EfficientNet.EfficientNetB0(butterfly_layer = args_butterfly,num_classes=num_classes).to(device);
    
if args_model == 'PreActResNet18':
    net = PreActResNet.PreActResNet18(butterfly_layer = args_butterfly,num_classes= num_classes).to(device);


criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), args_lr,
                      momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
best_acc = 0
for epoch in range(start_epoch, start_epoch+ args_epochs):
    train(epoch)
    test(epoch)
  
    scheduler.step()

#python Replacing\ Dense.py --lr 0.001 --epochs 200 --butterfly True --model "ResNet18" --dataset "CIFAR10"