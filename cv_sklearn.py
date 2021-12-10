'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
import numpy as np
import os
import argparse
import random
from sklearn.svm import OneClassSVM
from models import *
from utils import progress_bar
from ocsvm import train_ocsvm

def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    #TODO: Do we need deterministic in cudnn ? Double check
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print ("Seeded everything")
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--gamma', default=0.1, type=float, help='gamma parameter for OCSVM')

parser.add_argument('--resume', type=str,
                    help='resume from checkpoint')
parser.add_argument('--train', action='store_true',
                    help='resume from checkpoint')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
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
    testset, batch_size=1000, shuffle=True, num_workers=2)

testset_svhn = torchvision.datasets.SVHN(
    root='./data', split='test', download=True, transform=transform_test)
testloader_svhn = torch.utils.data.DataLoader(
    testset_svhn, batch_size=100, shuffle=True, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

# Model
print('==> Building model..')
# net = VGG('VGG19')
net = ResNet18()
# net = PreActResNet18()
# net = GoogLeNet()
# net = DenseNet121()
# net = ResNeXt29_2x64d()
# net = MobileNet()
# net = MobileNetV2()
# net = DPN92()
# net = ShuffleNetG2()
# net = SENet18()
# net = ShuffleNetV2(1)
# net = EfficientNetB0()
# net = RegNetX_200MF()
# net = SimpleDLA()
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load(args.resume)
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)


# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs, features = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))


def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    all_features = None
    set_seed(0)
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            if batch_idx > 2:
                break
            inputs, targets = inputs.to(device), targets.to(device)
            outputs, features = net(inputs)
            if all_features is None:
                all_features = outputs.cpu().detach().numpy()
            else:
                all_features = np.concatenate((all_features, outputs.cpu().detach().numpy()))
            print("Start fitting")
#             train_ocsvm(features.cpu().detach().numpy())
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

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
        torch.save(state, './checkpoint/ckpt.pth')
        best_acc = acc
    return all_features

def test_svhn(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    all_features = None
    set_seed(0)
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader_svhn):
            if batch_idx > 2:
                break
            inputs, targets = inputs.to(device), targets.to(device)
            outputs, features = net(inputs)
            if all_features is None:
                all_features = outputs.cpu().detach().numpy()
            else:
                all_features = np.concatenate((all_features, outputs.cpu().detach().numpy()))
            print("Start fitting")
#             train_ocsvm(features.cpu().detach().numpy())
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    return all_features

if args.train:
    for epoch in range(start_epoch, start_epoch+200):
        train(epoch)
        test(epoch)
        scheduler.step()
else:
    cifar_features = test(0)
    svhn_features = test_svhn(0)
    all_features = np.concatenate((cifar_features, svhn_features))
    clf = OneClassSVM( kernel="rbf", gamma = args.gamma).fit(all_features)
    predicted_classes = clf.predict(all_features)
    outlier_indices = []
    for i in range(len(predicted_classes)):
        if predicted_classes[i] < 0:
            print("Outlier index: {}".format(i))
            outlier_indices.append(i)
    outlier_indices = np.array(outlier_indices)
    tp = len(np.where( outlier_indices > 3000 )[0])
    recall = tp / 300
    precision = tp / len(outlier_indices)
    print(recall)
    print(precision)
    print(all_features.shape)
    
