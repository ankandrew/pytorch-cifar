'''Train CIFAR10 with PyTorch.'''
import argparse
import os

import torch.backends.cudnn as cudnn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from models import *
from ols.label_smooth import LabelSmoothingLoss
from ols.online_label_smooth import OnlineLabelSmoothing
from utils import progress_bar, model_to_class


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--resume', action='store_true', help='resume from checkpoint')

    parser.add_argument('--model', choices=list(model_to_class.keys()), required=True, help='Model to use')

    parser.add_argument('--loss', choices=['ce', 'ls', 'ols'], required=True, help='Loss fn to use')
    parser.add_argument('--alpha', type=float, required=False, default=0.5, help='Alpha for ols')
    parser.add_argument('--smooth', type=float, required=False, default=0.1,
                        help='Initial smoothing for 1st epoch for ols')
    parser.add_argument('--decay_a', type=float, required=False, default=0.0,
                        help='Decrease to the alpha balancing term.')
    parser.add_argument('--decay_n', type=int, required=False, default=1,
                        help='Every n epochs apply hard_decay_factor.')

    args = parser.parse_args()

    # # DEBUG
    # args = parser.parse_args([
    #     '--model', 'resnet18',
    #     '--loss', 'ols',
    #     '--alpha', '0.5',
    #     '--smooth', '0.1',
    #     '--decay_a', '0.0',
    #     '--decay_n', '1',
    # ])

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    best_acc = 0  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch

    # Data
    CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
    CIFAR10_STD = (0.2023, 0.1994, 0.2010)

    print('==> Preparing data..')
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=128, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=100, shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck')

    # Model
    print(f'==> Building {args.model} model ..')
    net = model_to_class[args.model]()
    net = net.to(device)
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    save_filename = f'{args.model}-{args.loss}'
    if args.loss == 'ls':
        save_filename += f'-smooth_{args.smooth}'
    elif args.loss == 'ols':
        save_filename += f'-alpha_{args.alpha}-smooth_{args.smooth}-a_{args.decay_a}-n_{args.decay_n}'
    save_filename += '.pth'
    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load(f'./checkpoint/{save_filename}')
        net.load_state_dict(checkpoint['net'])
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch']

    if args.loss == 'ce':
        criterion = nn.CrossEntropyLoss()
    elif args.loss == 'ls':
        criterion = LabelSmoothingLoss(classes=len(classes), smoothing=args.smooth)
        print(f'{"#" * 10}LS{"#" * 10}\nsmoothing={args.smooth}\n{"#" * 30}')
    else:
        criterion = OnlineLabelSmoothing(alpha=args.alpha, n_classes=len(classes), smoothing=args.smooth,
                                         hard_decay_factor=args.decay_a, hard_decay_epochs=args.decay_n)
        print(f'{"#" * 11}OLS{"#" * 11}\nalpha={args.alpha}, smoothing={args.smooth}\n{"#" * 30}')
        criterion = criterion.to(device)

    optimizer = optim.SGD(net.parameters(), lr=args.lr,
                          momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)


    # Training
    def train(epoch):
        if isinstance(criterion, OnlineLabelSmoothing):
            criterion.train()
        print('\nEpoch: %d' % epoch)
        net.train()
        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(trainloader):
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

            progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))


    def test(epoch):
        global best_acc
        net.eval()
        if isinstance(criterion, OnlineLabelSmoothing):
            criterion.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(testloader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = net(inputs)
                loss = criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                             % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))

        # Save checkpoint.
        acc = 100. * correct / total
        if acc > best_acc:
            print('Saving..')
            state = {
                'net': net.state_dict(),
                'acc': acc,
                'epoch': epoch,
            }
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            torch.save(state, f'./checkpoint/{save_filename}')
            best_acc = acc


    for epoch in range(start_epoch, start_epoch + 200):
        train(epoch)
        test(epoch)
        scheduler.step()
        if isinstance(criterion, OnlineLabelSmoothing):
            criterion.next_epoch()
    print(f'{"#"*9}Best accuracy{"#"*8} \n{best_acc}\n{"#"*30}')
