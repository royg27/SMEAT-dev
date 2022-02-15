import time
import argparse
import os
import torch
import numpy as np
import random
import wandb
from torchvision import transforms
import torchvision.datasets as datasets
from models.resnet import WideResNet, PreActResNet
from tqdm import tqdm
from torch.optim.lr_scheduler import MultiStepLR

parser = argparse.ArgumentParser(description='SMEAT: Single Model Ensemble Adversarial Training')
parser.add_argument('--model', default="wrn", type=str,
                    help='model type: wrn for wrn-28-10 or rn for PreAct ResNet-18')
parser.add_argument('--ensemble_size', default=3, type=int, help='ensemble_size')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--batch_size', default=512, type=int, help='batch size')
parser.add_argument('--batch_repetition', default=4, type=int, help='batch size')
parser.add_argument('--seed', default=1, type=int, help='batch size')
parser.add_argument('--epochs', default=250, type=int, help='total epochs to run')
parser.add_argument('--decay', default=3e-4, type=float, help='weight decay')
parser.add_argument('--gamma', default=0.1, type=float, help='lr drop rate')
parser.add_argument('--epsilon', default=1, type=float, help='Relevant for adversarial and NoClass methods')
parser.add_argument('--wandb', default=1, type=int, help='upload to WANDB')


args, unknown = parser.parse_known_args()
timestr = time.strftime("%Y%m%d-%H%M%S")

run_name = f'model_{args.model}_ensemble_size{args.ensemble_size}_{timestr}'

print('Chosen args:')
print(args)

# Integration with WANDB
if not args.wandb:
    os.environ['WANDB_MODE'] = 'offline'
else:
    os.environ['WANDB_MODE'] = 'online'


config = vars(args)
wandb.init(
    project='SMEAT',
    entity="royg",
    config=config
)

wandb.run.name = run_name

use_cuda = torch.cuda.is_available()

# Reproducibility
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
random.seed(args.seed)
np.random.seed(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Data

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
])

trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=1,
                                          pin_memory=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

# Create Model

if args.model == 'rn':
    model = PreActResNet(ensemble_size=args.ensemble_size)
elif args.model == 'wrn':
    model = WideResNet(ensemble_size=args.ensemble_size)
else:
    raise Exception(f'{args.model} is an invalid option. Should be rn or wrn.')

model = torch.nn.DataParallel(model).cuda()
wandb.watch(model)

# create optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.decay)
scheduler = MultiStepLR(optimizer, milestones=[80, 160, 180], gamma=0.1)

# Train
def train():
    correct = 0
    total = 0
    #
    model.train()
    for batch_idx, (inputs, targets) in tqdm(enumerate(trainloader)):
        b_size = targets.shape[0]
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs = torch.tile(inputs, (args.ensemble_size * args.batch_repetition, 1, 1, 1))
        targets = torch.tile(targets, (args.ensemble_size * args.batch_repetition, ))
        rand_idx = torch.randperm(b_size * args.ensemble_size * args.batch_repetition)
        inputs = inputs[rand_idx]
        targets = targets[rand_idx]
        inp_list = [inputs[i * b_size * args.batch_repetition: (i + 1) * b_size * args.batch_repetition] for i in range(args.ensemble_size)]
        inputs = torch.cat(inp_list, dim=1)
        '''
        import matplotlib.pyplot as plt
        class_dict = {0: 'airplane',
              1: 'car',
              2: 'bird',
              3: 'cat',
              4: 'deer',
              5: 'dog',
              6: 'frog',
              7: 'horse',
              8: 'ship',
              9: 'truck'}shuffle_indices = [
          tf.concat([tf.random.shuffle(main_shuffle[:to_shuffle]),
                     main_shuffle[to_shuffle:]], axis=0)
          for _ in range(FLAGS.ensemble_size)]
        plt.figure()
        plt.imshow(inputs[0,:3,:,:].permute(1,2,0).detach().cpu().numpy())
        plt.title(class_dict[targets[0].cpu().item()])
        plt.show()
        plt.figure()
        plt.imshow(inputs[0,3:6,:,:].permute(1,2,0).detach().cpu().numpy())
        plt.title(class_dict[targets[128].cpu().item()])
        plt.show()
        plt.figure()
        plt.imshow(inputs[0,6:9,:,:].permute(1,2,0).detach().cpu().numpy())
        plt.title(class_dict[targets[256].cpu().item()])
        plt.show()
        '''
        logits_list = model(inputs)
        logits = torch.cat(logits_list)
        loss = torch.nn.CrossEntropyLoss()(input=logits, target=targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # calc statistics
        total += logits.shape[0]
        correct += torch.argmax(logits, -1).eq(targets).sum()
        #
    return correct / total


@torch.no_grad()
def test():
    correct = 0
    total = 0
    #
    model.eval()
    for batch_idx, (inputs, targets) in tqdm(enumerate(testloader)):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()

        inputs = torch.tile(inputs, (1, args.ensemble_size, 1, 1,))
        '''
        import matplotlib.pyplot as plt
        class_dict = {0: 'airplane',
              1: 'car',
              2: 'bird',
              3: 'cat',
              4: 'deer',
              5: 'dog',
              6: 'frog',
              7: 'horse',
              8: 'ship',
              9: 'truck'}
        plt.figure()
        plt.imshow(inputs[0,:3,:,:].permute(1,2,0).detach().cpu().numpy())
        plt.title(class_dict[targets[0].cpu().item()])
        plt.show()
        plt.figure()
        plt.imshow(inputs[0,3:6,:,:].permute(1,2,0).detach().cpu().numpy())
        plt.title(class_dict[targets[128].cpu().item()])
        plt.show()
        plt.figure()
        plt.imshow(inputs[0,6:9,:,:].permute(1,2,0).detach().cpu().numpy())
        plt.title(class_dict[targets[256].cpu().item()])
        plt.show()
        '''
        logits_list = model(inputs)
        logits = logits_list[0]
        for i in range(1, args.ensemble_size):
            logits += logits_list[1]
        logits /= args.ensemble_size
        # calc statistics
        total += logits.shape[0]
        correct += torch.argmax(logits, -1).eq(targets).sum()
        #
    return correct / total


def adjust_learning_rate(optimizer, epoch):
    """decrease the learning rate at 100 and 150 epoch"""
    lr = args.lr
    if 'wrn' in args.model:
        if epoch == 0:
            print('LR scheduling of WRN')
        if epoch >= 80:
            lr *= args.gamma
        if epoch >= 160:
            lr *= args.gamma
        if epoch >= 180:
            lr *= args.gamma
    else:
        if epoch == 0:
            print('Vanilla LR scheduling')
        if epoch >= 100:
            lr *= args.gamma
        if epoch >= 150:
            lr *= args.gamma
        if epoch >= 200:
            lr *= args.gamma
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    print(f'setting lr to {lr}')


for epoch in range(args.epochs):
    best_acc = 0
    print(f'Epoch {epoch}')
    train_acc = train()
    test_acc = test()
    print(f'Epoch {epoch} : train acc is {train_acc} | test acc is {test_acc}')
    wandb.log({'train clean acc': train_acc, 'test clean acc': test_acc})
    # adjust_learning_rate(optimizer, epoch)
    scheduler.step()
    # save model
    if test_acc > best_acc:
        best_acc = test_acc
        torch.save(model.state_dict(), f'./checkpoints/' + run_name)
