import os, sys
import random
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from model import GameModelPooling
from dataset import N_FOLD, GameDataset
from utils.utils import *

device = ('cuda' if torch.cuda.is_available() else 'cpu')
print('Device: ', device)

parser = argparse.ArgumentParser(description='Human Behavior Prediction')
parser.add_argument('--num-kernels', default=128, type=int, help='number of kernels')
parser.add_argument('--pooling', default='avg_pool', type=str, help='pooling mode')
parser.add_argument('--non-local', action='store_true', help='use non-local module')
# parser.add_argument('--bn', action='store_true', help='use batch normalization')
parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
parser.add_argument('--wd', default=5e-4, type=float, help='weight decay')
parser.add_argument('--epoch', default=100, type=int, help='training epochs')
parser.add_argument('--batch-size', type=int, default=16, metavar='N',
                    help='input batch size for training (default: 16)')
parser.add_argument('--aug', action='store_true', help='use data augmentation') 
parser.add_argument('--inference', action='store_true', help='inference on test data') 
args = parser.parse_args()
print(args)

def main():

    TRAIN_LOSS = []
    TRAIN_ACCU = []
    VAL_LOSS = []
    VAL_ACCU = []

    for idx in range(N_FOLD):
        prYellow('Start training for the %d/%d cross-validation fold.' % (idx+1, N_FOLD))

        prGreen('==> Building model..')
        model = GameModelPooling(kernels=args.num_kernels, mode=args.pooling, non_local=args.non_local, residual=True)
        model_name = model.__class__.__name__
        print('model: ', model_name)
        print('Total number of parameters: ', end='')
        total_num_param = sum([param.nelement() for param in model.parameters()])
        prGreen(total_num_param)

        model = model.to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.wd)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[args.epoch-40, args.epoch-20], gamma=0.1)

        train_dataset = GameDataset(fold_index=idx, mode='train', prob_file='data/v2/hb_train_truth.csv', feature_file='data/v2/hb_train_feature.csv')
        val_dataset = GameDataset(fold_index=idx, mode='val', prob_file='data/v2/hb_train_truth.csv', feature_file='data/v2/hb_train_feature.csv')
        print('Number of training samples: ', len(train_dataset))
        print('Number of val samples: ', len(val_dataset))
        train_loader = torch.utils.data.DataLoader(
                train_dataset, batch_size=args.batch_size, shuffle=True,
                num_workers=1, pin_memory=True)
        val_loader = torch.utils.data.DataLoader(
                val_dataset, batch_size=10, shuffle=False,
                num_workers=1, pin_memory=True)

        for epoch in range(args.epoch):
            train_loss, train_accu = train(train_loader, model, optimizer, epoch)
            val_loss, val_accu = val(val_loader, model, epoch)
            scheduler.step()

        TRAIN_LOSS.append(train_loss)   
        TRAIN_ACCU.append(train_accu)   
        VAL_LOSS.append(val_loss)
        VAL_ACCU.append(val_accu)

    prYellow('Save training/validation results===>')
    if not os.path.exists('results/'):
        os.mkdir('results/')
    with open('results/results_%s_%s.txt' % (args.pooling, args.non_local), 'a+') as f:
        f.write('%s\t%d\t%s\t%d\n' % (args.pooling, args.num_kernels, args.non_local, total_num_param))
        for ii in range(len(TRAIN_LOSS)):
            f.write('%f %f %f %f\n' % (TRAIN_LOSS[ii], TRAIN_ACCU[ii], VAL_LOSS[ii], VAL_ACCU[ii]))
        f.write('\n')
    f.close()

def train(train_loader, model, optimizer, epoch, augmentation=args.aug):
    print('Epoch: %03d/%d' % (epoch, args.epoch), end='\t')

    total_loss = 0.0
    total_num = 0
    total_accu = 0

    model.train()
    for i, batch in enumerate(train_loader): 
        sample, label = batch['sample'], batch['label']
        label = label.unsqueeze(1)
        sample, label = sample.to(device), label.to(device)

        if augmentation and random.random() < 0.5: 
            sample = torch.flip(sample, [1])
            sample = torch.transpose(sample, 2, 3)
            output = model(sample)
            row_prob = output.sum(2)
        else:
            output = model(sample)
            row_prob = output.sum(3)

        loss = F.mse_loss(row_prob, label, reduction='mean')

        # optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # training log
        total_loss += loss.item() * sample.size()[0]
        total_num += sample.size()[0]
        total_accu += (torch.argmax(row_prob, 2).squeeze() == torch.argmax(label, 2).squeeze()).sum()

    total_loss = total_loss / total_num
    total_accu = float(total_accu) / total_num
    print('Training loss: ', total_loss, end='\t')
    print('Training accuracy: ', total_accu)

    return total_loss, total_accu

def val(val_loader, model, epoch):
    print('Epoch: %03d/%d' % (epoch, args.epoch), end='\t')

    total_loss = 0.0
    total_num = 0
    total_accu = 0

    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(val_loader): 
            sample, label = batch['sample'], batch['label']
            label = label.unsqueeze(1)

            sample, label = sample.to(device), label.to(device)
            output = model(sample)
            row_prob = output.sum(3)

            loss = F.mse_loss(row_prob, label, reduction='mean')

            # validation log
            total_loss += loss.item() * sample.size()[0]
            total_num += sample.size()[0]
            total_accu += (torch.argmax(row_prob, 2).squeeze() == torch.argmax(label, 2).squeeze()).sum()

        total_loss = total_loss / total_num
        total_accu = float(total_accu) / total_num
        print('Val loss: ', total_loss, end='\t')
        print('Val accuracy: ', total_accu)

        return total_loss, total_accu

def test(test_loader, model, fl):
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(test_loader): 
            sample = batch['sample']
            sample = sample.to(device)
            output = model(sample)
            row_prob = output.sum(3)

            action = torch.argmax(row_prob, 2).squeeze() + 1
            action = action.detach().cpu().numpy()
            row_prob = row_prob.detach().squeeze().cpu().numpy()
            for k in range(row_prob.shape[0]):
                print(row_prob[k], action[k])
                fl.write('%f,%f,%f,%d\n' % (row_prob[k][0], row_prob[k][1], row_prob[k][2], action[k]))

def inference():
    prYellow('Start training on all data for inference.')

    prGreen('==> Building model..')
    model = GameModelPooling(kernels=args.num_kernels, mode=args.pooling, non_local=args.non_local, residual=True)
    model_name = model.__class__.__name__
    print('model: ', model_name)
    print('Total number of parameters: ', end='')
    total_num_param = sum([param.nelement() for param in model.parameters()])
    prGreen(total_num_param)

    model = model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.wd)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[args.epoch-40, args.epoch-20], gamma=0.1)

    train_dataset = GameDataset(mode='train_all', prob_file='data/v2/hb_train_truth.csv', feature_file='data/v2/hb_train_feature.csv')
    test_dataset = GameDataset(mode='test', prob_file=None, feature_file='data/v2/hb_test_feature.csv')
    print('Number of training samples: ', len(train_dataset))
    print('Number of val samples: ', len(test_dataset))
    train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True,
            num_workers=1, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=10, shuffle=False,
            num_workers=1, pin_memory=True)

    for epoch in range(args.epoch):
        train_loss, train_accu = train(train_loader, model, optimizer, epoch)
        scheduler.step()

    prYellow('Finished training. Run inference ===>')
    if not os.path.exists('results/'):
        os.mkdir('results/')
    fl = open('results/inference.csv', 'a+')
    fl.write('f1,f2,f3,action\n')
    test(test_loader, model, fl)
    fl.close()
    prYellow('Done!')

if __name__ == '__main__':
    if args.inference:
        inference()
    else:
        main()
