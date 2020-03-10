import os, sys
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from model import GameModelPooling
from dataset import N_FOLD, GameDataset
from utils.utils import *

EPOCH = 100
BATCH_SIZE = 16
LR = 0.01
NUM_KERNELS = 32
POOLING = 'max_pool'

device = ('cuda' if torch.cuda.is_available() else 'cpu')
print('Device: ', device)

def main():

    TRAIN_LOSS = []
    TRAIN_ACCU = []
    VAL_LOSS = []
    VAL_ACCU = []

    for idx in range(N_FOLD):
        prYellow('Start training for the %d/%d cross-validation fold.' % (idx+1, N_FOLD))

        prGreen('==> Building model..')
        model = GameModelPooling(kernels=NUM_KERNELS, use_bn=True, mode=POOLING, non_local=False)
        model_name = model.__class__.__name__
        print('model: ', model_name)
        print('Total number of parameters: ', end='')
        prGreen(sum([param.nelement() for param in model.parameters()]))
        model = model.to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=1e-3)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[EPOCH-40, EPOCH-20], gamma=0.1)

        train_dataset = GameDataset(fold_index=idx, mode='train', prob_file='data/hb_train_truth.csv', feature_file='data/hb_train_feature.csv')
        val_dataset = GameDataset(fold_index=idx, mode='val', prob_file='data/hb_train_truth.csv', feature_file='data/hb_train_feature.csv')
        print('Number of training samples: ', len(train_dataset))
        print('Number of val samples: ', len(val_dataset))
        train_loader = torch.utils.data.DataLoader(
                train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                num_workers=1, pin_memory=True)
        val_loader = torch.utils.data.DataLoader(
                val_dataset, batch_size=10, shuffle=False,
                num_workers=1, pin_memory=True)

        for epoch in range(EPOCH):
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
    with open('results/results.txt', 'a+') as f:
        f.write('%s\t%d\n' % (POOLING, NUM_KERNELS))
        for ii in range(len(TRAIN_LOSS)):
            f.write('%f %f %f %f\n' % (TRAIN_LOSS[ii], TRAIN_ACCU[ii], VAL_LOSS[ii], VAL_ACCU[ii]))
        f.write('\n')
    f.close()

def train(train_loader, model, optimizer, epoch, augmentation=False):
    print('Epoch: %03d/%d' % (epoch, EPOCH), end='\t')

    total_loss = 0.0
    total_num = 0
    total_accu = 0

    model.train()
    for i, batch in enumerate(train_loader): 
        sample, label = batch['sample'], batch['label']
        label = label.unsqueeze(1)

        sample, label = sample.to(device), label.to(device)
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
    print('Epoch: %03d/%d' % (epoch, EPOCH), end='\t')

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

if __name__ == '__main__':
    main()