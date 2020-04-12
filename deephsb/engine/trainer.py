import os, sys
import random
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from deephsb.model import GameModelPooling
from deephsb.dataset import GameDataset
from deephsb.utils import *

from .solver import build_optimizer, build_lr_scheduler

class Trainer(object):
    def __init__(self, cfg, device, output_dir='outputs/'):
        self.cfg = cfg
        self.device = device
        self.output_dir = output_dir

    def cross_validate(self):

        TRAIN_LOSS = []
        TRAIN_ACCU = []
        VAL_LOSS = []
        VAL_ACCU = []

        for idx in range(self.cfg.N_FOLD):
            prYellow('Start training for the %d/%d cross-validation fold.' % (idx+1, self.cfg.N_FOLD))

            prGreen('==> Building model..')
            model = GameModelPooling(in_planes=self.cfg.MODEL.IN_PLANES,
                                     out_planes=self.cfg.MODEL.OUT_PLANES,
                                     kernels=self.cfg.MODEL.KERNELS, 
                                     mode=self.cfg.MODEL.POOLING, 
                                     bias=self.cfg.MODEL.BIAS,
                                     non_local=self.cfg.MODEL.NON_LOCAL, 
                                     residual=self.cfg.MODEL.RESIDUAL)
            model_name = model.__class__.__name__
            print('model: ', model_name)
            print('Total number of parameters: ', end='')
            total_num_param = sum([param.nelement() for param in model.parameters()])
            prGreen(total_num_param)

            model = model.to(self.device)
            optimizer = build_optimizer(self.cfg, model)
            scheduler = build_lr_scheduler(self.cfg, optimizer)

            train_dataset = GameDataset(fold_index=idx, mode='train', 
                                        prob_file=self.cfg.DATASET.TRAIN_PROB_FILE, 
                                        feature_file=self.cfg.DATASET.TRAIN_FEATURE_FILE,
                                        N_FOLD=self.cfg.N_FOLD)

            val_dataset = GameDataset(fold_index=idx, mode='val', 
                                      prob_file=self.cfg.DATASET.TRAIN_PROB_FILE, 
                                      feature_file=self.cfg.DATASET.TRAIN_FEATURE_FILE,
                                      N_FOLD=self.cfg.N_FOLD)

            print('Number of training samples: ', len(train_dataset))
            print('Number of val samples: ', len(val_dataset))
            train_loader = torch.utils.data.DataLoader(
                    train_dataset, batch_size=self.cfg.SOLVER.SAMPLES_PER_BATCH, shuffle=True,
                    num_workers=1, pin_memory=True)
            val_loader = torch.utils.data.DataLoader(
                    val_dataset, batch_size=10, shuffle=False,
                    num_workers=1, pin_memory=True)

            for epoch in range(self.cfg.SOLVER.TOTAL_EPOCH):
                train_loss, train_accu = self.train(train_loader, model, optimizer, epoch)
                val_loss, val_accu = self.val(val_loader, model, epoch)
                scheduler.step()

            TRAIN_LOSS.append(train_loss)   
            TRAIN_ACCU.append(train_accu)   
            VAL_LOSS.append(val_loss)
            VAL_ACCU.append(val_accu)

        prYellow('Save training/validation results===>')
        out_path = os.path.join(self.output_dir, 'cross_validation')
        print(out_path)
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        with open('%s/results_%s_%s.txt' % (out_path, self.cfg.MODEL.POOLING, self.cfg.MODEL.NON_LOCAL), 'a+') as f:
            f.write('%s\t%d\t%s\t%d\n' % (self.cfg.MODEL.POOLING, self.cfg.MODEL.KERNELS, self.cfg.MODEL.NON_LOCAL, total_num_param))
            for ii in range(len(TRAIN_LOSS)):
                f.write('%f %f %f %f\n' % (TRAIN_LOSS[ii], TRAIN_ACCU[ii], VAL_LOSS[ii], VAL_ACCU[ii]))
            f.write('\n')
        f.close()

    def train(self, train_loader, model, optimizer, epoch):
        augmentation = self.cfg.DATASET.AUGMENTATION
        print('Epoch: %03d/%d' % (epoch, self.cfg.SOLVER.TOTAL_EPOCH), end='\t')

        total_loss = 0.0
        total_num = 0
        total_accu = 0

        model.train()
        for i, batch in enumerate(train_loader): 
            sample, label = batch['sample'], batch['label']
            label = label.unsqueeze(1)
            sample, label = sample.to(self.device), label.to(self.device)

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

    def val(self, val_loader, model, epoch):
        print('Epoch: %03d/%d' % (epoch, self.cfg.SOLVER.TOTAL_EPOCH), end='\t')

        total_loss = 0.0
        total_num = 0
        total_accu = 0

        model.eval()
        with torch.no_grad():
            for i, batch in enumerate(val_loader): 
                sample, label = batch['sample'], batch['label']
                label = label.unsqueeze(1)

                sample, label = sample.to(self.device), label.to(self.device)
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

    def test(self, test_loader, model, fl):
        model.eval()
        with torch.no_grad():
            for i, batch in enumerate(test_loader): 
                sample = batch['sample']
                sample = sample.to(self.device)
                output = model(sample)
                row_prob = output.sum(3)

                action = torch.argmax(row_prob, 2).squeeze() + 1
                action = action.detach().cpu().numpy()
                row_prob = row_prob.detach().squeeze().cpu().numpy()
                for k in range(row_prob.shape[0]):
                    print(row_prob[k], action[k])
                    fl.write('%f,%f,%f,%d\n' % (row_prob[k][0], row_prob[k][1], row_prob[k][2], action[k]))

    def inference(self):
        prYellow('Start training on all data for inference.')

        prGreen('==> Building model..')
        model = GameModelPooling(in_planes=self.cfg.MODEL.IN_PLANES,
                                 out_planes=self.cfg.MODEL.OUT_PLANES,
                                 kernels=self.cfg.MODEL.KERNELS, 
                                 mode=self.cfg.MODEL.POOLING, 
                                 bias=self.cfg.MODEL.BIAS,
                                 non_local=self.cfg.MODEL.NON_LOCAL, 
                                 residual=self.cfg.MODEL.RESIDUAL)
        model_name = model.__class__.__name__
        print('model: ', model_name)
        print('Total number of parameters: ', end='')
        total_num_param = sum([param.nelement() for param in model.parameters()])
        prGreen(total_num_param)

        model = model.to(self.device)
        optimizer = build_optimizer(self.cfg, model)
        scheduler = build_lr_scheduler(self.cfg, optimizer)

        train_dataset = GameDataset(mode='train_all', 
                                    prob_file=self.cfg.DATASET.TRAIN_PROB_FILE, 
                                    feature_file=self.cfg.DATASET.TRAIN_FEATURE_FILE)
        test_dataset = GameDataset(mode='test', 
                                   prob_file=None, 
                                   feature_file=self.cfg.DATASET.TRAIN_FEATURE_FILE)
                                   
        print('Number of training samples: ', len(train_dataset))
        print('Number of val samples: ', len(test_dataset))
        train_loader = torch.utils.data.DataLoader(
                train_dataset, batch_size=self.cfg.SOLVER.SAMPLES_PER_BATCH, shuffle=True,
                num_workers=1, pin_memory=True)
        test_loader = torch.utils.data.DataLoader(
                test_dataset, batch_size=10, shuffle=False,
                num_workers=1, pin_memory=True)

        for epoch in range(self.cfg.SOLVER.TOTAL_EPOCH):
            train_loss, train_accu = self.train(train_loader, model, optimizer, epoch)
            scheduler.step()

        prYellow('Finished training. Run inference ===>')
        out_path = os.path.join(self.output_dir, 'inference')
        print(out_path)
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        fl = open('%s/inference.csv' % (out_path), 'a+')
        fl.write('f1,f2,f3,action\n')
        self.test(test_loader, model, fl)
        fl.close()
        prYellow('Inference is finished!')
