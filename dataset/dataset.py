from __future__ import print_function, division
import numpy as np
import random

import torch
import torch.utils.data

N_FOLD = 5

class GameDataset(torch.utils.data.Dataset):
    def __init__(self,
                 mode='train',
                 feature_file='hb_train_feature.csv', 
                 prob_file=None,
                 fold_index=0):

        self.mode = mode
        assert self.mode in ['train', 'train_all', 'val', 'test']
        feature = np.genfromtxt(feature_file, delimiter=',', skip_header=1)
        self.mean, self.std = feature.mean(), feature.std()
        row_feature = feature[:,:9].reshape((-1,3,3))
        col_feature = feature[:,9:].reshape((-1,3,3))
        payoff_matrix = np.stack([row_feature, col_feature], axis=1)
        if self.mode != 'test':
            payoff_matrix_split = np.split(payoff_matrix, N_FOLD, axis=0)

        if prob_file is not None:
            col_prob = np.genfromtxt(prob_file, delimiter=',', skip_header=1)
            col_action = col_prob[:,-1]
            col_prob = col_prob[:,:3]
            col_action_split = np.split(col_action, N_FOLD, axis=0)
            col_prob_split = np.split(col_prob, N_FOLD, axis=0)

        if self.mode == 'train':
            selected_indices = np.array(range(N_FOLD))
            selected_indices = np.delete(selected_indices, fold_index)
            input_data = np.concatenate([payoff_matrix_split[x] for x in selected_indices]).astype(np.float32)
            input_prob = np.concatenate([col_prob_split[x] for x in selected_indices]).astype(np.float32)
            input_action = np.concatenate([col_action_split[x] for x in selected_indices]).astype(np.uint8)
            self.input_data = torch.from_numpy(input_data)
            self.input_prob = torch.from_numpy(input_prob)
            self.input_action = input_action

        elif self.mode == 'val':
            selected_indices = fold_index
            input_data = (payoff_matrix_split[selected_indices]).astype(np.float32)
            input_prob = (col_prob_split[selected_indices]).astype(np.float32)
            input_action = (col_action_split[selected_indices]).astype(np.uint8)
            self.input_data = torch.from_numpy(input_data)
            self.input_prob = torch.from_numpy(input_prob)
            self.input_action = input_action

        elif self.mode == 'train_all':
            payoff_matrix = payoff_matrix.astype(np.float32)
            self.input_data = torch.from_numpy(payoff_matrix)
            self.input_prob = torch.from_numpy(col_prob.astype(np.float32))
            self.input_action = col_action.astype(np.uint8)

        elif self.mode == 'test':
            payoff_matrix = payoff_matrix.astype(np.float32)
            self.input_data = torch.from_numpy(payoff_matrix)
                
    def __getitem__(self, index):
        if self.mode == 'test':
            sample = self.input_data[index]
            # normalization
            sample = sample / sample.max()
            return {'sample': sample}
        else:
            sample = self.input_data[index]
            # normalization
            sample = sample / sample.max()
            label = self.input_prob[index]
            action = self.input_action[index]
            return {'sample': sample,
                    'label': label,
                    'action': action}

    def __len__(self):  # number of possible position
        return self.input_data.size()[0]
