# -*- coding: utf-8 -*-
"""
Created on Mon Sep 26 17:41:58 2022

@author: emibu
"""
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
# >> See documentation on CTC loss with pytorch:
# >> https://pytorch.org/docs/stable/generated/torch.nn.CTCLoss.html 


def train_loss(num_of_timesteps, train_size, train_x_new,
               max_str_len,train_y, cm, num_epochs, train_data):
    
    loss = nn.CTCLoss(blank=0)
    optimizer = torch.optim.Adam(cm.parameters(), lr=0.0001)
    train_label_len = np.zeros((train_size, 1), dtype=int) 
    for i in range(train_size):
        train_label_len[i] = len(train_data.loc[i, 'IDENTITY'])
    train_label_len_tensor = torch.tensor(train_label_len, dtype=torch.long)    
    train_input_len = np.ones([train_size, 1]) * (num_of_timesteps-2)
    
    
    epochen_counter = 1

    # train the Model:
    for k in range(num_epochs):
        train_data_permuted = torch.permute(train_x_new, (0,3,2,1))
        predictions = cm(train_data_permuted)  #use the CharModel
        predictions = predictions[:, 2:, :]
        permuted_predictions = torch.permute(predictions, (1,0,2))
        steps, bs, number_of_labels = permuted_predictions.size()

        #log_softmax_values = F.log_softmax(permuted_predictions, 2)

        new_targets = train_y

        new_input_lengths = torch.tensor(train_input_len, dtype=torch.int32)
        new_target_lengths = train_label_len_tensor

        new_calculated_loss = loss(permuted_predictions, new_targets, new_input_lengths, new_target_lengths)
        epochen_counter +=1
        new_calculated_loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        print(new_calculated_loss)
    return predictions    
