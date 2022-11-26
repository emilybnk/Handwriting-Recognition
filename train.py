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

def create_mini_batches(big_batch, mini_batch_size):
  mini_batches = []
  data = big_batch
  n_minibatches = data.size()[0] // mini_batch_size
  
  index = 0
  for i in range(n_minibatches):
    minibatch = data[index:index+mini_batch_size]
    index = index+mini_batch_size
    mini_batches.append(minibatch)
  return mini_batches


def train_loss(num_of_timesteps, train_size, train_x_new,
               max_str_len,train_y, cm, num_epochs, train_data, device):
    
    loss = nn.CTCLoss(blank=0)
    optimizer = torch.optim.Adam(cm.parameters(), lr=0.0001)  # lr = learning rate
    
    train_label_len = np.zeros((train_size, 1), dtype=int) 
    for i in range(train_size):
        train_label_len[i] = len(train_data.loc[i, 'IDENTITY'])
    train_label_len_tensor = torch.tensor(train_label_len, dtype=torch.long).to(device)
    my_mini_train_label_len_tensor = create_mini_batches(train_label_len_tensor, 25)
      
    train_input_len = np.ones([train_size, 1]) * (num_of_timesteps-2)
    new_input_lengths = torch.tensor(train_input_len, dtype=torch.int32).to(device)
    my_mini_new_input_lengths = create_mini_batches(new_input_lengths, 25)

    my_mini_x = create_mini_batches(train_x_new, 25)

    my_mini_y = create_mini_batches(train_y, 25)
    
    
    epochen_counter = 1
    
    
    # train the Model:
    predictions_list = []
    for k in range(num_epochs):
      index = 0
      for kl in range(len(my_mini_x)):
        train_data_permuted = torch.permute(my_mini_x[index], (0,3,2,1))
        predictions = cm(train_data_permuted)  #use the CharModel
        predictions = predictions[:, 2:, :]
        permuted_predictions = torch.permute(predictions, (1,0,2))
        steps, bs, number_of_labels = permuted_predictions.size()

        #log_softmax_values = F.log_softmax(permuted_predictions, 2)

        new_targets = my_mini_y[index]

        new_input_lengths = my_mini_new_input_lengths[index]
        new_target_lengths = my_mini_train_label_len_tensor[index]

        new_calculated_loss = loss(permuted_predictions, new_targets, new_input_lengths, new_target_lengths)
        new_calculated_loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        print("Loss in epoch " + str(epochen_counter) + " and Mini-Batch " + str(index) + " is: " + str(new_calculated_loss))

        index +=1

      epochen_counter +=1
    #for i in predictions_list:
     #   print(i.shape)
    return None    
