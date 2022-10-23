# -*- coding: utf-8 -*-
"""
Created on Mon Sep 26 17:39:21 2022

@author: emibu
"""
#%% Model

import torch
from torch import nn
import numpy as np
from torch.nn import functional as F

class CharModel(nn.Module):
    # based on https://www.youtube.com/watch?v=IcLEJB2pY2Y&t=3366s
    def __init__(self, num_chars):
        super().__init__()
        # >> for convolutional layer, see documentation: https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
        # >> arguments in convolutional layer: input channels, output channels, kernel size, padding
        # input: 1 because we have 1 channel since we converted the images to greyscale
        # output:128, is arbitrarily chosen
        self.conv1 = nn.Conv2d(1,32, kernel_size=(3,3), padding=("same"))  
        # >> for max pool layer, see documentation: https://pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html
        self.batch1 = nn.BatchNorm2d(32)
        self.max_pool_1 = nn.MaxPool2d(kernel_size=(2,2))
        
        self.conv2 = nn.Conv2d(32,64, kernel_size=(3,3), padding=("same"))
        self.batch2 = nn.BatchNorm2d(64)
        self.max_pool_2 = nn.MaxPool2d(kernel_size=(2,2))
        self.drop_l = nn.Dropout(0.3)
        
        self.conv3 = nn.Conv2d(64,128, kernel_size=(3,3), padding=("same"))
        self.batch3 = nn.BatchNorm2d(128)
        self.max_pool_3 = nn.MaxPool2d(kernel_size=(2,1))
        self.drop_2 = nn.Dropout(0.3)
        
        self.linear_1 = nn.Linear(1024, 64)
        
        self.lstm = nn.LSTM(64,256,bidirectional=True,batch_first=True)
        self.lstm2 = nn. LSTM(512,256,bidirectional=True,batch_first=True)
        
        self.linear = nn.Linear(512, num_chars+1) # +1 ctc blank
        
        self.soft = nn.LogSoftmax()
                

    def forward(self, images):
        bs, c, h, w = images.size()    #batch size, channels, height and width
        #print(bs,c,h,w)
        # Apply convolutional layer to data, then apply relu activation function
        x = self.conv1(images)
        #print(x.size())
        x = self.batch1(x)
        #print(x.size())
        x = F.relu(x)
        #print(x.size())
        x = self.max_pool_1(x)
        #print(x.size())
        
        x = self.conv2(x)
        #print(x.size())
        x = self.batch2(x)
        #print(x.size())
        x = F.relu(x)
        #print(x.size())
        x = self.max_pool_2(x)
        #print(x.size())
        x = self.drop_l(x)
        #print(x.size())
        
        x = self.conv3(x)
        #print(x.size())
        x = self.batch3(x)
        #print(x.size())
        x = F.relu(x)
        #print(x.size())
        x = self.max_pool_3(x)
        #print(x.size())
        x = self.drop_2(x)
        #print(x.size())
        
        x = x.permute(0,3,2,1)
        #print(x.size())
        x = x.reshape(bs, x.size(1), -1)
        #print(x.size())
        x = self.linear_1(x)
        #print(x.size())
        
        x, (_,_) = self.lstm(x)#x.view(1,-1,64))
        #print("lstm: "+str(x.size()))
        x, (_,_) = self.lstm2(x)
        #print("lstm2: "+str(x.size()))
        x = self.linear(x)
        #print(x.size())
        x = self.soft(x)
        return x