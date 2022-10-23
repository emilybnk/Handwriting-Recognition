# -*- coding: utf-8 -*-
"""
Created on Mon Sep 26 16:13:46 2022

@author: emibu
"""
import numpy as np
import torch


def max_str(train, train_size, test, test_size, valid, valid_size):
    # derive the count of letters of the longest name
    liste1 = [train.loc[i,"IDENTITY"] for i in range(train_size)]
    liste2 = [test.loc[i,"IDENTITY"] for i in range(test_size)]
    liste3 = [valid.loc[i,"IDENTITY"] for i in range(valid_size)]
    all_lists = liste1 + liste2 + liste3
    longest_name = max(all_lists, key=len)
    return len(longest_name)
        

def min_str(train, train_size, test, test_size, valid, valid_size):
    liste1 = [train.loc[i,"IDENTITY"] for i in range(train_size)]
    liste2 = [test.loc[i,"IDENTITY"] for i in range(test_size)]
    liste3 = [valid.loc[i,"IDENTITY"] for i in range(valid_size)]
    all_lists = liste1 + liste2 + liste3
    longest_name = min(all_lists, key=len)
    return len(longest_name)


def label_to_num(label, alphabets):
    # convert names (labels) to numbers as definded in the alphabets dictionary
    # takes a whole name as input
    label = list(label)
    liste = [alphabets[i] for i in label]
    return np.array(liste)
   


def encode_labels(size, data, max_str_len,alphabets):
    # placeholder for real labels
    label_placeholder = np.ones([size, max_str_len]) * 0      
    # placeholder is filled with real data in encoded form
    for i in range(size):
        # -1 remains if label is shorter than max_str_len
        label_placeholder[i, 0:len(data.loc[i, 'IDENTITY'])] = label_to_num(data.loc[i, 'IDENTITY'],alphabets) 
    # convert labels to torch tensor
    encoded_labels = torch.tensor(label_placeholder, dtype=torch.float32)
    return encoded_labels
     