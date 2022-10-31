# -*- coding: utf-8 -*-
"""
Created on Mon Sep 19 18:23:07 2022

@author: emibu
"""
from read_data import encode
import os
from pathlib import Path
import torch
from read_data import read_labels
from encode import max_str
from encode import min_str
from encode import encode_labels
from Model import CharModel
from train import train_loss
from decode import decode_preds
from decode import ctc_decode
from evaluation import accuracy_name

#%% Variables

train_size=100000
valid_size=10
test_size=10
num_epochs = 200

# character to number
alphabets = {"A":1,"B":2,"C":3,"D":4,"E":5,"F":6,"G":7,"H":8,"I":9,"J":10,"K":11,
           "L":12,"M":13,"N":14,"O":15,"P":16,"Q":17,"R":18,"S":19,"T":20,"U":21,
           "V":22,"W":23,"X":24,"Y":25,"Z":26,"-":27,"'":28," ":29}     


num_of_characters = len(alphabets) + 1  # +1 for ctc pseudo blank
num_of_timesteps = 64                   # length of predicted labels (for images to be divided into 64 time steps) arbitrary     

 
#%% Preprocessing

path = (Path().home()/"OneDrive"/"Studium"/"Master"/"Semester 0"/"Deep Learning in NLP"/"Data")
os.chdir(path)
train_data = read_labels("written_name_train_v2.csv")
valid_data = read_labels("written_name_validation_v2.csv")
test_data = read_labels("written_name_test_v2.csv")

# use encode function from read_data2 file
train_x_new = encode("train", train_size, train_data)
valid_x_new = encode("validation", valid_size, valid_data)


#%% Variables #2

max_str_len = max_str(train_data, train_size,test_data, test_size,valid_data, valid_size)
# Target sequence length of longest target in batch (padding length)
min_str_len = min_str(train_data, train_size,test_data, test_size,valid_data, valid_size)
# Minimum target length

#%% Encode

train_y = encode_labels(train_size, train_data, max_str_len, alphabets)
train_y = torch.tensor(train_y, dtype=torch.float32)
#print(train_y.size())    

#%% Model
cm = CharModel(29) #29 characters in alphabets

#%% Training
x_pred = train_loss(num_of_timesteps, train_size, train_x_new,
               max_str_len,train_y, cm, num_epochs, train_data)

#%% Decode

encoded = decode_preds(x_pred, train_size, alphabets)
# >> result is a list of strings of the form "AAAA°NNN°NNNNNN°AA" (name "Anna", uncleaned)

# >> Derive “Anna” from “"AAAA°NNN°NNNNNN°AA"
decoded = ctc_decode(encoded)
        
#%% check accuracy of train and on validation set
# train:
identity_train = [name for name in train_data["IDENTITY"]]
accuracy_train = accuracy_name(decoded, identity_train)    #data = decoded train
    
'''
# validation:
identity_valid = [name for name in valid_data["IDENTITY"]]
permuted_val = torch.permute(valid_x_new, (0,3,2,1))
pred_val = cm(permuted_val)
val_dec = decode_preds(pred_val, valid_size, alphabets)
val_dec = ctc_decode(val_dec)

accuracy_validation = accuracy_name(val_dec, identity_valid)

# adapt hyperparameter
'''










