# -*- coding: utf-8 -*-
"""
Created on Mon Sep 19 18:23:07 2022

@author: emibu
"""

import os
from pathlib import Path
import torch
import argparse

from read_data import read_labels
from read_data import encode
from encode import max_str
from encode import min_str
from encode import encode_labels
from Model import CharModel
from train import create_mini_batches
from train import train_loss
from decode import decode_preds
from decode import ctc_decode
from evaluation import accuracy_name

#%% not working yet
'''
parser = argparse.ArgumentParser(description='Read the path where the data is stored from the argument line')
parser.add_argument('--command_line_path', 
                    type=Path,
                    default=Path().home()/"OneDrive"/"Studium"/"Master"/"Semester 0"/"Deep Learning in NLP"/"Data", 
                    help='Stores path of data as pathlib.Path in "command_line_path" variable. If none is given, default is used.')
args = parser.parse_args()
'''

#%% Variables

train_size = 30000
valid_size = 3000
test_size = 300
num_epochs = 100

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
# character to number
alphabets = {"A":1,"B":2,"C":3,"D":4,"E":5,"F":6,"G":7,"H":8,"I":9,"J":10,"K":11,
           "L":12,"M":13,"N":14,"O":15,"P":16,"Q":17,"R":18,"S":19,"T":20,"U":21,
           "V":22,"W":23,"X":24,"Y":25,"Z":26,"-":27,"'":28," ":29}     


num_of_characters = len(alphabets) + 1  # +1 for ctc pseudo blank
num_of_timesteps = 64                   # length of predicted labels (for images to be divided into 64 time steps) arbitrary     

#%% Preprocessing

path = (Path().home()/"OneDrive"/"Studium"/"Master"/"Semester 0"/"Deep Learning in NLP"/"Data")
#path = args.command_line_path
os.chdir(path)
train_data = read_labels("written_name_train_v2.csv")
valid_data = read_labels("written_name_validation_v2.csv")
test_data = read_labels("written_name_test_v2.csv")

# use encode function from read_data2 file
train_x_new = encode("train", train_size, train_data, device)#, args)
valid_x_new = encode("validation", valid_size, valid_data, device)#, args)


#%% Variables #2

max_str_len = max_str(train_data, train_size,test_data, test_size,valid_data, valid_size)
# Target sequence length of longest target in batch (padding length)
min_str_len = min_str(train_data, train_size,test_data, test_size,valid_data, valid_size)
# Minimum target length

#%% Encode

train_y = encode_labels(train_size, train_data, max_str_len, alphabets, device)
train_y = torch.tensor(train_y, dtype=torch.float32).to(device)
#print(train_y.size())    

#%% Model
cm = CharModel(29).to(device) #29 characters in alphabets

#%% Training
train_loss(num_of_timesteps, train_size, train_x_new,
               max_str_len,train_y, cm, num_epochs, train_data, device)

#%% Decode

# Folgende 3 Zeilen problematisch, weil predictions für kompletten Datensatz gemacht werden
#train_data_permuted1 = torch.permute(train_x_new, (0,3,2,1))
#predictions = cm(train_data_permuted1)  #use the CharModel
#predictions1 = predictions[:, 2:, :]
# Ende problematische Zeilen

# NEU: Predictions für Mini-Batches nacheinander erstellen:
# Ganz oben in dieser Datei einfügen!!!: " from train import create_mini_batches  "

mini_x_for_pred = create_mini_batches(train_x_new, 25)
mini_y_for_pred = create_mini_batches(train_y, 25)

predictions_list = []
index = 0
for kl in range(len(mini_x_for_pred)):
    train_data_permuted = torch.permute(mini_x_for_pred[index], (0,3,2,1))
    predictions = cm(train_data_permuted)  #use the CharModel
    predictions = predictions[:, 2:, :]
    #permuted_predictions = torch.permute(predictions, (1,0,2))
    predictions_list.append(predictions)

    index +=1  

# Der weitere Code in unserem Projekt (z.B. decode_preds usw.) erwartet die Predictions in Form eines Tensors, wir haben aber jetzt 
# eine Liste von Tensoren. Also, am besten die Liste wieder in einen großen Tensor umwandeln. Das müsste durch Stacken gehen.
# Im weiteren Code Name der Variable ändern: > "predictions" >> "tensor_predictions"

tensor_predictions = torch.stack(predictions_list)
tensor_predictions = tensor_predictions.view(tensor_predictions.size(0)*tensor_predictions.size(1),tensor_predictions.size(2),tensor_predictions.size(3))

encoded = decode_preds(tensor_predictions, train_size, alphabets)
# >> result is a list of strings of the form "AAAA°NNN°NNNNNN°AA" (name "Anna", uncleaned)

# >> Derive “Anna” from “"AAAA°NNN°NNNNNN°AA"
decoded = ctc_decode(encoded)
        
#%% check accuracy of train and on validation set
# train:
identity_train = [name for name in train_data["IDENTITY"]]
accuracy_train = accuracy_name(decoded, identity_train)    #data = decoded train
    
print("first 100 decoded names: ", decoded[0:100])
print("accuracy: ", accuracy_train)
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










