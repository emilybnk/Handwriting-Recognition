# -*- coding: utf-8 -*-
"""
Created on Mon Sep 26 18:04:07 2022

@author: emibu
"""
def accuracy_name(decoded_data, identity_train):    
    # if all names are predicted correctly accuracy=1
    # if none are predicted correctly accuracy=0
    # whole name has to be predicted correctly not just 4 out of 5 letters or so
    correct_count = 0
    for k in range(len(decoded_data)):
        if decoded_data[k] == identity_train[k]:
            correct_count += 1
    return correct_count / len(identity_train)

# def accuracy_letters    --> upcoming
    # how much of one name is predicted correctly
