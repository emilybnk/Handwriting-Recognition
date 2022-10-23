# -*- coding: utf-8 -*-
"""
Created on Mon Sep 26 18:00:07 2022

@author: emibu
"""
import torch

def num_to_label(num, alphabets):
    # convert numbers to labels for converting back the predictions
    # takes on number as input and converts into the respective letter
    ret = ""
    if num==0:  # CTC Blank (0)
        ret = "°"
    else:
        dict_list = list(alphabets)
        ret = dict_list[num-1]
    return ret

def decode_preds(preds, size, alphabets):
    # based on https://www.youtube.com/watch?v=IcLEJB2pY2Y&t=3366s
    preds = torch.softmax(preds, 2)    # convert the tensor to a tensor of probabilities
    preds = torch.argmax(preds, 2)     # pick the maximum values of the tensor
    preds = preds.detach().cpu().numpy()   # convert to numpy array
    cap_preds = []
    for j in range(size):
        temp = []
        # convert each array back to a label:
        for k in preds[j]:
            #k = k-1             # -1: pseudo blank for ctc
            temp.append(num_to_label(k, alphabets))
        tp = "".join(temp)
        cap_preds.append(tp)   # append all string-labels to a list
    return cap_preds

# "Cleaning" of "raw prediction"
# remove the unnecessary letters and °
def ctc_decode(encoded):
    liste = []
    for i in encoded:
        temp2 = ""
        temp = i.split("°")
        for k in temp:
            # >> remove duplicate letters
            result = "".join(dict.fromkeys(k))
            temp2 += result
        liste.append(temp2)
    return liste