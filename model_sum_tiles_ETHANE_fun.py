# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 12:15:07 2019

 This script computes ETHAE by means of pre-trained NN for a specific patch.
 
 Top routine model_sum_tiles_main
 
 Date = 20.02.2020

"""
# This network is done for 2 different regions


import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats.stats import pearsonr 
import tensorflow as tf
from tensorflow.keras import layers
import os, sys
#from Data_for_NN_CMAQ_fun4 import training_and_test_sets


def mst(path,pathW1,test_x,test_y,prefix, gas, num,w):



    test_x = np.asarray(test_x) 
    test_x = np.squeeze(test_x)  
    test_y = np.asarray(test_y) 
    test_y = np.squeeze(test_y)

    [xi,yi,zi] = test_x.shape
    [xo,yo] = test_y.shape



    if w=='':
        model = tf.keras.Sequential([
        layers.SimpleRNN(yo, activation='tanh',kernel_initializer='orthogonal', dropout=0.25),
        layers.Dense(yo, activation='tanh',kernel_initializer='orthogonal'),
        layers.Dense(yo, activation='tanh',kernel_initializer='orthogonal')]) #'orthogonal
#


    c = 0
    for i in range(1,num): #20
        c = c+1
        pathW = pathW1+'Weighs_'+gas+prefix+str(i)+'_hd5'
        model.load_weights(pathW, by_name=False)

        if i == 1:
            result = model.predict(test_x)
        else:
            result = result + model.predict(test_x)
        print('loop = ',i)
        
    print('prefix ',prefix)
    result = result/c    


    np.save(path+gas+prefix,result[:,0:400], allow_pickle=True, fix_imports=True)

    
