# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 12:19:04 2019

@author: Vlasenko

Function for computing weights for each subdomain


"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers




def RNN(train_x,train_y,test_x,test_y, path,pathw, gas, prefix, num):
    [xi,yi,zi] = test_x.shape
    [xo,yo] = test_y.shape

    model = tf.keras.Sequential([
    layers.SimpleRNN(yo, activation='tanh',kernel_initializer='orthogonal', dropout=0.25),
    layers.Dense(yo, activation='tanh',kernel_initializer='orthogonal'),
    layers.Dense(yo, activation='tanh',kernel_initializer='orthogonal')]) 

  
    model.compile(optimizer=tf.train.AdamOptimizer(0.00025251),
                  loss='mse',
                  metrics=['mae'])


    [foo,L]=test_y.shape

    for i in range(0,150):
        model.fit(train_x,train_y, shuffle=True, epochs=1, batch_size=15
                  , verbose = 2)


    for i in range(0,150):
        model.fit(train_x,train_y, shuffle=True, epochs=1, batch_size=1500000
                  , verbose = 2)
    
    result = model.predict(test_x)
    pathW = pathw+gas+prefix+str(num)
    model.save_weights(pathW)
    np.save(path+gas+prefix,result, allow_pickle=True, fix_imports=True)   



#

