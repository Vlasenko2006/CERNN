# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 11:57:30 2019

@author: Vlasenko


This is the driver program for estimating concentration anomalies in the near-land atmosphere using RNN. 
The inputs are meteorological anomalies.

Driver program calls function to prepare training, and test data and function to estimates RNN's weights and cocentrations.

Note that instead of using one single NN for a whole computational domain, we decompose it into several squared subdomains, 
with coordinates [(lx1,ly1);(lx2,ly2)].  When estimates have finished, the overall concentration in the entire domains is assembled
from these subdomains.   

Note that splitting the area into subdomains saves computational memory and allows parallel computations. 


1 !!!!!!!!!!!!!!!!!       IMPORTANT     !!!!!!!!!!!!!!!!!!!

Concentration and meteorlogical anomalies must be normalized by mapping them to the interval [-1,1]
with zero-mean. Normalization will significantly speed up the training and increase the accuracy of 
the neural network.

The best way to do that is to code and run two lines below

Your_data = Your_data - Climatology
Your_data = Your_data/np.max(np.abs(Your_data))


1 !!!!!!!!!!!!!!!!!     IMPORTANT     !!!!!!!!!!!!!!!!!!!





"""

import numpy as np
from test_training_sets_ETHA_winter_fun import tt2_vars
from RNN_parts_fun import RNN

W = 'w' # 'w' - if we want to write test and training sets, '' - if we load all these stuff 
path = 'Your path to the normalized data'
pathT= 'Path to your training and test data'
LX = 30 # coputational domain's length scale in X direction minus subdomain's length scale  
LY = 30 # the same as for LX, but in Y direction. 
lx = 30 # subdomain's length in x direction
ly = 30 # subdomain's length in y direction
stride =10

for lx1 in range(0,LX,stride): # 
    lx2 = lx1+lx
    for ly1 in range(0,LY,stride):
        ly2 = ly1+ly
        print('lx2,ly2',lx2,ly2)
        gas = 'ETHANE_' # or any other gas
        prefix = str(lx1)+'_'+str(lx2)+'_'+str(ly1)+'_'+str(ly2)
        

        
        if W =='w':
            tt2_vars(lx1,lx2,ly1,ly2,prefix,gas,path,pathT) # prepare test and training sets
       
       
        path = 'Your path'  # Winter training and test set

        train_x = np.load(pathT+'train_x'+gas+prefix+'.npy', allow_pickle=True, fix_imports=True)
        train_y = np.load(pathT+'train_y'+gas+prefix+'.npy', allow_pickle=True, fix_imports=True)  
        test_x = np.load(pathT+'test_x'+gas+prefix+'.npy', allow_pickle=True, fix_imports=True)
        test_y = np.load(pathT+'test_y'+gas+prefix+'.npy', allow_pickle=True, fix_imports=True)  

        l = len(train_x)
        l1 = l-0


        for num in range(1,4): # we prepare 4 RNNs with the same architecture on the same training data. 
                               # Since each RNN overfits differently, the averaging of 4 RNN estimates
                               # minimizes the overfittning error
    

            train_x = np.asarray(train_x)
            train_x = np.squeeze(train_x)   
            train_y = np.asarray(train_y) 
            train_y = np.squeeze(train_y)
            RNN(train_x,train_y,test_x,test_y,path, gas, prefix, num,'w') # function calling RNN
    
    
    