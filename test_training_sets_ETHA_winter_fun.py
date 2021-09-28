# -*- coding: utf-8 -*-
"""
Created on Wed Aug 21 16:38:05 2019

@author: Vlasenko

This script splits the meteorlogical and concentration samples into test and training sets. 

"""

import numpy as np

def ld(var):
    act_lev = len(var[1])
    llen=np.zeros([act_lev,1])
    for i in range(0,act_lev):
        llen[i,0]= len(var[1][i])
    return(llen)


    
  
'''

Function tt2_vars.

inputs:

gas =  name of species

prefix =  location of the subdomain (do not touch)

path = path to the concentration and meteorlogical anomalies. They should be in the same folder  

pathT =  path where you whant to store the training and test data

1 !!!!!!!!!!!!!!!!!       IMPORTANT     !!!!!!!!!!!!!!!!!!!

Concentration and meteorological anomalies must be normalized by mapping them to the interval [-1,1]
with zero-mean. Normalization will significantly speed up the training and increase the accuracy of the neural network.

The best way to do that is to code and run the two lines below:

Your_data = Your_data - Climatology
Your_data = Your_data/np.max(np.abs(Your_data))

1 !!!!!!!!!!!!!!!!!     IMPORTANT     !!!!!!!!!!!!!!!!!!!

 
'''    
  
def tt2_vars(lx1,lx2,ly1,ly2,prefix,gas,path, pathT):    
    
    
    Label = np.load(path+gas+'.npy', allow_pickle=True, fix_imports=True) # loading your lables (concentrations) 
    aux = np.squeeze(np.asarray(Label))
    del Label
    Label = 0.9*aux[:,:] 
    print('Label.shape', Label.shape)
    del aux
    Label = list(Label)
    days = len(Label)
    Label= np.squeeze(np.array(Label)) # this is just a safety procedure. Does not affec the result

    Var = ['T2M','FUWIND','FVWIND']   # This is array containing feature names, i.e., meteorological variables.
                                      # You can defien your own. The first letter F means that values
                                      # are defined on edges and not in the centers of the grid cell.
    counter = 0
    lag = 0


    season=days/29

    print('season =',season)
    print('days =',days)

    for var in range(len(Var)):
        counter = counter +1
        varnam = Var[var]
        print('Varnam = ', varnam)
        print('path = ', path+varnam+'.npy')
        Svar = np.load(path+varnam+'.npy', allow_pickle=True, fix_imports=True)
        Svar= np.squeeze(np.asarray(Svar)) #  this is just a safety procedure. Does not affec the result

        print('Svar.shape = ', Svar.shape)
 
        LL = list(Svar)
       
        if varnam[0]=='F':
             LL = list(Svar)

        '''
        To get rid of the boundary condition problem, the inputs have a larger subdomain
        by SlabelxThe inputs subdomain is larger than the output subdomain by Slabelx and
        Slabely offsets in x and y directions. These offsets allow getting rid of the boundary 
        condition problem.  
        '''           
            
        Slabelx = 5 # offset in x-direction
        Slabely = 5 # offset in y-direction
        if counter ==1:

            LLabel = list(Label)
          
            LL1 = []
            LL2 = []
        
        
            for fi in range(0,days):
                tmp2 =np.reshape(LLabel[fi][0:4836],[78,62], order='C')
                t2 =tmp2[lx1+Slabelx:lx2-Slabelx,ly1+Slabely:ly2-Slabely]
                [s4,s5]=t2.shape
                LL1 = LL1+[(0.5-(np.mod(fi,season))/season)] # adding time, not really helpful. Might be removed
                LL2 = LL2+[np.reshape(t2,[s4*s5])]
            del LL
            del LLabel
            LL = LL1
            LLabel = LL2
            

                       
            LT=list()
            for fi in range(0,days):
                LT=LT+[(LL[fi])]

        else:  
            if varnam[0]=='F':
                LL1 = []
                print('CONDITION F', LL[0].shape)          
                for fi in range(0,days):
                    tmp3 = LL[fi]
                    t3 =tmp3[lx1+Slabelx:lx2-Slabelx,ly1+Slabely:ly2-Slabely] # adding offsets
                    [s1,s2]=t3.shape
                    LL1 = LL1+[np.reshape(t3,[s1*s2])]
                del LL
                LL = LL1              
        
            for fi in range(0,days):
                LT[fi]= np.append(LT[fi],LL[fi]) 
        del Svar

        
    ll2 = 3*(days-np.mod(days,4))/4 # defining the size of training set
    print('float ll2 = ', days)
    ll2 = int(ll2)


    s = 3 # temporal offset. We need it to prevent the data appear from different years  

    II=[]

    for ii in range(s,ll2-0):    # pack your data in the training set
        if ii == s:
            train_x = [([ LT[ii], LT[ii-1],LT[ii-2]])] 
            train_y = [(LLabel[ii+lag])]
            II = II+[(ii,ii)]
        else:
            if np.mod(ii,season)>s and np.mod(ii,season)<season -s:
                train_x = train_x + [([ LT[ii], LT[ii-1],LT[ii-2]])]
                train_y = train_y + [(LLabel[ii+lag])]   
                II = II+[(ii,ii)]                   
      
    for ii in range(ll2+s,days-0): # pack your data in the test set
        if ii == ll2+s:
            test_x = [([ LT[ii], LT[ii-1],LT[ii-2]])]
            test_y = [(LLabel[ii+lag])]
            II = II+[(ii,ii)]
        else:
            if np.mod(ii,season)>s and np.mod(ii,season)<season -s:
                test_x = test_x + [([ LT[ii], LT[ii-1],LT[ii-2]])]
                test_y = test_y + [(LLabel[ii+lag])]
                II = II+[(ii,ii)]
 

    pathT = 'Your path to the training and test data'

    np.save(pathT+'test_x'+gas+prefix, test_x, allow_pickle=True, fix_imports=True)
    np.save(pathT+'test_y'+gas+prefix, test_y, allow_pickle=True, fix_imports=True)
    np.save(pathT+'train_x'+gas+prefix, train_x, allow_pickle=True, fix_imports=True)
    np.save(pathT+'train_y'+gas+prefix, train_y, allow_pickle=True, fix_imports=True)
    
