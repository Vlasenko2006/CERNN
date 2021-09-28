# CERNN

This is a core code of the chemistry estimating recurrent neural network

It is designed to emulate the estimates of the Community Multiscale Air Quality model (CMAQ). In simple words, it estimates the concentrations of various pollutants in the atmosphere from meteorological data like wind, temperature, humidity, etc.. Find more here: DOI: 10.1016/j.atmosenv.2021.118236 . The core consists of 3 subroutines 

RNN_main.py - this is the driver. It calls the subroutine that packs the data and calls the network to compute the weights and predict concentrations

test_training_sets_ETHA_winter_fun.py - this subroutine splits the meteorological and concentration samples into test and training sets.

RNN_parts_fun.py - this is the network that computes weights concentrations.



1 !!!!!!!!!!!!!!!!!       IMPORTANT     !!!!!!!!!!!!!!!!!!!

Concentration and meteorological anomalies for the test_training_sets_ETHA_winter_fun.py must be normalized by mapping
them to the interval [-1,1] with zero-mean. Normalization will significantly speed up the training and increase the 
accuracy of the neural network.

The best way to do that is to code and run the two lines below:

Your_data = Your_data - Climatology
Your_data = Your_data/np.max(np.abs(Your_data))


1 !!!!!!!!!!!!!!!!!     IMPORTANT     !!!!!!!!!!!!!!!!!!!
