# Chemistry estimating recurrent neural network (CERNN)

The health of a population directly depends on the quality of the surrounding air. Numerical atmospheric chemistry modeling requires extensive computations measured in thousands of CPU hours. Neural networks offer an alternative way of estimating air quality and require orders of magnitude lesser computational resources. CERRN is such a model. It emulates the Community Multiscale Air Quality model (CMAQ). Despite its extremely simple architecture, it produces reliable chemical estimates 700 faster than CMAQ. It estimates the concentrations of various pollutants in the atmosphere from meteorological data like wind, temperature, humidity, etc.., See more in [1].  

## Example of CERRN etimates:
Compare real and estimated $$NO_2$$ anomalies estimated by CERNN(left) and CMAQ (right, referred here to as the true)


![Sample Output](https://github.com/Vlasenko2006/CERNN/blob/main/CERNN.jpg)

The **model inputs** are normalized meteorological anomalies, which you compute as:
```
Your_data = Your_data - Climatology
Your_data = Your_data/np.max(np.abs(Your_data))
```
Note that climatology is the daily means of meteorological(chemical) time series, computed for the 30-year period.

The **model output** are the normalized concentration anomalies. Denormalize them subject to your climatological chemical data. 


## Prerequisites:

1. Python 3


## Content:
The core consists of 3 subroutines: 
1. RNN_main.py - this is the driver. It calls the subroutine that packs the data and calls the network to compute the weights and predict concentrations
2. test_training_sets_ETHA_winter_fun.py - this subroutine splits the meteorological and concentration samples into test and training sets.
3. RNN_parts_fun.py - The neural network that computes weights concentrations.



## Setting up CERRN Environment with Anaconda
To run the CERRN, you need several packages installed. These packages are listed in the ```cernn_env.yaml```. Here is the file's content:

```
name: cernn_env
channels:
  - conda-forge
  - defaults
dependencies:
  - python=3.9
  - numpy
  - tensorflow
```
To install these packages with Ananconda, follow the steps below:

### 1️⃣ Create the Environment
Run in terminal
```bash
conda env create -f cernn_env.yaml
```
Then run 
```
conda activate cernn_env
```

### **Scientific References**
1. Vlasenko A., et al. Atmospheric Environment Vol. 254(2021). *Simulation of chemical transport model estimates by means of a neural network using meteorological data*. [https://www.sciencedirect.com/science/article/pii/S1352231021000546](https://www.sciencedirect.com/science/article/pii/S1352231021000546)
