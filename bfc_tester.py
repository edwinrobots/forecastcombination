'''
Created on 7 Aug 2015

@author: edwin
'''

import numpy as np

def samplegp(inputs, mu0, s0, l0):
    pass

def samplegammas(inputs, scale0, rate0):
    pass

if __name__ == '__main__':
    print "Generating some simulated data..."
    
    F = 5 # number of forecasters
    N = 100 # number of data points 
    T = 100 # number of time steps
    P = 1 # number of time periods
    
    times = np.arange(T)
    periods = np.ones(N)
    
    y_truth = samplegp(times, mu0=0, s0=1, l0=10)
    x = np.zeros((N, F))    
    
    for f in range(F):
        # parameters for bias 
        a = samplegp(times, 0, 4, 10) #np.random.rand() * 4 - 2 # max a is 2. Scale
        c = samplegp(times, 0, 4, 10) #np.random.rand() * 10 # max c is 10. Offset
        
        # paramters for noise - a student t distribution with mean 0
        lambda_e = samplegammas(times, 1, 10)#np.random.rand() # degrees of freedom
        Lambda_e = samplegammas(times, 1, 10)#np.random.rand() * 30 # inverse squared scale (precision)
        
        noise_samples = np.random.standard_t(lambda_e, N) * 1.0/Lambda_e**0.5
        
        x[:, f] = y_truth * a + c + noise_samples
        
    print "Going to run this now..."