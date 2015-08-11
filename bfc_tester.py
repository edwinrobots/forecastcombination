'''
Created on 7 Aug 2015

@author: edwin
'''

import numpy as np
from bayesian_forecast_combination import BayesianForecasterCombination as BFC
from scipy.stats import multivariate_normal as mvn
import wishart

def samplegp(inputs, mu0, s0, l0):
    if inputs.ndim==1:
        inputs = inputs[:, np.newaxis]
    K = np.exp(-0.5 * (inputs - inputs.T)**2 / l0**2)
    
    samples = mvn.rvs(mu0, K, inputs.shape)
    return samples

def samplewishart(inputs, scale0, rate0, l0):
    if inputs.ndim==1:
        inputs = inputs[:, np.newaxis]
    K = np.exp(-0.5 * (inputs - inputs.T)**2 / l0**2)
    
    samples = wishart(len(inputs), K)
    return samples

if __name__ == '__main__':
    print "Generating some simulated data..."
    
    F = 5 # number of forecasters
    N = 100 # number of data points 
    Ntest = 10
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
        lambda_e = samplewishart(times, 1, 10, 10)#np.random.rand() # degrees of freedom
        Lambda_e = samplewishart(times, 1, 10, 10)#np.random.rand() * 30 # inverse squared scale (precision)
        
        noise_samples = np.random.standard_t(lambda_e, N) * 1.0/Lambda_e**0.5
        
        x[:, f] = y_truth * a + c + noise_samples
        
    print "Going to run this now..."
    
    y = np.copy(y_truth)
    trainidxs = np.arange(Ntest)
    y[trainidxs] = np.nan
    combiner = BFC(x, y, times)
    
    combiner.fit()
    y_test = y[Ntest:, 0]
    
    print 'RMSE %.3f' % np.sqrt(np.mean((y_truth[Ntest:]-y_test)**2))