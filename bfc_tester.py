'''
Created on 7 Aug 2015

@author: edwin
'''

import numpy as np
from bayesian_forecast_combination import BayesianForecasterCombination as BFC
from scipy.stats import multivariate_normal as mvn
from scipy.stats import gamma
import wishart

def samplegp(inputs, mu0, s0, l0):
    if inputs.ndim==1:
        inputs = inputs[:, np.newaxis]
    K = np.exp(-0.5 * (inputs - inputs.T)**2 / l0**2) + 1e-6 * np.eye(len(inputs))
    
    mu0 = np.zeros(K.shape[0]) + mu0 # make sure mu0 is a 1D array of correct length
    samples = mvn.rvs(mu0, K, 1)
    return samples

def samplewishart(inputs, scale0, rate0, l0):
    if inputs.ndim==1:
        inputs = inputs[:, np.newaxis]
    K = np.exp(-0.5 * (inputs - inputs.T)**2 / l0**2) + 1e-6 * np.eye(len(inputs))
    
    samples = wishart.wishartrand(len(inputs), K)
    return samples

if __name__ == '__main__':
    print "Generating some simulated data..."
    
    F = 5 # number of forecasters
    N = 100 # number of data points 
    Ntest = 10
    T = 100 # number of time steps
    P = 1 # number of time periods
    
    times = np.tile(np.arange(T)[:, np.newaxis], (P, 1))
    periods = np.tile(np.arange(P)[:, np.newaxis], (1, T)).reshape(N, 1)
    
    y_truth = samplegp(times, mu0=0, s0=1, l0=10)
    x = np.zeros((N, F))    
    
    A = {}
    C = {}
    E = {}
    DF = {}
    Prec = {}
    
    for f in range(F):
        # parameters for bias 
        a = samplegp(times, 0, 4, 10) #np.random.rand() * 4 - 2 # max a is 2. Scale
        c = samplegp(times, 0, 4, 10) #np.random.rand() * 10 # max c is 10. Offset
        
        A[f] = a
        C[f] = c
        
        # paramters for noise - a student t distribution with mean 0
        lambda_e = gamma.rvs(1, scale=1.0/10.0)#np.random.rand() # degrees of freedom
        Lambda_e = samplewishart(times, 1, 10, 10)#np.random.rand() * 30 # inverse squared scale (precision)
        
        DF[f] = lambda_e
        Prec[f] = Lambda_e
        
        noise = np.zeros(N)
        for p in range(P):
            pidxs = (periods == p).reshape(-1)
            b = gamma.rvs(lambda_e/2.0, lambda_e/2.0)
            noise[pidxs] = mvn.rvs(np.zeros(T), np.linalg.inv(b * Lambda_e[pidxs][pidxs]), 1)
        E[f] = noise
        x[:, f] = y_truth * a + c + noise
        
    print "Going to run this now..."
    
    y = np.copy(y_truth)
    trainidxs = np.arange(Ntest)
    y[trainidxs] = np.nan
    combiner = BFC(x, y, times, periods)
    
    combiner.fit()
    y_test = y[Ntest:, 0]
    
    print 'RMSE %.3f' % np.sqrt(np.mean((y_truth[Ntest:]-y_test)**2))