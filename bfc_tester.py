'''
Created on 7 Aug 2015

@author: edwin
'''

import numpy as np
from bayesian_forecast_combination import BayesianForecasterCombination as BFC
from scipy.stats import multivariate_normal as mvn
from scipy.stats import gamma
from scipy.stats import wishart
import logging 
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.DEBUG)    

def samplegp(K, mu0, s0):
    K = s0 * K
    mu0 = np.zeros(K.shape[0]) + mu0 # make sure mu0 is a 1D array of correct length
    samples = mvn.rvs(mu0, K, 1)
    return samples

def samplewishart(inputs, shape0, scale0, l0):
    if inputs.ndim==1:
        inputs = inputs[:, np.newaxis]
    K = scale0 * np.exp(-0.5 * (inputs - inputs.T)**2 / l0**2) + 1e-6 * np.eye(len(inputs))
    
    samples = wishart.rvs(df=shape0 + len(inputs), scale=K)
    return samples

if __name__ == '__main__':
    print "Generating some simulated data..."
    
    F = 5 # number of forecasters
    N = 100 # number of data points 
    Ntest = 10
    T = 10 # number of time steps
    P = N / T # number of time periods
    
    times = np.tile(np.arange(T)[:, np.newaxis], (P, 1))
    periods = np.tile(np.arange(P)[:, np.newaxis], (1, T)).reshape(N, 1)
    
    y_truth = np.zeros((N, 1))
    
    for p in range(P):
        pidxs = (periods == p).reshape(-1)
        l0 = 1
        Ktimes = np.exp(-0.5 * (times[pidxs, :] - times[pidxs, :].T)**2 / l0**2) + 1e-6 * np.eye(np.sum(pidxs))    
        
        y_truth[pidxs, 0] = samplegp(Ktimes, mu0=0, s0=4)
    
    plt.figure(1)
    plt.plot(y_truth, color='black')
    
    x = np.zeros((N, F))    
    
    A = {}
    C = {}
    E = {}
    DF = {}
    Prec = {}
    
    for f in range(F):
        # parameters for bias
        # Use a linear kernel to allow bias to grow with time. Initial noise accounted for by e.
        Ktimes = 0.01 * times[0:T].dot(times[0:T].T)
        l0 = 100 # keep it fairly smooth over y
        Ky = np.exp(-0.5 * (times[0:T] - times[0:T].T)**2 / l0**2)
            
        Ka = (1.0 + Ktimes) * Ky + 1e-6 * np.eye(Ktimes.shape[0])
        a = samplegp(Ka, 0, 1) #np.random.rand() * 4 - 2 # max a is 2. Scale
        a = np.tile(a[:, np.newaxis], (P, 1))
        
        Kc = Ktimes * Ky + 1e-6 * np.eye(Ktimes.shape[0])
        c = samplegp(Kc, 0, 1) # np.random.rand() * 10 # max c is 10. Offset
        c = np.tile(c[:, np.newaxis], (P, 1))
        
        plt.figure(2)
        plt.plot(a)
        
        plt.figure(3)
        plt.plot(c)
        #print "Offset c=%.3f for forecaster %f" % (c, f)
        
        A[f] = a
        C[f] = c
        
        # paramters for noise - a student t distribution with mean 0
        lambda_e = gamma.rvs(1, scale=1.0)#np.random.rand() # degrees of freedom
        print lambda_e
        Lambda_e = samplewishart(times, 10000000, 10000, 20)#np.random.rand() * 30 # inverse squared scale (precision)
        
        DF[f] = lambda_e
        Prec[f] = Lambda_e
        
        noise = np.zeros((N, 1))
        for p in range(P):
            pidxs = (periods == p).reshape(-1)
            b = 1#gamma.rvs(lambda_e/2.0, lambda_e/2.0)
            print 'b=%.3f for period %i for forecaster %i' % (b, p, f)
            noise[pidxs, 0] = mvn.rvs(np.zeros(T), np.linalg.inv(b * Lambda_e[pidxs, :][:, pidxs]), 1)
        
        plt.figure(4)
        plt.plot(noise)
            
        E[f] = noise
        x[:, f:f+1] = y_truth * a + c + noise
        
        plt.figure(1)
        plt.plot(x[:, f])
        
    print "Going to run this now..."
    
    #y_truth = np.ones((N, 1))
    
    y = np.copy(y_truth)
    trainidxs = np.arange(Ntest)
    y[trainidxs] = np.nan
    
    perfectx = np.tile(y_truth, (1, 5))
    
    combiner = BFC(perfectx, y, times, periods)
    
    combiner.fit()
    y, var = combiner.predict(times[0:Ntest], periods[0:Ntest])
    y_test = y[0:Ntest, 0]
    
    print 'RMSE %.3f' % np.sqrt(np.mean((y_truth[:Ntest] - y_test)**2))
    plt.figure(5)
    plt.plot(y_truth, color='black')
    plt.plot(y, color='red')