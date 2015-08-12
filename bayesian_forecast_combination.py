'''
Created on 31 Jul 2015

@author: edwin
'''

import numpy as np
from scipy.special import psi
from scipy.linalg import block_diag
import logging

class BayesianForecasterCombination():
    """
    Bayesian Crowd Forecasting? Indepenedent...
    """ 
    
    # Data Dimensions --------------------------------------------------------------------------------------------------
    F = 1 # number of forecasters
    P = 1 # number of forecast periods
    T = 1 # length of each forecast period
    N = 1 # P x T
    
    # Posterior variances over the model params at each data point
    shape_b = [] # N x F posterior parameters for the noise scale gamma distribution. I don't think this needs to vary with
    # time or targets since v_e already varies. Since v_e and b are multiplied, scaling v_e has the same effect as scaling b.
    rate_b = [] # N x F
    
    shape_Lambda = [] # N x F
    rate_Lambda = [] # N x F
    
    shape_lambda = [] # N x F expected value of lambda, the degree of freedom of the student t noise distribution. Used to determine shape_b
    rate_lambda = [] # N x F 
    
    # Posterior means and covariance hyperparams for each forecaster's general behaviour    
    s_a = [] # F posterior output scales 
    s_c = [] # F bias c absorbs any consistent bias at a given time/target point, so that there is still zero-mean noise e

    l_time = 10 # length scale for the forecasters' variation over time. Could be extended to differe for each forecaster.
    l_target = 10 # length scale fore the forecasters' variation over y-space.  
    
    s_y = None # 
    l_y = None # lengthscale for time
        
    # Hyperparameters (priors) -----------------------------------------------------------------------------------------
    l0_time = 10 # length scale for the forecasters' variation over time. Could be extended to differe for each forecaster.
    l0_target = 10 # length scale fore the forecasters' variation over y-space.  
    
    mu0_a = 1 # Prior mean signal strength for all models. If binary_a, this is in latent function space.
    # Prior covariance for the signal strength as a function of time is given by an exponential kernel with hyperparams:
    s0_a = 1 # output scale
    
    shape0_b = 1 # Priors for the noise
    rate0_b = 1 # 1
    
    shape0_Lambda = 1 # 
    rate0_Lambda = 1 #
    
    shape0_lambda = 1 # 1
    rate0_lambda = 1 # 1 
    
    mu0_c = 0 # Prior mean bias
    s0_c = 1 # output scale of bias
    
    mu0_y = 0 # Prior for the targets
    l0_y = 10 # length scale for the targets
    s0_y = 1 # output scale for the targets
    
    def __init__(self, x, y, times, periods):
        
        # Observations -------------------------------------------------------------------------------------------------

        # N x 1 target values, including training labels and predictions. Test indexes values should initially be NaNs.    
        y = np.array(y)
        if y.ndim==1:
            y = y[:, np.newaxis]
        self.y = y
        
        self.testidxs = np.isnan(self.y)
        self.trainidxs = not np.isnan(self.y)
        self.flat_trainidxs = np.tile(self.trainidxs, (1, self.F)).flatten()
        
        # N x F observations of individual forecasts. Missing observations are NaNs.
        x = np.array(x)
        if x.ndim==1:
            x = x[:, np.newaxis]
        self.x = x
        
        self.N = len(y)
        self.F = x.shape[1]
        
        # N time values. If none are give, we assume that the N observations are in time order and evenly spaced.        
        times = np.array(times)
        if times.ndim==1:
            times = times[:, np.newaxis]
        if times.shape[1] == 1:
            times = np.tile(times, (1, self.F))
            
        self.times = times
                    
        if not self.l_time:
            self.l_time = self.l0_time
        if not self.l_target:
            self.l_target = self.l0_target
            
        self.periods = periods # N index values indicating which period each forecast relates to.

        # Model Parameters (latent variables) ------------------------------------------------------------------------------
        # Posterior expectations at each data point
        self.a = np.zeros((self.N, self.F)) # N x F inverse signal strength for the ground truth. Expected value = posterior mean per data point
        self.b = np.zeros((self.P, self.F)) # P x F noise scale
        self.c = np.zeros((self.N, self.F)) # N x F bias offset. Expected value = posterior mean    
        self.e = np.zeros((self.N, self.F)) # N x F noise. Expected value = posterior mean for each data point. Prior mean zero.
        self.Lambda_e = {} 
        self.lambda_e = np.zeros(self.F) # F degrees of freedom in student's t distribution over noise
        
        self.cov_a = {}
        self.s_a = np.zeros(self.F) + self.s0_a
        self.cov_c = {}
        self.s_c = np.zeros(self.F) + self.s0_c
        self.cov_e = {}
        self.cov_y = np.zeros((self.N, self.N))
        self.s_y = self.s0_y
            
    def fit(self):
        """
        Run VB to fit the model and predict the latent variables y at the same time.
        
        Complete this next.
        """
        tolerance = 1e-3
        change = np.inf
        maxiter = 100
        niter = 0

        d_time = self.times - self.times.T        
        
        while change > tolerance and niter < maxiter:
            
            logging.debug("Iteration " + str(niter))
            
            y_old = self.y
            
            d_y = self.y.T - self.y # using first and second order Taylor expansions for the uncertain inputs.
            
            K_time = np.exp(- 0.5 * d_time**2 / self.l_time**2 ) # squared exponential kernel
            K_y = np.exp(- 0.5 * d_y**2 / self.l_target**2 ) # squared exponential kernel
            self.K = self.s_a * K_time * K_y
            
            self.expec_y() # begin by estimating y from sensible priors
            self.expec_c() # find the added bias
            self.expec_a() # find any scaling bias
            self.expec_b_and_e() # find the noise variance
            
            change = np.max(np.abs(self.y - y_old))
            niter += 1
    
    def predict(self, testtimes, testperiods):
        """
        Use the posterior GP over y to interpolate and predict the specified times and periods.
        """
        distances = testtimes[:, np.newaxis] - self.times.T
        K_test_train = self.s_y * np.exp(- 0.5 * distances**2 / self.l_y**2 ) # squared exponential kernel   

        distances = testtimes[np.newaxis, :] - testtimes[:, np.newaxis]
        K_test_test = self.s_y * np.exp(- 0.5 * distances**2 / self.l_y**2 ) # squared exponential kernel
                
        innovation = self.x - (self.mu0_y * self.a + self.c) # observation minus prior over forecasters' predictions
        innovation = innovation.flatten()[:, np.newaxis]
        y = self.mu0_y + K_test_train.dot(self.invS_y).dot(innovation)
        
        amat = np.diag((self.a + np.sqrt(self.v_a)).flatten())
        v_y = np.diag(K_test_test - K_test_train.dot(self.invS_y).dot(amat).dot(K_test_train.T))
        
        return y, v_y    
 
    def expec_y(self):
        innovation = self.x - (self.mu0_y * self.a + self.c + self.e) # observation minus prior over forecasters' predictions
        innovation[self.trainidxs, :] = self.y[self.trainidxs, :]
        innovation = innovation.flatten()[:, np.newaxis]
                    
        if not self.l_y:
            self.l_y = self.l0_y
        
        test_times = self.times[self.testidxs, 0][:, np.newaxis]
        train_times = self.times.flatten()[:, np.newaxis]
        distances = train_times - train_times.T # Ntest x N        
        distances_star = test_times - train_times.T # Ntest x N
        distances_starstar = test_times - test_times.T # Ntest x N
        K = self.s_y * np.exp(- 0.5 * distances**2 / self.l_y**2 ) # squared exponential kernel. Ntest x N
        K_star = self.s_y * np.exp(- 0.5 * distances_star**2 / self.l_y**2 ) # squared exponential kernel. Ntest x N
        K_starstar = self.s_y * np.exp(- 0.5 * distances_starstar**2 / self.l_y**2 ) # squared exponential kernel. Ntest x N

        a_diag = np.diag(self.a.flatten())

        # need to remove the test indexes as they have no observation noise
        S_y = a_diag.dot(K).dot(a_diag.T)
        S_y += self.flat_trainidxs * (self.mu0_y**2 * block_diag(self.cov_a) + block_diag(self.cov_e) + block_diag(self.cov_c))
        self.invS_y = np.linalg.inv(S_y)
        self.y[self.testidxs, :] = self.mu0_y + K_star.dot(a_diag).dot(self.invS_y).dot(innovation.T)
        self.cov_y = K_starstar - K_star.dot(a_diag).dot(self.invS_y).dot(a_diag.T).dot(K_star)
        
        # update hyperparameters as necessary
        shape0_s = 1
        rate0_s = self.s0_y * shape0_s
        shape_s = shape0_s + 0.5 * self.N 
        rate_s = rate0_s + 0.5 * np.trace(np.linalg.inv(K/self.s_y).dot(self.y.dot(self.y.T) + a_diag.dot(self.cov_y).dot(a_diag)))
        self.s_y = rate_s / shape_s # inverse of precision
            
    def expec_a(self):
        for f in range(self.F):
            innovation = self.x[:, f] - (self.y * self.mu0_a + self.c[:, f] + self.e[:, f]) # observation minus prior over forecasters' predictions 
            K = self.s_a[f] * self.K       
            
            y_diag = np.diag(self.y)
            
            invS = np.linalg.inv(y_diag.dot(K).dot(y_diag.T) + self.cov_y * self.mu0_a**2 + self.cov_c[f] + self.cov_e[f])
            kalmangain = K.dot(self.y).dot(invS)
            self.a[:, f] = kalmangain.dot(innovation)
            self.cov_a[f] = K - kalmangain.dot(self.y).dot(K)

            rate0_s = self.s0_a
            shape_s = 1 + 0.5 * self.N 
            af = self.a[:, f][:, np.newaxis]
            rate_s = rate0_s + 0.5 * np.trace(np.linalg.inv(self.K).dot(af.dot(af.T) + y_diag.dot(self.cov_a[f]).dot(y_diag.T)))
            self.s_a[f] = rate_s / shape_s # inverse of the precision

    def expec_c(self):        
        for f in range(self.F):      
            innovation = self.x[:, f] - (self.y * self.a[:, f] + self.mu0_c + self.e[:, f]) # observation minus prior over forecasters' predictions
            
            K = self.s_c[f] * self.K             
            kalmangain = K.dot(np.linalg.inv(K + self.y * self.cov_a[f] * self.y.T +
                         self.a[:, f][:, np.newaxis] * self.cov_y * self.a[:, f][np.newaxis, :]
                         + self.cov_e[f])) 
            self.c[:, f] = kalmangain.dot(innovation)
            self.cov_c[f] = K - kalmangain.dot(K)
            
            rate0_s = self.s0_c
            shape_s = 1 + 0.5 * self.N 
            cf = self.c[:, f][:, np.newaxis]
            rate_s = rate0_s + 0.5 * np.trace(np.linalg.inv(self.K).dot(cf + self.cov_c[f]))
            self.s_c[f] = shape_s / rate_s
                
    def expec_e(self):
        """
        Noise of each observation
        """
        innovation = self.x - (self.y * self.a + self.c) # mu0_e == 0
        
        for f in range(self.F):
            inn_f = innovation[:, f][:, np.newaxis]
            
            # UPDATE e -----------------------------
            for p in range(self.P):
                pidxs = self.periods==p
                inn_fp = inn_f[pidxs]
                
                prior_precision = self.Lambda_e[f] * self.b[p, f]
                K = 1.0 / prior_precision
                kalmangain = K.dot(np.linalg.inv(K + self.cov_c[f][pidxs][pidxs] + 
                         self.y[pidxs, :] * self.cov_a[f][pidxs][pidxs] * self.y[pidxs, :].T +
                         self.a[pidxs, f][:, np.newaxis] * self.cov_y[pidxs][pidxs] * self.a[pidxs, f][np.newaxis, :]))
                self.e[pidxs, f] = kalmangain.dot(inn_fp)
                self.cov_e[f] = K - kalmangain.dot(K)
        
    def expec_Lambda_b(self):
        """
        Parameters of the noise in general -- captures the increase in noise over time, and its relationship with y.
        """              
        for f in range(self.F):
            inn_f = self.e[:, f][:, np.newaxis]
                         
            # UPDATE Lambda ------------------------ Could we also compute this using GP equations?            
            self.shape_Lambda[f] = self.shape0_Lambda + self.P
            self.rate_Lambda[:, f] = np.linalg.inv(self.K)
            for p in range(self.P):
                pidxs = self.periods==p
                inn_fp = inn_f[pidxs].dot(inn_f[pidxs]) + self.cov_e[f][pidxs, pidxs] 
                self.rate_Lambda[f] += inn_fp * self.b[p, f] # should there be separate b values for each data point? i.e. we would use self.b[f][pidx,pidxs]
            self.Lambda_e[f] = self.shape_Lambda[f] / self.rate_Lambda[f] # P x P                            
                            
            # UPDATE b --------------------------- Check against bird paper.            
            self.shape_b[f] = self.lambda_e[f] + 1.0
            expec_log_b = np.zeros(self.P)
            for p in range(self.P):
                pidxs = self.periods==p
                inn_fp = inn_f[pidxs].dot(inn_f[pidxs]) + self.cov_e[f][pidxs, pidxs]                 
                self.rate_b[p, f] = self.lambda_e[f] + np.trace(inn_fp * self.Lambda_e[f])
                self.b[p, f] = self.shape_b[:, f] / self.rate_b[f]
                expec_log_b[p] = psi(self.shape_b[f]) - np.log(self.rate_b[p, f])                        
            # UPDATE lambda -----------------------
            self.shape_lambda[f] = self.shape0_lambda + 0.5 * self.N
            self.rate_lambda[f] = self.rate0_lambda - 0.5 * np.sum(1 + expec_log_b - self.b[:, f])
            self.lambda_e[f] = self.shape_lambda[f] / self.rate_lambda[f]            