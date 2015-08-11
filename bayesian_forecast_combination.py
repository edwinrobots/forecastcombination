'''
Created on 31 Jul 2015

@author: edwin
'''

import numpy as np
from scipy.special import psi
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
    
    # Observations -----------------------------------------------------------------------------------------------------
    x = [] # N x F observations of individual forecasts. Missing observations are NaNs.
    periods = [] # N index values indicating which period each forecast relates to.
    times = [] # N time values. If none are give, we assume that the N observations are in time order and evenly spaced.
    y = [] # N x 1 target values, including training labels and predictions. Test indexes values should initially be NaNs.    
    test_idxs = [] # test_idxs = np.isnan(self.y)
    
    # Model Parameters (latent variables) ------------------------------------------------------------------------------
    # Posterior expectations at each data point
    a = [] # N x F inverse signal strength for the ground truth. Expected value = posterior mean per data point
    # There are N for each forecaster because we evaluate mean function from GPa at each time & target point. Since 
    # targets are inferred simultaneously, we have *uncertain inputs*. 
    # e = [] # N x F unscaled noise. Expected value = posterior mean for each data point. # e has mean zero, 
    # and precision v_e, which is scaled by b. v_e increases with time and can vary with y.
    b = [] # N x N x F noise scale 
    c = [] # F bias. Expected value = posterior mean
    
    lambda_e = [] # F degrees of freedom in student's t distribution over noise 
    
    # Posterior variances over the model params at each data point
    v_a = [] # N x F Posterior variance of a at each time step
    Lambda_e = [] # N x N x F Unscaled noise precision at each time step. Referred to as Lambda in Zhu, Leung & He
    v_c = [] # F bias variance
    v_y = [] # N posterior target variance
    cov_y = []
    
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
    s_e = [] # F posterior output scales

    l_time = 10 # length scale for the forecasters' variation over time. Could be extended to differe for each forecaster.
    l_target = 10 # length scale fore the forecasters' variation over y-space.  
    
    s_y = None # 
    l_y = None # lengthscale for time
        
    # GP objects used to obtain the model parameter estimates above ----------------------------------------------------
    GPa = [] # F signal strength for the ground truth
    GPc = [] # F bias 
    GPe = [] # F noise 
    GPy = [] # P, one per period 
    
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
    
    s0_e = 1 # output scale of noise
    
    mu0_c = 0 # Prior mean bias
    s0_c = 1 # output scale of bias
    
    mu0_y = 0 # Prior for the targets
    l0_y = 10 # length scale for the targets
    s0_y = 1 # output scale for the targets
    
    def __init__(self, x, y, times):
        y = np.array(y)
        if y.ndim==1:
            y = y[:, np.newaxis]
        self.y = y
        
        x = np.array(x)
        if x.ndim==1:
            x = x[:, np.newaxis]
        self.x = x
        
        self.N = len(y)
        self.F = x.shape[1]
        
        times = np.array(times)
        if times.ndim==1:
            times = times[:, np.newaxis]
        if times.shape[1] == 1:
            times = np.tile(times, (1, self.F))
            
        self.times = times
                    
        if not self.l_time:
            self.l_time = self.l0_time
        if not self.l_target:
            self.l_target = self.l0_target#

        # Initialise the parameter arrays...
        self.a = np.zeros((self.N, self.F))
        self.b = np.zeros((self.N, self.F))
        self.c = np.zeros((self.N, self.F))
        self.Lambda_e = np.zeros((self.N, self.F))
        self.lambda_e = np.zeros(self.F)
        
        self.v_a = np.zeros((self.N, self.F))
        self.s_a = np.zeros(self.F) + self.s0_a
        self.s_e = np.zeros(self.F) + self.s0_e
        self.v_c = np.zeros((self.N, self.F))
        self.s_c = np.zeros(self.F) + self.s0_c
        self.v_y = np.zeros(self.N)
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
        innovation = self.x - (self.mu0_y * self.a + self.c) # observation minus prior over forecasters' predictions
        innovation = innovation.flatten()[:, np.newaxis]
        
        # Uncertainty in self.a?  
        # update hyperparameters as necessary
        shape0_s = 1
        rate0_s = self.s0_y * shape0_s
        shape_s = shape0_s + 0.5 * self.N 
        rate_s = rate0_s + 0.5 * np.sum(innovation**2)
        self.s_y = shape_s / rate_s
            
        if not self.l_y:
            self.l_y = self.l0_y
        
        test_times = self.times[self.test_idxs, :].flatten()[:, np.newaxis]
        train_times = self.times.flatten()[:, np.newaxis]        
        distances = test_times.T - train_times.T # Ntest x N
        
        K = self.s_y * np.exp(- 0.5 * distances**2 / self.l_y**2 ) # squared exponential kernel

        # need to remove the test indexes as they have no observation noise
        amat = np.diag((self.a + np.sqrt(self.v_a)).flatten())
        b = np.zeros((self.N, self.F))
        b[self.test_idxs, :] = self.b[self.test_idxs, :] 
        b = b.flatten()
        Lambda_e = self.Lambda_e.flatten()[:, np.newaxis]
        
        self.invS_y = amat.T.dot(np.linalg.inv(amat.dot(K).dot(amat.T) + b * Lambda_e))
        
        self.y[self.test_idxs, :] = self.mu0_y + K.dot(self.invS_y).dot(innovation)[self.test_idxs, :]
        self.cov_y = K - K.dot(self.invS_y).dot(amat).dot(K)
        self.v_y[self.test_idxs, :] = np.diag(self.cov_y)[self.test_idxs] # should we really be using this, not cov_y, as it will overestimate variance?
            
    def expec_a(self):
        # update hyperparameters as necessary
        y = np.diag(self.y + np.sqrt(self.v_y)) # use this to compute E[y^2]
        
        for f in range(self.F):
            innovation = self.x[:, f] - (self.y * self.mu0_a + self.c) # observation minus prior over forecasters' predictions 

            rate0_s = self.s0_a
            shape_s = 1 + 0.5 * self.N 
            rate_s = rate0_s + 0.5 * np.sum(innovation**2)
            self.s_a[f] = shape_s / rate_s
            K = self.s_a[f] * self.K       
            
            invS = np.linalg.inv(y.dot(K).dot(y) + np.linalg.inv(self.b[:, f] * self.Lambda_e[:, f]))
            kalmangain = K.dot(self.y).dot(invS)
            self.a[:, f] = kalmangain.dot(innovation)
            cov_a = K - kalmangain.dot(y).dot(K)
            self.v_a[:, f] = np.diag(cov_a)

    def expec_c(self):        
        # Apply same treatment as done to expec a.
        # update hyperparameters as necessary

        for f in range(self.F):      
            innovation = self.x[:, f] - (self.y * self.a + self.mu0_c) # observation minus prior over forecasters' predictions
            
            rate0_s = self.s0_c
            shape_s = 1 + 0.5 * self.N 
            rate_s = rate0_s + 0.5 * np.sum(innovation**2)
            self.s_c[f] = shape_s / rate_s
            K = self.s_c[f] * self.K             
            
            kalmangain = K.dot(np.linalg.inv(K + np.linalg.inv(self.Lambda_e[:, f] * self.b[:, f])))
            self.c[:, f] = kalmangain.dot(innovation)
            cov_c = K - kalmangain.dot(K)
            self.v_c[:, f] = np.diag(cov_c)                
                
    def expec_e(self):
        innovation = self.x - (self.mu0_y * self.a + self.c) # N x F
        innovation = innovation + np.sqrt(self.v_a * self.mu0_y**2) 
        
        for f in range(self.F):
            inn_f = innovation[:, f][:, np.newaxis]
            
            rate0_s = self.s0_e
            shape_s = 1 + 0.5 * self.N 
            rate_s = rate0_s + 0.5 * np.sum(inn_f**2)
            self.s_e[f] = shape_s / rate_s
            K = self.s_e[f] * self.K            
                
            aKa = self.a[:, f].dot(K).dot(self.a[:, f]) + self.v_a[:, f].dot(K)
            
            # UPDATE b --------------------------- Check against bird paper.            
            self.shape_b[f] = 0.5 * (self.lambda_e[f] + 1.0)            
            self.rate_b[:, f] = 0.5 * (self.lambda_e[f] + np.diag(aKa + inn_f.dot(inn_f.T)) * self.Lambda_e[:, f])
            self.b[:, f] = self.shape_b[:, f] / self.rate_b[f]
            expec_log_b = psi(self.shape_b[f]) - np.log(self.rate_b[:, f])
                    
            # UPDATE lambda -----------------------
            self.shape_lambda[f] = self.shape0_lambda + 0.5 * self.N
            self.rate_lambda[f] = self.rate0_lambda - 0.5 * np.sum(1 + expec_log_b - self.b[:, f])
            self.lambda_e[f] = self.shape_lambda[f] / self.rate_lambda[f]
    
            # UPDATE Lambda ------------------------ Could we also compute this using GP equations?
            self.shape_Lambda[:, f] = self.shape0_Lambda + 1
            self.rate_Lambda[:, f] = self.rate0_Lambda + np.diag(aKa + inn_f.dot(inn_f.T)) * self.b[:, f]   
            self.Lambda_e[:, f] = self.shape_Lambda[:, f] / self.rate_Lambda[:, f]