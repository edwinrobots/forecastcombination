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
    b = [] # N x F noise scale 
    c = [] # F bias. Expected value = posterior mean
    
    lambda_e = [] # F degrees of freedom in student's t distribution over noise 
    
    # Posterior variances over the model params at each data point
    v_a = [] # N x F Posterior variance of a at each time step
    Lambda_e = [] # N x F Unscaled noise precision at each time step. Referred to as Lambda in Zhu, Leung & He
    v_c = [] # F bias variance
    v_y = [] # N posterior target variance
    cov_y = []
    
    scale_b = [] # N x F posterior parameters for the noise scale gamma distribution. I don't think this needs to vary with
    # time or targets since v_e already varies. Since v_e and b are multiplied, scaling v_e has the same effect as scaling b.
    rate_b = [] # N x F
    
    scale_Lambda = [] # N x F
    rate_Lambda = [] # N x F
    
    scale_lambda = [] # N x F expected value of lambda, the degree of freedom of the student t noise distribution. Used to determine scale_b
    rate_lambda = [] # N x F 
    
    # Posterior means and covariance hyperparams for each forecaster's general behaviour    
    s_a = [] # F posterior output scales 
    s_c = [] # F bias c absorbs any consistent bias at a given time/target point, so that there is still zero-mean noise e
    s_e = [] # F posterior output scales
    s_b = []
    s_lambda = []

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
    
    scale0_b = 1 # Priors for the noise
    rate0_b = 1 # 1
    
    scale0_Lambda = 1 # 
    rate0_Lambda = 1 #
    
    scale0_lambda = 1 # 1
    rate0_lambda = 1 # 1 
    
    s0_e = 1 # output scale of noise
    s0_b = 1 # 
    s0_lambda = 1 #
    
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

        # Initialise the parameter arrays...
    
    def fit(self):
        """
        Run VB to fit the model and predict the latent variables y at the same time.
        
        Complete this next.
        """
        tolerance = 1e-3
        change = np.inf
        maxiter = 100
        niter = 0
        while change > tolerance and niter < maxiter:
            
            logging.debug("Iteration " + str(niter))
            
            y_old = self.y
            
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
        distances = testtimes[np.newaxis, :] - self.times[:, np.newaxis]
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
        # Uncertainty in self.a?  
        # update hyperparameters as necessary
        if not self.s_y:
            self.s_y = self.s0_y
#         else:
#             self.s_y = # variational update -- see gpgrid and the bird tracking paper 
            
        if not self.l_y:
            self.l_y = self.l0_y
        # If we are running an optimiser, l_y should be set by the optimiser function.
    
        times_flat = self.times[:, np.newaxis]        
        distances = times_flat.T - times_flat # Ntest x N
        
        K = self.s_y * np.exp(- 0.5 * distances**2 / self.l_y**2 ) # squared exponential kernel
                
        innovation = self.x - (self.mu0_y * self.a + self.c) # observation minus prior over forecasters' predictions
        innovation = innovation.flatten()[:, np.newaxis]

        # need to remove the test indexes as they have no observation noise
        amat = np.diag((self.a + np.sqrt(self.v_a)).flatten())
        b = np.zeros((self.N, self.F))
        b[self.test_idxs, :] = self.b[self.test_idxs, :] 
        b = b.flatten()
        Lambda_e = self.Lambda_e.flatten()
        self.invS_y = amat.T.dot(np.linalg.inv(amat.dot(K).dot(amat.T)
                                                 + np.diag(1.0/(b * Lambda_e)) )) # + self.v_c
        self.y[self.test_idxs, :] = self.mu0_y + K.dot(self.invS_y).dot(innovation)[self.test_idxs, :]
        self.cov_y = K - K.dot(self.invS_y).dot(amat).dot(K)
        self.v_y[self.test_idxs, :] = np.diag(self.cov_y)[self.test_idxs] # should we really be using this, not cov_y, as it will overestimate variance?
            
    def expec_a(self):
        # update hyperparameters as necessary
        if not self.s_a:
            self.s_a = self.s0_a
#         else:
#             self.s_a = # variational update -- see gpgrid and the bird tracking paper 
            
        if not self.l_time:
            self.l_time = self.l0_time
        if not self.l_target:
            self.l_target = self.l0_target#
        # If we are running an optimiser, l_y should be set by the optimiser function.
        d_time = self.times[np.newaxis, :] - self.times[:, np.newaxis]
        d_y = self.y.T - self.y # using first and second order Taylor expansions for the uncertain inputs.
        
        K_time = np.exp(- 0.5 * d_time**2 / self.l_time**2 ) # squared exponential kernel
        K_y = np.exp(- 0.5 * d_y**2 / self.l_target**2 ) # squared exponential kernel
        K = self.s_a * K_time * K_y
        
        y = np.diag(self.y + np.sqrt(self.v_y))
        
        for f in range(self.F):
            innovation = self.x[:, f] - (self.y * self.mu0_a + self.c) # observation minus prior over forecasters' predictions 
            invS = np.linalg.inv(y.dot(K).dot(y) + np.diag(1.0/(self.b[:, f]*self.Lambda_e[:, f])))# + self.v_c)
            kalmangain = K.dot(self.y).dot(invS)
            self.a[:, f] = kalmangain.dot(innovation)
            cov_a = K - kalmangain.dot(y).dot(K)
            self.v_a[:, f] = np.diag(cov_a)

    def expec_c(self):        
        # Apply same treatment as done to expec a.
        # update hyperparameters as necessary
        if not self.s_c:
            self.s_c = self.s0_c
#         else:
#             self.s_c = # variational update -- see gpgrid and the bird tracking paper 

        if not self.l_time:
            self.l_time = self.l0_time
        if not self.l_target:
            self.l_target = self.l0_target#
            
        # If we are running an optimiser, l_y should be set by the optimiser function.
        d_time = self.times[np.newaxis, :] - self.times[:, np.newaxis]
        d_y = self.y.T - self.y
        
        K_time = np.exp(- 0.5 * d_time**2 / self.l_time**2 ) # squared exponential kernel
        K_y = np.exp(- 0.5 * d_y**2 / self.l_target**2 ) # squared exponential kernel
        K = self.s_c * K_time * K_y
        
        innovation = self.x - self.y * self.a - self.mu0_c # observation minus prior over forecasters' predictions 
        kalmangain = K.dot(np.linalg.inv(K + np.linalg.inv(self.Lambda_e*self.b)))# + self.v_y * (self.a**2 + self.v_a))
        self.c = kalmangain.dot(innovation)
        cov_c = K - kalmangain.dot(K)
        self.v_c = np.diag(cov_c)                
                
    def expec_b_and_e(self):
        # update hyperparameters as necessary
        if not self.s_e:
            self.s_e = self.s0_e
            self.s_b = self.s0_b
            self.s_lambda = self.s0_lambda
#         else:
#             self.s_c = # variational update -- see gpgrid and the bird tracking paper 
            
        if not self.l_time:
            self.l_time = self.l0_time
        if not self.l_target:
            self.l_target = self.l0_target
        # If we are running an optimiser, l_y should be set by the optimiser function.
        d_time = self.times[np.newaxis, :] - self.times[:, np.newaxis]
        d_y = self.y.T - self.y
        
        K_time = np.exp(- 0.5 * d_time**2 / self.l_time**2 ) # squared exponential kernel
        K_y = np.exp(- 0.5 * d_y**2 / self.l_target**2 ) # squared exponential kernel
        K_b = self.s_b * K_time * K_y
        K_e = self.s_e * K_time * K_y
        K_lambda = self.s_lambda * K_time * K_y
        
        innovation = self.x - self.mu0_y * self.a - self.c # N x F
        
        self.scale_b = (self.lambda_e + 1.0) * 0.5
        traceterm = K_b.dot(innovation**2 + self.v_a + (self.a**2 + self.v_a) * self.v_y[:, np.newaxis]) * self.Lambda_e # N x F # + self.v_c
        self.rate_b = (self.lambda_e + traceterm) * 0.5
        
        self.b = self.scale_b / self.rate_b # expectation of b
        expec_log_b = psi(self.scale_b) - np.log(self.rate_b)
                
        # now update lambda 
        self.scale_lambda = self.scale0_lambda + np.sum(K_lambda, axis=1)[:, np.newaxis] * 0.5 # Nx1
        self.rate_lambda = self.rate0_lambda - 0.5 * K_lambda.dot(1 + expec_log_b - self.b) # NxF
        self.lambda_e = self.scale_lambda / self.rate_lambda # N x F

        # Update the distribution over the Lambda_e
        self.scale_Lambda = self.scale0_Lambda + np.sum(K_e, axis=1)[:, np.newaxis] * 0.5 # Nx1
        traceterm = K_e.dot( innovation**2 + self.v_a + (self.a**2 + self.v_a) * self.v_y[:, np.newaxis]) * self.b # N x F #  + self.v_c
        self.rate_Lambda = self.rate0_Lambda + 0.5 * traceterm   
        self.Lambda_e = self.scale_Lambda / self.rate_Lambda