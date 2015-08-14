'''
Created on 31 Jul 2015

@author: edwin
'''

import numpy as np
from scipy.special import psi
from scipy.linalg import block_diag, cholesky, solve_triangular
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
    l_time = 10 # length scale for the forecasters' variation over time. Could be extended to differe for each forecaster.
    l_target = 10 # length scale fore the forecasters' variation over y-space.  
    l_y = None # lengthscale for time
        
    # Hyperparameters (priors) -----------------------------------------------------------------------------------------
    l0_time = 10 # length scale for the forecasters' variation over time. Could be extended to differe for each forecaster.
    l0_target = 10 # length scale fore the forecasters' variation over y-space.  
    
    mu0_a = 1 # Prior mean signal strength for all models.
    s0_a = 1 # output scale
    
    mu0_c = 0 # Prior mean bias
    s0_c = 1 # output scale of bias
    
    mu0_y = 0 # Prior for the targets
    l0_y = 10 # length scale for the targets
    s0_y = 1 # output scale for the targets    
    
    shape0_b = 1 # Priors for the noise
    rate0_b = 1
    
    shape0_Lambda = 1 
    rate0_Lambda = 1
    
    shape0_lambda = 1
    rate0_lambda = 1 
        
    def __init__(self, x, y, times, periods):
        # Observations -------------------------------------------------------------------------------------------------
        # N x 1 target values, including training labels and predictions. Test indexes values should initially be NaNs.    
        y = np.array(y)
        if y.ndim==1:
            y = y[:, np.newaxis]
        self.y = y
        
        # N x F observations of individual forecasts. Missing observations are NaNs.
        x = np.array(x)
        if x.ndim==1:
            x = x[:, np.newaxis]
        self.x = x
        
        self.N = len(y)
        self.F = x.shape[1]
        
        self.testidxs = np.isnan(self.y).flatten()
        self.trainidxs = (np.isnan(self.y) == False).flatten()

        self.y[self.testidxs, :] = self.mu0_y 
                
        # N time values. If none are give, we assume that the N observations are in time order and evenly spaced.        
        times = np.array(times, dtype=float)
        if times.ndim==1:
            times = times[:, np.newaxis]            
        self.times = times
                    
        if not self.l_time:
            self.l_time = self.l0_time
        if not self.l_target:
            self.l_target = self.l0_target
            
        periods = np.array(periods)
        if periods.ndim==1:
            periods = periods[:, np.newaxis]
        self.periods = periods # N index values indicating which period each forecast relates to.

        # Model Parameters (latent variables) ------------------------------------------------------------------------------
        # Posterior expectations at each data point
        self.a = np.ones((self.N, self.F)) # Inverse signal strength for the ground truth. Varies depending on y and time.
        self.cov_a = np.zeros((self.F, self.N, self.N))
        self.cov_a[:, np.arange(self.N), np.arange(self.N)] += 1
        self.s_a = np.zeros(self.F) + self.s0_a
        self.mu0_a = np.zeros(self.N) + self.mu0_a
        
        self.c = np.zeros((self.N, self.F)) # Bias offset. Expected value = posterior mean. Varies depending on y and time.   
        self.cov_c = np.zeros((self.F, self.N, self.N))
        self.cov_c[:, np.arange(self.N), np.arange(self.N)] += 1
        self.s_c = np.zeros(self.F) + self.s0_c
                
        self.e = np.zeros((self.N, self.F)) # Noise value for individual data points. Prior mean zero.
        self.cov_e = np.zeros((self.F, self.N, self.N))
        self.cov_e[:, np.arange(self.N), np.arange(self.N)] += 1
        
        self.b = np.ones((self.P, self.F)) # Noise precision scale, one value for each run/period.        
        
        self.Lambda_e = {} # F x N x N Noise precision varies over y and time. 
        self.lambda_e = np.zeros(self.F) # F degrees of freedom in student's t noise distribution. Constant for each forecaster.
        
        self.cov_y = np.zeros((self.N, self.N))
        self.cov_y[np.arange(self.N), np.arange(self.N)] += 1
        self.s_y = self.s0_y
        self.mu0_y = np.zeros((self.N, 1))
            
    def sqexpkernel(self, d, l):
        K = np.exp(- 0.5 * d*2 / l**2 ) + 1e-6 * np.ones(d.shape)
        return K
            
    def fit(self):
        """
        Run VB to fit the model and predict the latent variables y at the same time.
        """
        tolerance = 1e-3
        change = np.inf
        maxiter = 100
        niter = 0

        d_time = self.times - self.times.T        
        K_time = self.sqexpkernel(d_time, self.l_time)
        
        while change > tolerance and niter < maxiter:
            
            logging.debug("Iteration " + str(niter))
            
            y_old = self.y
            
            d_y = self.y.T - self.y # using first and second order Taylor expansions for the uncertain inputs.
            
            K_y = self.sqexpkernel(d_y, self.l_target)
            self.K = K_time * K_y
            self.L_K = cholesky(self.K, lower=True, check_finite=False, overwrite_a=False)            
            
            self.expec_y() # begin by estimating y from sensible priors
            self.expec_c() # find the added bias
            self.expec_a() # find any scaling bias
            self.expec_e() # find the noise values
            self.expec_Lambda_b() # find the noise parameters common to all runs
            
            change = np.max(np.abs(self.y - y_old))
            niter += 1
    
    def predict(self, testtimes, testperiods):
        """
        Use the posterior GP over y to interpolate and predict the specified times and periods.
        """
        distances = testtimes[:, np.newaxis] - self.times.T
        K_test_train = self.s_y * self.sqexpkernel(distances, self.l_y) # squared exponential kernel   

        distances = testtimes[np.newaxis, :] - testtimes[:, np.newaxis]
        K_test_test = self.s_y * self.sqexpkernel(distances, self.l_y) # squared exponential kernel
                
        innovation = self.x - (self.mu0_y * self.a + self.c) # observation minus prior over forecasters' predictions
        innovation = innovation.flatten()[:, np.newaxis]
        B = solve_triangular(self.Ly, innovation.T, lower=True, overwrite_b=True)
        A = solve_triangular(self.Ly.T, B, overwrite_b=True)        
        y = self.mu0_y + K_test_train.dot(A)
        
        amat = np.diag((self.a + np.sqrt(self.v_a)).flatten())
        V = solve_triangular(self.Ly, amat.T.dot(K_test_train.T), lower=True, overwrite_b=True)        
        v_y = np.diag(K_test_test - V.T.dot(V))
        
        return y, v_y    
 
    def expec_y(self):                    
        if not self.l_y:
            self.l_y = self.l0_y
        
        train_times = self.times
        distances = train_times - train_times.T # Ntest x N        
        
        train_periods = self.periods
        nonmatchingperiods = (train_periods - train_periods.T) != 0
        distances[nonmatchingperiods] = np.inf
        
        Kprior = self.sqexpkernel(distances, self.l_y)
        K = self.s_y * Kprior
        
        # learn from the training labels
        innovation = self.y[self.trainidxs, :] - self.mu0_y[self.trainidxs, :]
        L_y = cholesky(K[self.trainidxs, :][:, self.trainidxs], lower=True, check_finite=False, overwrite_a=False)
        B = solve_triangular(L_y, innovation, lower=True, overwrite_b=True, check_finite=False)
        A = solve_triangular(L_y.T, B, overwrite_b=True, check_finite=False)
        V = solve_triangular(L_y, K[self.trainidxs][:, self.testidxs], lower=True, check_finite=False)
        
        mu_f = K[self.testidxs][:, self.trainidxs].dot(A)                
        cov = np.zeros((self.N, self.N))
        cov_f = K[self.testidxs][:, self.testidxs] - V.T.dot(V)
        # now update the test indexes from the x  observations
        for f in range(self.F):
            mu_fminus1 = mu_f
            K = cov_f
            
            innovation = self.x[self.testidxs, f:f+1] - (mu_fminus1 * self.a[self.testidxs, f:f+1] 
                                                     + self.c[self.testidxs, f:f+1] + self.e[self.testidxs, f:f+1]) # observation minus prior over forecasters' predictions
            a_diag = np.diag(self.a[self.testidxs, f])

            S_y = a_diag.T.dot(K).dot(a_diag)
            cov_a = self.cov_a[f, self.testidxs][:, self.testidxs]
            cov_a = np.diag(mu_fminus1.reshape(-1)).dot(cov_a).dot(np.diag(mu_fminus1.reshape(-1)).T)
            S_y += cov_a + self.cov_e[f, self.testidxs][:, self.testidxs] + self.cov_c[f, self.testidxs][:, self.testidxs]
            
            L_y = cholesky(S_y, lower=True, check_finite=False, overwrite_a=True)
            
            B = solve_triangular(L_y, innovation, lower=True, overwrite_b=True, check_finite=False)
            A = solve_triangular(L_y.T, B, overwrite_b=True, check_finite=False)
            V = solve_triangular(L_y, a_diag.dot(K), lower=True, overwrite_b=True, check_finite=False)
        
            mu_f = mu_fminus1 + K.dot(a_diag).dot(A)
            cov_f = K - V.T.dot(V)
            
        self.y[self.testidxs, :] = mu_f[self.testidxs]
        cov[self.testidxs][:, self.testidxs] = cov_f
        self.cov_y = cov
        
        # update hyper-parameters as necessary
        shape0_s = 1
        rate0_s = self.s0_y * shape0_s
        shape_s = shape0_s + 0.5 * self.N 
        
        L_Ky = cholesky(Kprior, lower=True, check_finite=False, overwrite_a=True)  
        
        B = solve_triangular(L_Ky, self.y.dot(self.y.T).T + self.cov_y, lower=True, overwrite_b=True)
        A = solve_triangular(L_Ky.T, B, overwrite_b=True)
                
        rate_s = rate0_s + 0.5 * np.trace(A)
        self.s_y = rate_s / shape_s # inverse of precision
            
    def expec_a(self):
        for f in range(self.F):
            innovation = self.x[:, f:f+1] - (self.y * self.mu0_a + self.c[:, f:f+1] + self.e[:, f:f+1]) # observation minus prior over forecasters' predictions 
            K = self.s_a[f] * self.K       
            
            y_diag = np.diag(self.y.reshape(-1))
            diag_mu0_a = np.diag(self.mu0_a)
            S_a = y_diag.dot(K).dot(y_diag.T) + diag_mu0_a.dot(self.cov_y).dot(diag_mu0_a.T) + self.cov_c[f] + self.cov_e[f]
            La = cholesky(S_a, lower=True, overwrite_a=True, check_finite=False)
            B = solve_triangular(La, innovation, lower=True, overwrite_b=True, check_finite=False)
            A = solve_triangular(La.T, B, overwrite_b=True, check_finite=False)           
            V = solve_triangular(La, y_diag.dot(K), check_finite=False, lower=True)
            self.a[:, f] = K.dot(self.y).dot(A)
            self.cov_a[f] = K - V.T.dot(V)

            rate0_s = self.s0_a
            shape_s = 1 + 0.5 * self.N 
            af = self.a[:, f][:, np.newaxis]
            
            B = solve_triangular(self.L_K, af.dot(af.T).T + y_diag.dot(self.cov_a[f]).dot(y_diag.T).T, lower=True, overwrite_b=True)
            A = solve_triangular(self.L_K.T, B, overwrite_b=True)            
            
            rate_s = rate0_s + 0.5 * np.trace(A)
            self.s_a[f] = rate_s / shape_s # inverse of the precision

    def expec_c(self):        
        for f in range(self.F):      
            innovation = self.x[:, f:f+1] - (self.y * self.a[:, f:f+1] + self.mu0_c + self.e[:, f:f+1]) # observation minus prior over forecasters' predictions
            
            K = self.s_c[f] * self.K
            y_diag = np.diag(self.y[:, 0])             
            a_diag = np.diag(self.a[:, f])
            S_c = K + y_diag.dot(self.cov_a[f]).dot(y_diag.T) + a_diag.dot(self.cov_y).dot(a_diag.T) + self.cov_e[f]
            Lc = cholesky(S_c, lower=True, overwrite_a=True, check_finite=False)
            B = solve_triangular(Lc, innovation, lower=True, overwrite_b=True, check_finite=False)
            A = solve_triangular(Lc.T, B, overwrite_b=True, check_finite=False)           
            V = solve_triangular(Lc, K, check_finite=False, lower=True)            
            
            self.c[:, f] = self.mu0_c + K.dot(A).reshape(-1)
            self.cov_c[f] = K - V.T.dot(V) # WHY DO SOME DIAGONALS IN THE TRAINING IDXS END UP < 0? RELATED TO LOWER S_C VALUES?
            
            rate0_s = self.s0_c
            shape_s = 1 + 0.5 * self.N 
            cf = self.c[:, f][:, np.newaxis]
            
            B = solve_triangular(self.L_K, cf.T + self.cov_c[f].T, lower=True, overwrite_b=True)
            A = solve_triangular(self.L_K.T, B, overwrite_b=True)       
            
            rate_s = rate0_s + 0.5 * np.trace(A)
            self.s_c[f] = shape_s / rate_s
                
    def expec_e(self):
        """
        Noise of each observation
        """
        innovation = self.x - (self.y * self.a + self.c) # mu0_e == 0
        
        for f in range(self.F):
            inn_f = innovation[:, f][:, np.newaxis]
            
            # UPDATE e -----------------------------
            self.cov_e[f] = np.zeros((self.N, self.N))
            for p in range(self.P):
                pidxs = self.periods==p
                inn_fp = inn_f[pidxs]
                 
                prior_precision = self.Lambda_e[f] * self.b[p, f]
                K = 1.0 / prior_precision
                
                Se = K + self.cov_c[f][pidxs][pidxs] + self.y[pidxs, :] * self.cov_a[f][pidxs][pidxs] * self.y[pidxs, :].T + \
                        self.a[pidxs, f][:, np.newaxis] * self.cov_y[pidxs][pidxs] * self.a[pidxs, f][np.newaxis, :]
                Le = cholesky(Se, lower=True, overwrite_a=True, check_finite=False)
                B = solve_triangular(Le, inn_fp, lower=True, overwrite_b=True, check_finite=False)
                A = solve_triangular(Le.T, B, overwrite_b=True, check_finite=False)           
                V = solve_triangular(Le, K, check_finite=False, lower=True)   
                
                self.e[pidxs, f] = K.dot(A)
                self.cov_e[f][pidxs, :][:, pidxs] = K - V.T.dot(V)
        
    def expec_Lambda_b(self):
        """
        Parameters of the noise in general -- captures the increase in noise over time, and its relationship with y.
        """              
        for f in range(self.F):                         
            # UPDATE Lambda ------------------------       
            self.Lambda_e[f] = {} # we'll need P separate matrices because the target values y and times can differ
            shape_Lambda = self.shape0_Lambda + self.P
            rate_Lambda = self.K
            for p in range(self.P):
                pidxs = self.periods==p
                inn_f = self.e[pidxs, f] # deviations from mean of 0
                inn_fp = inn_f.dot(inn_f) + self.cov_e[f][pidxs][:, pidxs]
                rate_Lambda += inn_fp * self.b[p, f] # should there be separate b values for each data point? i.e. we would use self.b[f][pidx,pidxs]
            self.Lambda_e[f] = shape_Lambda / rate_Lambda # P x P                            
                            
            # UPDATE b --------------------------- Check against bird paper.            
            shape_b = self.lambda_e[f] + 1.0
            expec_log_b = np.zeros(self.P)
            for p in range(self.P):
                pidxs = self.periods==p
                inn_f = self.e[pidxs, f]
                inn_fp = inn_f.dot(inn_f) + self.cov_e[f][pidxs][:, pidxs]                 
                rate_b = self.lambda_e[f] + np.trace(inn_fp * self.Lambda_e[f][p])
                self.b[p, f] = shape_b / rate_b
                expec_log_b[p] = psi(shape_b) - np.log(rate_b)                        
            # UPDATE lambda -----------------------
            shape_lambda = self.shape0_lambda + 0.5 * self.N
            rate_lambda = self.rate0_lambda - 0.5 * np.sum(1 + expec_log_b - self.b[:, f])
            self.lambda_e[f] = shape_lambda / rate_lambda            