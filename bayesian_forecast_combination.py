'''
Created on 31 Jul 2015

@author: edwin
'''

import numpy as np
from scipy.special import psi
from scipy.linalg import block_diag, cholesky, solve_triangular
import logging
from statsmodels.tsa.vector_ar.var_model import var_acf

class BayesianForecasterCombination():
    """
    Bayesian Crowd Forecasting? Indepenedent...
    
    # TO DO:
    # 1. Prediction function not returning the right results -- needs to include observed values of x when available.
    # 2. Kernel for time dimension needs to be changed to a linear one as in simulated data sampler. 
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
    
    m_mu0_y = 0 # Hyperprior for the targets
    v_mu0_y = 1 
    l0_y = 3 # length scale for the targets
    s0_y = 1 # output scale for the targets    
    
    shape0_Lambda = 1 
    scale0_Lambda = 1
    
    shape0_lambda = 1
    scale0_Lambda = 1 
        
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
        
        self.silveridxs = np.isnan(self.y).flatten()
        self.goldidxs = (np.isnan(self.y) == False).flatten()

        self.y[self.silveridxs, :] = self.m_mu0_y 
                
        # N time values. If none are give, we assume that the N observations are in time order and evenly spaced.        
        times = np.array(times, dtype=float)
        if times.ndim==1:
            times = times[:, np.newaxis]            
        self.times = times
        
        self.T = np.max(self.times) + 1
                    
        if not self.l_time:
            self.l_time = self.l0_time
        if not self.l_target:
            self.l_target = self.l0_target
            
        periods = np.array(periods)
        if periods.ndim == 2:
            periods = periods.reshape(-1)
        self.periods = periods # N index values indicating which period each forecast relates to.
        
        self.P = np.max(self.periods) + 1

        # Model Parameters (latent variables) ------------------------------------------------------------------------------
        
        d_time = self.times - self.times.T    
        self.K_time = self.sqexpkernel(d_time, self.l_time)
        
        d_y = self.y - self.y.T # using first and second order Taylor expansions for the uncertain inputs.
        self.K_target = self.sqexpkernel(d_y, self.l_target)
        self.K = self.K_time * self.K_target + 1e-6 * np.eye(self.N)
        
        # Posterior expectations at each data point
        self.a = np.ones((self.N, self.F)) # Inverse signal strength for the ground truth. Varies depending on y and time.
        self.s_a = np.zeros(self.F) + self.s0_a        
        self.cov_a = np.zeros((self.F, self.N, self.N))
        for f in range(self.F):
            self.cov_a[f, :, :] = self.s_a[f] * self.K
        
        self.c = np.zeros((self.N, self.F)) # Bias offset. Expected value = posterior mean. Varies depending on y and time.   
        self.s_c = np.zeros(self.F) + self.s0_c
        self.cov_c = np.zeros((self.F, self.N, self.N))
        for f in range(self.F):
            self.cov_c[f, :, :] = self.s_c[f] * self.K
        
        self.b = np.ones((self.P, self.F)) # Noise precision scale, one value for each run/period.        
        self.Lambda_e = {} # F x N x N Noise precision varies over y and time. 
        
        self.e = np.zeros((self.N, self.F)) # Noise value for individual data points. Prior mean zero.
        self.cov_e = {}
        for f in range(self.F):
            self.cov_e[f] = np.zeros((self.N, self.N))             
            shape_Lambda = self.shape0_Lambda
            scale_Lambda = self.scale0_Lambda * self.K_time[0:self.T, :][:, 0:self.T]
            self.Lambda_e[f] = scale_Lambda / shape_Lambda
            for p in range(self.P):
                pidxs = self.periods==p
                self.cov_e[f][np.ix_(pidxs, pidxs)] = self.Lambda_e[f] * self.K_target[pidxs][:, pidxs] / self.b[p, f] + np.eye(self.T) * 1e-6
        
        self.lambda_e = np.zeros(self.F) # F degrees of freedom in student's t noise distribution. Constant for each forecaster.
        
        distances = self.times - self.times.T # Ntest x N
        nonmatchingperiods = (self.periods[:, np.newaxis] - self.periods[np.newaxis, :]) != 0
        distances[nonmatchingperiods] = np.inf
        
        self.l_y = self.l0_y
        self.s_y = self.s0_y
        self.K_y = self.sqexpkernel(distances, self.l_y) + 1e-6 * np.eye(self.N)
        self.cov_y = self.s_y * self.K_y
        self.cov_y[self.goldidxs, :] = 0
        self.cov_y[:, self.goldidxs] = 0
            
    def sqexpkernel(self, d, l):
        K = np.exp(- 0.5 * d**2 / l**2 )
        return K
            
    def fit(self):
        """
        Run VB to fit the model and predict the latent variables y at the same time.
        """
        tolerance = 1e-3
        change = np.inf
        maxiter = 100
        niter = 0
        
        while change > tolerance and niter < maxiter:
            
            y_old = np.copy(self.y)
            
            d_y = self.y - self.y.T # using first and second order Taylor expansions for the uncertain inputs.
            self.K_target = self.sqexpkernel(d_y, self.l_target)
            self.K = self.K_time * self.K_target + 1e-6 * np.eye(self.N)
            self.L_K = cholesky(self.K, lower=True, check_finite=False)            
            
            self.expec_y() # begin by estimating y from sensible priors         
            self.expec_c() # find the added bias            
            self.expec_a() # find any scaling bias
            self.expec_e() # find the noise values
            self.expec_Lambda_b() # find the noise parameters common to all runs   
            
            change = np.max(np.abs(self.y - y_old))
            niter += 1
            
            logging.debug("Completed iteration " + str(niter) + ", change = " + str(change))            
    
    def predict(self, testtimes, testperiods):
        """
        Use the posterior GP over y to interpolate and predict the specified times and periods.
        """
        y, cov = self.posterior_y(testtimes, testperiods)    
        v_y = np.diag(cov)
        return y, v_y    
 
    def expec_y(self):          
        mu, cov = self.posterior_y()
        self.y[self.silveridxs, :] = mu
        self.cov_y[ np.ix_(self.silveridxs, self.silveridxs) ] = cov
        
        # update hyper-parameters as necessary
#         shape0_s = 1
#         rate0_s = self.s0_y * shape0_s
#         shape_s = shape0_s + 0.5 * self.N 
#         
#         rate_s = rate0_s
#         for p in range(self.P):     
#             pidxs = self.periods == p   
#             L_Ky = cholesky(self.K_y[pidxs][:, pidxs], lower=True, check_finite=False)
#             devs = self.y[pidxs, :] - self.mu0_y
#             var_y = np.diag(np.diag(self.cov_y[pidxs][:, pidxs]))
#             B = solve_triangular(L_Ky, devs.dot(devs.T) + var_y, lower=True, overwrite_b=True)
#             A = solve_triangular(L_Ky.T, B, overwrite_b=True)
#             rate_s += 0.5 * np.trace(A)
#         self.s_y = rate_s / shape_s # inverse of precision
        
    def posterior_y(self, predict_times=None, predict_periods=None):
        
        K_train = self.s_y * self.K_y      
        K_gold = K_train[self.goldidxs, :][:, self.goldidxs]  
        
        if not np.any(predict_times) or not np.any(predict_periods):
            K_predict = K_train
            silveridxs = self.silveridxs
            testidxs = self.silveridxs
        else:
            predict_times = np.concatenate((self.times[self.silveridxs], predict_times), axis=0)
            distances = predict_times - predict_times.T # Ntest x N
            nonmatchingperiods = (predict_periods - predict_periods.T) != 0
            distances[nonmatchingperiods] = np.inf
            K_predict = self.sqexpkernel(distances, self.l_y) + 1e-6 * np.eye(self.N)
            
            silveridxs = np.arange(1, np.sum(self.silveridxs))
            testidxs = np.arange(np.sum(self.silveridxs), len(predict_times))            
        
        # update the prior mean
        v_obs_y = np.var(self.y)
        self.mu0_y = (self.m_mu0_y * v_obs_y + np.mean(self.y) * self.v_mu0_y) / (self.v_mu0_y + v_obs_y)
        print "mu0_y = %.3f" % self.mu0_y
        
        # learn from the training labels
        innovation = self.y[self.goldidxs, :] - self.mu0_y
        L_y = cholesky(K_gold, lower=True, check_finite=False)
        B = solve_triangular(L_y, innovation, lower=True, overwrite_b=True, check_finite=False)
        A = solve_triangular(L_y.T, B, overwrite_b=True, check_finite=False)
        V = solve_triangular(L_y, K_predict[:, self.goldidxs].T, lower=True, check_finite=False)
        
        mu = self.mu0_y + K_predict[testidxs][:, self.goldidxs].dot(A)                
        cov = K_predict - V.T.dot(V)
        # now update the test indexes from the x  observations
        for f in range(self.F):
            mu_fminus1 = mu
            cov_f = cov[silveridxs][:, silveridxs]# + 1e-6 * np.eye(len(mu)) # jitter
            
            innovation = self.x[self.silveridxs, f:f+1] - (mu_fminus1 * self.a[self.silveridxs, f:f+1] 
                                                     + self.c[self.silveridxs, f:f+1] + self.e[self.silveridxs, f:f+1]) # observation minus prior over forecasters' predictions
            print np.min(innovation)
            a_diag = np.diag(self.a[self.silveridxs, f])

            var_a = np.diag(np.diag(self.cov_a[f, self.silveridxs][:, self.silveridxs]))
            var_a = np.diag(mu_fminus1.reshape(-1)).dot(var_a).dot(np.diag(mu_fminus1.reshape(-1)).T)
            var_e = np.diag(np.diag(self.cov_e[f][self.silveridxs][:, self.silveridxs]))
            var_c = np.diag(np.diag(self.cov_c[f, self.silveridxs][:, self.silveridxs]))
            S_y = cov_f + var_a + var_e + var_c 
            
            L_y = cholesky(S_y, lower=True, check_finite=False)
            
            B = solve_triangular(L_y, innovation, lower=True, overwrite_b=True, check_finite=False)
            A = solve_triangular(L_y.T, B, overwrite_b=True, check_finite=False)
            V = solve_triangular(L_y, a_diag.dot(cov[silveridxs, :]), lower=True, overwrite_b=True, check_finite=False)
        
            mu = mu_fminus1 + cov[silveridxs][:, silveridxs].dot(a_diag).dot(A)
            cov = cov - V.T.dot(V)
         
        return mu, cov[testidxs][:, testidxs]
            
    def expec_a(self):
        for f in range(self.F):
            innovation = self.x[:, f:f+1] - (self.y * self.mu0_a + self.c[:, f:f+1] + self.e[:, f:f+1]) # observation minus prior over forecasters' predictions 
            K = self.s_a[f] * self.K       
            
            y_diag = np.diag(self.y.reshape(-1))
            
            var_y = np.diag(np.diag(self.cov_y))
            var_c = np.diag(np.diag(self.cov_c[f]))
            var_e = np.diag(np.diag(self.cov_e[f]))
            
            S_a = y_diag.dot(K).dot(y_diag.T) + self.mu0_a**2 * var_y + var_c + var_e
            La = cholesky(S_a, lower=True, overwrite_a=True, check_finite=False)
            B = solve_triangular(La, innovation, lower=True, overwrite_b=True, check_finite=False)
            A = solve_triangular(La.T, B, overwrite_b=True, check_finite=False)           
            V = solve_triangular(La, y_diag.dot(K), check_finite=False, lower=True)
            self.a[:, f] = self.mu0_a + K.dot(y_diag).dot(A).reshape(-1)
            self.cov_a[f] = K - V.T.dot(V)

            rate0_s = self.s0_a
            shape_s = 1 + 0.5 * self.N 
            af = self.a[:, f][:, np.newaxis]
            
            var_a = np.diag(np.diag(self.cov_a[f]))
            B = solve_triangular(self.L_K, af.dot(af.T).T + y_diag.dot(var_a).dot(y_diag.T).T, lower=True, overwrite_b=True)
            A = solve_triangular(self.L_K.T, B, overwrite_b=True)            
            
            rate_s = rate0_s + 0.5 * np.trace(A)
#             self.s_a[f] = rate_s / shape_s # inverse of the precision

    def expec_c(self):        
        for f in range(self.F):      
            innovation = self.x[:, f:f+1] - (self.y * self.a[:, f:f+1] + self.mu0_c + self.e[:, f:f+1]) # observation minus prior over forecasters' predictions
            
            K = self.s_c[f] * self.K
            y_diag = np.diag(self.y[:, 0])             
            a_diag = np.diag(self.a[:, f])
            
            var_a = np.diag(np.diag(self.cov_a[f])) 
            var_y = np.diag(np.diag(self.cov_y))
            var_e = np.diag(np.diag(self.cov_e[f]))
            
            S_c = K + y_diag.dot(var_a).dot(y_diag.T) + a_diag.dot(var_y).dot(a_diag.T) + var_e
            Lc = cholesky(S_c, lower=True, overwrite_a=True, check_finite=False)
            B = solve_triangular(Lc, innovation, lower=True, overwrite_b=True, check_finite=False)
            A = solve_triangular(Lc.T, B, overwrite_b=True, check_finite=False)           
            V = solve_triangular(Lc, K, check_finite=False, lower=True)            
            
            self.c[:, f] = self.mu0_c + K.dot(A).reshape(-1)
            self.cov_c[f] = K - V.T.dot(V) # WHY DO SOME DIAGONALS IN THE TRAINING IDXS END UP < 0? RELATED TO LOWER S_C VALUES? -- TRY FIXING COV_Y FIRST. ALSO CHECK Y_DIAG.COV_A.Y_DIAG
            
            rate0_s = self.s0_c
            shape_s = 1 + 0.5 * self.N 
            cf = self.c[:, f][:, np.newaxis] - self.mu0_c
            
            var_c = np.diag(np.diag(self.cov_c[f].T))            
            B = solve_triangular(self.L_K, cf.T + var_c, lower=True, overwrite_b=True)
            A = solve_triangular(self.L_K.T, B, overwrite_b=True)       
            
            rate_s = rate0_s + 0.5 * np.trace(A)
#             self.s_c[f] = shape_s / rate_s

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
                                 
                K = self.K_target[pidxs][:, pidxs] * self.Lambda_e[f] / self.b[p, f] + 1e-6 * np.eye(self.T)
                
                a_diag = np.diag(self.a[pidxs, f])
                y_diag = np.diag(self.y[pidxs, 0])
                
                var_c = np.diag(np.diag(self.cov_c[f][pidxs][:, pidxs]))
                var_a = np.diag(np.diag(self.cov_a[f][pidxs][:, pidxs]))
                var_y = np.diag(np.diag(self.cov_y[pidxs][:, pidxs]))
                
                S_e = K + var_c + y_diag.dot(var_a).dot(y_diag.T) + a_diag.dot(var_y).dot(a_diag.T)
                Le = cholesky(S_e, lower=True, overwrite_a=True, check_finite=False)
                B = solve_triangular(Le, inn_fp, lower=True, overwrite_b=True, check_finite=False)
                A = solve_triangular(Le.T, B, overwrite_b=True, check_finite=False)           
                V = solve_triangular(Le, K, check_finite=False, lower=True)   
                
                self.e[pidxs, f] = K.dot(A).reshape(-1)
                self.cov_e[f][np.ix_(pidxs, pidxs)] = K - V.T.dot(V) 
        
    def expec_Lambda_b(self):
        """
        Parameters of the noise in general -- captures the increase in noise over time, and its relationship with y.
        """              
        for f in range(self.F):                         
            # UPDATE Lambda ------------------------       
            shape_Lambda = self.T + 1 + self.shape0_Lambda + self.P
            scale_Lambda = self.scale0_Lambda * self.K_time[0:self.T, :][:, 0:self.T]
            for p in range(self.P):
                pidxs = self.periods==p
                inn_f = self.e[pidxs, f:f+1] # deviations from mean of 0
                inn_fp = inn_f.dot(inn_f.T) + self.cov_e[f][pidxs][:, pidxs]
                scale_Lambda += inn_fp / self.K_target[pidxs][:, pidxs] * self.b[p, f]
            self.Lambda_e[f] = scale_Lambda / (shape_Lambda - self.T - 1)# P x P                            
                            
            # UPDATE b --------------------------- Check against bird paper.            
            shape_b = self.lambda_e[f] + self.T/2.0
            expec_log_b = np.zeros(self.P)
            for p in range(self.P):
                pidxs = self.periods==p
                inn_f = self.e[pidxs, f]
                var_e = np.diag(np.diag(self.cov_e[f][pidxs][:, pidxs]))
                inn_fp = inn_f.dot(inn_f) + var_e            
                
                L_Lambda = cholesky(self.Lambda_e[f] * self.K_target[pidxs][:, pidxs] + 1e-6  * np.eye(self.T), lower=True, check_finite=False)
                B = solve_triangular(L_Lambda, inn_fp, overwrite_b=True, check_finite=False, lower=True)
                A = solve_triangular(L_Lambda.T, B, overwrite_b=True, check_finite=False)
                rate_b = self.lambda_e[f] + np.trace(A)/2.0
                self.b[p, f] = shape_b / rate_b
                expec_log_b[p] = psi(shape_b) - np.log(rate_b)
                       
            # UPDATE lambda -----------------------
            shape_lambda = self.shape0_lambda + 0.5 * self.N
            scale_Lambda = self.scale0_Lambda - 0.5 * np.sum(1 + expec_log_b - self.b[:, f])
            self.lambda_e[f] = shape_lambda / scale_Lambda            