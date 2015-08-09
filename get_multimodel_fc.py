import numpy as np
from scipy.optimize import curve_fit  #does least-squares fit of functions to data
import statsmodels.api as sm
import sys
import scipy.stats as stats
import numpy.polynomial.polynomial as poly
import GPy
from gpgrid import GPGrid

from python_gen import FunctionalFunction
from stat_sig_gen import gen_rnd_arr_bootstrap

def add_FunctionalFunctions(f1,f2):
    return FunctionalFunction(f1+f2)

#Class first written by Edwin to define objects that can be used to do Bayesian Model Averaging using the method of Raftery et al. (2005) (except without refining sigma to optimise the CRPS). Create a BMARaftery object with bmaobj=BMARaftery(xtr,ytr), where xtr is the training set of predictors (with first dimension corresponding to the different data points and the second to the different predictors) and ytr is the training set of verification data.
#In a modification to the R05 method, I added an alpha parameter to __init__(), which has the same meaning as the alpha used by Wang et al. 2012. alpha=1 gives the same result as the R05 method, and alpha>1 gives preference to more evenly distributed weights. Note that the scheme is not the same as that used by W12, since their scheme takes probabilistic forecasts as input.
class BMARaftery(object):
    
    #todo: change it so that multiple ensemble members from the same model have the same weights.
    
    def __init__(self, xtr, ytr, alpha=1):
        self.xtr = xtr # noisy predictions - training data
        #self.xtest = xtest # noisy predictions - test points  #PW - removed test point argument for object creation
        self.ytr = ytr # target variable for the training points
        if ytr.ndim == 1:
            self.ytr = self.ytr[:, np.newaxis]
        self.K = xtr.shape[1]  #PW - the no. of models
        self.N = xtr.shape[0]  #PW - the number of training points
        
        #initial guesses of the parameters
        #self.w = np.array(self.K) + 1.0/self.K
        self.w = np.array([1.0/self.K]*self.K)  #PW - equal weights to start
        #self.sigmasq = 1.0/self.rval
        self.sigmasq = np.var(self.ytr - self.xtr)
        #[np.var(self.ytr-x) for x in self.xtr.T]# not sure why this is min and it needs to have dimension K  #PW - initialise guess as the min over models of the mean squared difference between verification data and model forecasts.
        
        self.alpha=alpha #PW
        
        self.z = np.ones((self.N, self.K))
        
        self.a = np.ones((self.K)) #bias terms
        self.b = np.zeros((self.K))
        
        self.tol = 0.0001 
        
    def learnbias(self):
        #self.rval = np.array(self.K) # correlation coefficient #PW - I don't think this is needed
        for k in np.arange(self.K):
            # weight by responsibilities z of model k for each data point, i.e. don't use the points that k was not 
            # responsible for to calculate the bias.
            #self.a[k], self.b[k], _, _, _ = stats.linregress(self.xtr[:, k], self.ytr[:, 0]) #PW
            #self.b[k], self.a[k] = poly.polyfit(self.xtr[:, k], self.ytr[:, 0], 1, w=self.z[:, k])
            self.a[k] = 1.0
            self.b[k] = np.mean(self.ytr[:, 0] - self.xtr[:, k])
        
    def Estep(self):
        self.z = self.w * stats.norm.pdf(self.ytr, loc=self.a * self.xtr + self.b, scale=self.sigmasq)

        #PW - normalising
        for i in np.arange(self.N):
            norm_factor=np.sum(self.z[i,:])
            if abs(norm_factor)>0:
                self.z[i,:] /= norm_factor
            
    def Mstep(self):
        #self.w = np.sum(self.z,axis=0) / self.N  #Pure R05 method
        self.w = (np.sum(self.z,axis=0) +(self.alpha-1))/ (np.sum(self.z) + self.K*(self.alpha-1))  #Including alpha - gives pure R05 method if alpha=1.
        # we normalize by the sum over z because it is possible that numerical errors mean this sum != self.N, e.g. if
        # all the predictors are very far from the ground truth.
        if abs(np.sum(self.w)-1)>1e-3: return  #Check that weights sum near enough to 1.
        fcs = np.array(self.a * self.xtr + self.b)
        self.sigmasq = np.sum(self.z * (self.ytr - fcs)**2) / self.N # This should be for each model #PW - I think all the elements need to be summed over.
        
    def train_EM(self):        
        diff = np.inf
        #oldw = self.w
        #oldsigmasq = self.sigmasq
        niter = 0
        maxiter = 1000
        
        self.learnbias()        
        while diff > self.tol and niter < maxiter:
            oldw = self.w
            oldsigmasq = self.sigmasq

            self.Estep()
            self.Mstep()
            
            diff = np.sum(np.abs(self.w-oldw)) + np.sum(np.abs(self.sigmasq-oldsigmasq)) #PW - not the R05 criterion.
            niter+=1
            
        weightstr = 'Weights: '
        for k in range(self.K): 
            weightstr += '%.2f ' % self.w[k]
        print weightstr

        if diff > self.tol or diff == np.nan:
            print 'BMARaftery warning: convergence not achieved, diff='+str(diff)+', tol='+str(self.tol)
        

    #Predict mean forecast for given vector of component model predictions (x).
    def predict_mean(self, x):
        Ey = np.sum(self.w * (self.a*x[:] + self.b))
        return Ey
    
    #Predict variance of forecast for given vector of component model predictions (x), using eqn 7 of R05
    def predict_var(self, x):
        debiased_x = self.a*x[:]+self.b
        fc_mean=np.sum(self.w * (self.a*x[:] + self.b))
        Vy = np.sum(self.w * (debiased_x - fc_mean)**2) + self.sigmasq
        return Vy
    
    #PW - function to return forecast probability of a given data value, using gauss_mix_pdf(), defined in this file. x should be a vector of self.K predictors.
    def get_pdf(self, x, spreads=None):
        self.train_EM()
        sigmas=np.array([np.sqrt(self.sigmasq)]*self.K)
        means=self.a*x[:]+self.b
        fn=gauss_mix_pdf(means, sigmas, weights=self.w)
        
        #Add some helpful attributes
        if np.any(spreads):
            fn.loc=self.predict_mean(x[:], spreads)
            fn.mean=self.predict_mean(x[:], spreads)
            fn.sigma=np.sqrt(self.predict_var(x[:], spreads))  #the s.d. of the forecast
        else:
            fn.loc=self.predict_mean(x[:])
            fn.mean=self.predict_mean(x[:])
            fn.sigma=np.sqrt(self.predict_var(x[:]))  #the s.d. of the forecast
        fn.a=self.a
        fn.b=self.b
        fn.weights=self.w
        fn.sigmasq=self.sigmasq
        fn.x=x                           #the predictions
        fn.debiased_x=means #the means of the components of the BMA mixture
        
        return fn

class BMA_ss(BMARaftery):
    """
    Modification of BMARaftery class to use a model that assumes the input models have a spread-skill relationship, by 
    modelling the standard deviations of the BMA components as linear functions of the model ensemble standard deviations.
    Need to work out how to do the EM-stepping...
    """
    
    def __init__(self, xtr, spreads, ytr, alpha=1, constant_sigmasq=True):
        super(BMA_ss, self).__init__(xtr, ytr, alpha)

        self.constant_sigmasq = constant_sigmasq # if True, we learn a single model noise parameter for all data points,
        # if False, we use the spread provided by the models to determine the noise, but first adjust the spread to 
        # remove exaggeration/underexaggeration based on performance on training data
        if not self.constant_sigmasq:
            self.sigmasq = np.ones((self.N, self.K)) + spreads 

        self.spreads = spreads #ensemble standard deviations for each forecast
        
        #initial guesses of the parameters - corrections to the bias in the SD estimates provided by models
        self.spread_shape = 0.5 + np.ones(self.K) * self.N #parameters for distribution over 1.0/spread when model is correct, p(spread|m) 
        self.spread_scale = 2.0 + 2.0 / np.sum(self.spreads, axis=0)
            
        self.tol = 0.001 # greater tolerance needed here as convergence seems to be rather slow -- why?
        
    #Unlike in the corresponding function of BMARaftery class, the forecast values are not multiplied by a given factor,
    # as this would interfere with the scaling of the ensemble standard deviations.
    def learnbias(self):
        self.b = np.mean(self.ytr) - np.mean(self.xtr,axis=0)

        fcs = np.array(self.xtr + self.b)
        self.sq_devs = (fcs - self.ytr)**2
        
    def Estep(self):
        # Inverse spread approx. precision. 1-CDF = p(spread was at least this large when k was the generating model)
        self.z = np.log(self.w) + stats.norm.logpdf(self.ytr, loc=self.a * self.xtr + self.b, scale=self.sigmasq) + \
                    np.log(1 - stats.gamma.cdf(1.0/self.spreads, a=self.spread_shape, scale=self.spread_scale))
        self.z -= np.max(self.z, axis=1)[:, np.newaxis]
        self.z = np.exp(self.z) 
        self.z /= np.sum(self.z, axis=1)[:, np.newaxis]        
            
    def Mstep(self):
        self.w = (np.sum(self.z,axis=0) + (self.alpha-1))/ (np.sum(self.z) + self.K*(self.alpha-1))  #Including alpha - gives pure R05 method if alpha=1.
        if abs(np.sum(self.w)-1)>1e-3: return  #Check that weights sum near enough to 1.

        if self.constant_sigmasq:
            self.sigmasq = np.sum(self.z * self.sq_devs) / self.N
        else:
            for k in range(self.K):
                s_a_k = 1 + self.N / 2.0
                s_b_k = 1 + np.sum(self.sq_devs[:, k] / (2.0 * self.spreads[:, k])) 
                ##take the mode
                s_k = s_b_k / (s_a_k - 1)
                self.sigmasq[:, k] = s_k * self.spreads[:, k]
        self.spread_shape = 0.5 + np.sum(self.z, axis=0) / 2.0
        self.spread_scale = 2.0 + 2.0 / np.sum(self.z * self.spreads, axis=0)        
        
    #Predict mean forecast for given vector of component model predictions (x).
    def predict_mean(self, x, spreads):
        
        z = self.w * (1 - stats.gamma.cdf(1.0/spreads, a=self.spread_shape, scale=self.spread_scale))
        z /= np.sum(z)
        
        Ey = np.sum(z * (self.a*x[:] + self.b))
        return Ey
    
    #Predict variance of forecast for given vector of component model predictions (x), using eqn 7 of R05
    def predict_var(self, x, spreads):
        debiased_x = self.a*x[:]+self.b
        fc_mean = self.predict_mean(x, spreads)
        Vy = np.sum(self.w * (debiased_x - fc_mean)**2) + self.sigmasq
        return Vy       
    
# class BayesianForecastCombination():
#     """
#     Initial attempt
#     
#     2. Write down functions for calculating expectations of each parameter conditioned on the others
#     3. Put this into EM    
#     """
#     
#     # model parameters
#     # observations
#     xtr = [] # set of forecasts (training). 3D matrix, with N_periods x K x N_timesteps 
#     ytr = [] # set of ground truth (training)
#     
#     # GPs. CxK dimensional, where C is number of active components?
#     linear_output_scale = [] # linear scaling of ytr to obtain ftr
#     rbf_output_scale = [] # scaling the covariance
#     bias = [] # constant prior means for each component
#     lengthscale = [] # 2xCxK-dimensional (input variables are time and targets y)
#     
#     #Components
#     alpha = 5 # concentration parameter
#     c_ind = [] # component indicators for the data points in xtr. NxKxC
#     C = 2 # number of active components
#     
#     #Ground truth (1-D GP with time as input) using RBF kernel only
#     t_mean = 0
#     t_output_scale = 1
#     t_lengthscale = 1
#     
#     # Hyper-hyperparameters for the GP hyperparameters. The GP parameters will be draws from this base distribution.
#     h_output_scale = [1, 1] # parameters for a beta distribution over the mixing weights in the kernel function
#     h_bias_mean = 0 # parameters for a gaussian over the bias 
#     h_bias_var = 1 
#     h_lengthscale_scale = 1 #check values? The length scale will be optimized outside the VB process
#     h_lengthscale_shape = 1 #check values?   
#     
#     # Correlations between base models can be captured by expanding the correlation matrix to map correlations between
#     # GP components from different forecasts. The size of the matrix is CxC where C is the total number of components
#     # from all models, but elements relating to the same base models can be ignored (blank) as they don't
#     
#     def __init__(self, xtr, ytr, h_output_scale = [1, 1], h_bias_mean = 0, h_bias_var = 1, h_lengthscale_scale = 1, h_lengthscale_shape = 1):
#         self.xtr = np.array(xtr)
#         self.ytr = np.array(ytr)
#         if self.ytr.ndim == 1:
#             self.ytr = self.ytr[:, np.newaxis]
#         self.h_output_scale = h_output_scale
#         self.h_bias_mean = h_bias_mean
#         self.h_bias_var = h_bias_var
#         self.h_lengthscale_scale = h_lengthscale_scale
#         self.h_lengthscale_shape = h_lengthscale_shape
#         
#     def expec_c(self):
#         pass
#     
#     def expec_bias(self):
#         # A frequentist approximation from the training data
#         for c in range(C):
#             self.bias[c] = np.sum(self.c_ind[:,:,c] * (self.xtr - self.ytr), axis=0) / np.sum(self.c_ind[:,:,c], axis=0)
#           
#     
# class BayesianCombination(BMARaftery):
#     
#     def __init__(self, xtr, ytr, alpha=1):
#         '''
#         Constructor
#         '''
#         super(BMA_ss, self).__init__(xtr, ytr, alpha)
#         #initial guesses
#         self.w = np.ones(self.K) * 0.6 # 60% probability of being correct
#         
#         self.mu_wrong = np.mean(self.xtr)
#         self.sigmasq_wrong = np.var(self.xtr)  
#         
#         self.mu_y = np.mean(self.ytr)
#         self.sigmasq_y = np.var(self.ytr)
#         
#     def Estep(self):
#         X = self.a * self.xtr + self.b
#         self.z = self.w * stats.norm.pdf(self.ytr, loc=X, scale=self.sigmasq)
#         self.p_wrong = (1-self.w) * stats.norm.pdf(self.mu_wrong, loc=X, scale=self.sigmasq_wrong)
# 
#         self.z /= (self.z + self.p_wrong)
#             
#     def Mstep(self):
#         self.w = (np.sum(self.z,axis=0) +(self.alpha-1))/ (self.N + 2*(self.alpha-1))
#         if abs(np.sum(self.w)-1)>1e-3: return  #Check that weights sum near enough to 1.
#         fcs = np.array(self.a * self.xtr + self.b)
#         self.sigmasq = np.sum(self.z * (self.ytr - fcs)**2) / self.N
#         
#     #Predict mean forecast for given vector of component model predictions (x).
#     def predict_mean(self, x):
#         
#         numerator = self.mu_y / self.sigmsq_y * np.ones(2**self.K) # one entry for each term
#         denominator = 1.0 / self.sigmasq_y * np.ones(2**self.K) # at the end we sum up over a mixture of 2^K products       
#         
#         x = self.a*x[:] + self.b
#         
#         for k in range(self.K): # this needs to be recursive. Any way to avoid this?
#             numerator[0:2**k/2???] *= self.w[k] * x[k] / self.sigmasq # should this be model-specific?
#             denominator[:???] += 1.0 / self.sigmasq
#             
#             numerator[2**k/2:-1???] *= (1-self.w[k]) # do nothing else because it would cancel to 1 anyway
#             
#         
#         Ey = np.sum(numerator) / np.sum(denominator)
#         return Ey
#     
#     #Predict variance of forecast for given vector of component model predictions (x), using eqn 7 of R05
#     def predict_var(self, x):
#         debiased_x = self.a*x[:]+self.b
#         fc_mean=np.sum(self.w * (self.a*x[:] + self.b))
#         Vy = np.sum(self.w * (debiased_x - fc_mean)**2) + self.sigmasq
#         return Vy
    
class sensor_fusion():
    def __init__(self, xtr, ytr, cond_independence): 
        '''
        Constructor
        '''
        self.cond_independence = cond_independence
        self.xtr = xtr # noisy predictions - training data
        #self.xtest = xtest # noisy predictions - test points  #PW - removed test point argument for object creation
        self.ytr = ytr # target variable for the training points
        if ytr.ndim == 1:
            self.ytr = self.ytr[:, np.newaxis]
        self.K = xtr.shape[1]  #PW - the no. of models
        self.N = xtr.shape[0]  #PW - the number of training points
        
        #initial guesses of the parameters
        self.b = np.mean(self.ytr - self.xtr, axis=0) # bias
        self.C = np.ones((self.K, self.K)) # covariance
        
        self.mu_y = np.mean(self.ytr)
        self.sigmasq_y = np.var(self.ytr)
        
        if self.cond_independence:
            self.V = np.var(self.xtr + self.b[np.newaxis, :] - self.ytr, axis=0)[:, np.newaxis]
        else:
            self.C = np.cov(self.xtr.T + self.b[:, np.newaxis] - self.ytr.T)
            self.invC = np.linalg.inv(self.C)
        
    def predict_mean(self, x):
        x = (x+self.b)[:, np.newaxis]
        if self.cond_independence:
            mean = (self.mu_y/ self.sigmasq_y + np.sum(x/self.V)) / (1.0/self.sigmasq_y + np.sum(1.0/self.V))
        else:
            mean = (self.mu_y / self.sigmasq_y + np.sum(self.invC, axis=0)[np.newaxis,:].dot(x)) / (1.0/self.sigmasq_y + np.sum(self.invC))

        return mean
    
    def predict_var(self, x):
        var = 1.0/(1.0/self.sigmasq_y + np.sum(1.0/self.C))#self.sigmasq_y + np.sum(self.invC)
        return var     
        
    def get_pdf(self, x):
        mean = self.predict_mean(x)
        var = self.predict_var(x)        
        fn = gauss_mix_pdf(mean, np.sqrt(var))
        
        #Add some helpful attributes
        fn.mean = mean
        fn.sigma = np.sqrt(var)  #the s.d. of the forecast
        fn.b=self.b
        
        return fn        

#Function to define a Gaussian function with mean mu and s.d. sigma. Optional cst is factor to multiply the function by. Returns a lambda function so I can add attributes to it. Note, doing this in a separate function rather than in the main body of the code causes the loc and scale parameters of norm to be set correctly - in the main code, they would be left unset until the function is called, when they are set as whatever the mu and sigma variables are at the time it is called, which means they don't get the correct parameter values.
def gauss(mu, sigma, cst=None):
    from scipy.stats import norm
    
    if not cst:
        cst=1.
        
    return lambda x: cst*norm(loc=mu, scale=sigma).pdf(x)

#Function to return pdf corresponding to a mixture of Gaussians, with means and standard deviations given in arrays means and spreads, which should have lengths equal to the number of Gaussian components.
def gauss_mix_pdf(means,sigmas,weights='equal'):
    from numbers import Number
    if weights=='equal':  #equal weight is given to each Gaussian in the mixture
        if isinstance(means,Number):  #for single component
            weights=1.
        else:
            weights = np.array([1./len(means)]*len(means))
        
    fn=lambda x: np.sum(weights*1/np.sqrt(2*np.pi)/sigmas*np.exp(-(x-means)**2/2/sigmas**2))
    
    return fn

#Function to define Gaussian with zero mean, used in optimally fitting a Gaussian function to residuals below
def gauss_zero_mean(x,sigma):
    return gauss(0,sigma)(x)


#Function to return a prediction from a Gaussian process given training predictors X (with first dimension corresponding to the different data points and the second to the different predictors), training verification data veri and new predictors x (with length the same as the second dimension of X). A function is returned corresponding to the Gaussian pdf output by the GP.

def gp(X,veri,x):
    kernel1 = GPy.kern.RBF(input_dim=len(x), variance=1., lengthscale=1.)  #A Gaussian kernel - I used this just because it is the first example given at http://nbviewer.ipython.org/github/SheffieldML/notebook/blob/master/GPy/basic_gp.ipynb . I have no idea if it is the most sensible choice.
    kernel2 = GPy.kern.Linear(input_dim=len(x), variances=np.ones(len(x)), ARD=True)
    # A product of RBF and linear kernels. 
    from GPy.kern._src.prod import Prod
    kernel = Prod((kernel1, kernel2))
    m = GPy.models.GPRegression(X,veri[:,np.newaxis],kernel)
    m.optimize()
    
    mean, sigmasq = m.predict(x[np.newaxis,:])
    sigma = np.sqrt(sigmasq)
    #fn=gauss(mean, sigma)
    fn = gauss_mix_pdf(mean,sigma)  #this returns a function rather than a method, for which integrals are computed faster.
    
    #Set some helpful attributes of the function.
    fn.loc = mean #loc attribute for use with integrate_quad_gen() in python_gen.py when doing integrals from -inf to inf, which scipy.integrate.quad cannot do for general functions.
    fn.mean = mean #for use with get_fc_diag() in fc_verification.py, to get mean of forecasts quickly
    fn.sigma = sigma
    
    return fn
    

#Function to do linear regression of data in veri against predictors X (with first dimension corresponding to the different data points and the second to the different predictors), and return a Gaussian function centred on the regression prediction for the new predictors x (with length the same as the second dimension of X). The width of the Gaussian is fitted from the residuals in the training part. Optional cst is factor to multiply the function by.
def linfit_gauss(X,veri,x,cst=None):
    X = sm.add_constant(X) #to get intercept term in linear regression. Note this does nothing if X already contains a set of constant values, which will produce errors below.

    #Doing the linear fit
    est = sm.OLS(veri, X).fit()
    
    #Fitting Gaussian to residuals
    hist, bin_edges = np.histogram(est.resid, density=True)
    bin_centres = (bin_edges[:-1] + bin_edges[1:])/2
    sigma, var_matrix = curve_fit(gauss_zero_mean, bin_centres, hist)
    
    mean=est.params[0]+np.sum(x*est.params[1:])
    
#    fn=gauss(mean, sigma, cst=cst)
    
    fn=gauss_mix_pdf(mean,sigma)
                
    #Set some helpful attributes of the function.
    fn.loc=mean #loc attribute for use with integrate_quad_gen() in python_gen.py when doing integrals from -inf to inf, which scipy.integrate.quad cannot do for general functions.
    fn.mean=mean #for use with get_fc_diag() in fc_verification.py, to get mean of forecasts quickly  #27/05/15 not necessary any more, since I am now passing a function 
    fn.sigma=sigma
    fn.coeffs=est.params
    fn.resids=est.resid
    
    return fn


#Python function to combine data from different models in some way into a new deterministic or ensemble forecast. Returns an array of the multi-model ensemble forecast, or an array of functions corresponding to the probabilistic forecast derived from the multi-model ensemble.
#Input data_sm is a dictionary with keys corresponding to arrays containing data from the different models. The first 
#axis of each array is assumed to correspond to the forecast lead time, the second to different start dates, and the 
# last to different ensemble members, with possibly other dimensions between. For some methods, these arrays should have
# the same size (apart from the number of ensemble members), and if they don't then the larger arrays will be trimmed.
#data_veri is an array containing verification data, with corresponding elements to those that should be returned in 
# mult_fc.
#mult is a string specifying the method for combining the models:
#'equal_weights' for just combining all ensemble members for all models with equal weight. The mean value for each model is subtrcted first, so the anomalies are returned.
#'linfit' to return a Gaussian pdf, whose mean is a linear fit to the ensemble means of the single models, and whose s.d. is fitted to the residuals of the linear fit.
def get_multimodel_fc(data_sm, data_veri, mult):
    import fnmatch
    
    models=data_sm.keys()    
    
    #For methods that require all models to provide data in order to make a sensible multi-model forecast, first get the minimum size of each dimension (except that for the ensemble members) and limit the multimodel data array to have this size i.e. data_mult only has data where there is forecast data for all the models. This assumes that data in the same position for each model corresponds to forecasts for the same variable, place and time, and that the data beyond the minimum dimension size corresponds to forecasts not made by some models. 
    if np.any([fnmatch.fnmatch(mult,mult_meth) for mult_meth in ['bma*','equal_weights','gp','linfit','linfit-bs', 'sf*']]):

        data_sm_ndim=data_sm[models[0]].ndim
        shape_min=np.zeros((data_sm_ndim-1)) #skipping the last (ensemble member) dimension here
        for i in range(len(shape_min)): 
            shape_min[i]=np.min([data_sm[model].shape[i] for model in models])
        
        for model in models:
            #Discard data beyond dimension size limits here.
            for i in range(len(shape_min)):
                data_sm[model]=np.rollaxis(data_sm[model],i)
                data_sm[model]=data_sm[model][:shape_min[i],...]
                data_sm[model]=np.rollaxis(data_sm[model],0,i+1)
          

        if mult=='equal_weights':
        
            #Subtract the mean forecast value for each forecast lead time (mean over start years and ensemble members)
            for model in models:
                for i in np.arange(shape_min[0]):
                    data_sm[model][i, ...] -= np.mean(data_sm[model][i, ...])
            
            mult_fc = np.rollaxis(np.concatenate([np.rollaxis(data_sm[model],-1) for model in models]), 0, data_sm_ndim)   #use roll axis to get ensemble member dimension to be the first for concatenation, and then roll it back to be the last
        
        #Now for methods that should be trained using cross-1-out validation
        elif np.any([fnmatch.fnmatch(mult,mult_meth) for mult_meth in ['bma*','gp','linfit','linfit-bs', 'sf*']]):
            mult_fc=np.empty(data_sm[models[0]][...,0].shape, dtype=np.object)
            
            #For testing that the linfit forecast beats the equal weights forecast when all the data are used in the regression.
            #fc_all_data=np.zeros(mult_fc.shape)
            
            #Get fit for each date separately, using cross-1-out validation.
            ndates=data_sm[models[0]][...,0].shape[1]
            for date_ind in range(ndates):
                if date_ind % 10 ==0: 
                    print 'date_ind=',date_ind
                
                #Iterate over lead times and dimensions besides the start dates and ensemble members in the data arrays
                data_sm_iter=np.nditer(data_sm[models[0]][:,date_ind,...,0], flags=['multi_index','refs_ok'])
                while not data_sm_iter.finished:
                    
#                    data_sm_iter.multi_index=(1,)  #testing
                    
                    #Get index values corresponding to data at dates excluding the selected dates
                    date_inds=range(date_ind)+range(date_ind+1,ndates)
                    inds=(data_sm_iter.multi_index[0],)+(date_inds,)
                    if len(data_sm_iter.multi_index)>1:
                        inds=inds+data_sm_iter.multi_index[1:]
                    
                    #Index for the specified date
                    ind=(data_sm_iter.multi_index[0],)+(date_ind,)
                    if len(data_sm_iter.multi_index)>1:
                        ind=ind+data_sm_iter.multi_index[1:]
                    
                    #Getting predictors for the specified date
                    x = np.array([np.mean(data_sm[model][ind],axis=-1) for model in models])
                    spreads_x = np.array([np.var(data_sm[model][ind], axis=-1) for model in models])       
                    
                    if fnmatch.fnmatch(mult,'bma*'):  
                        X = np.array([np.mean(data_sm[model][inds],axis=-1) for model in models]).T  #use ensemble means of forecasts from each model as predictors for now - to use each ensemble member, I need to code the BMA so that it can assign equal weights ensemble members from the same model.
                        if mult=='bma': #BMA method of Raftery et al. (2005)
                            bmaobj = BMARaftery(X, data_veri[inds])
                            mult_fc[ind]=bmaobj.get_pdf(x)                              
                        elif fnmatch.fnmatch(mult,'bma_alpha=*'):  #set * to the value to use for alpha in BMARaftery
                            alpha=float(mult[10:])
                            bmaobj = BMARaftery(X, data_veri[inds], alpha=alpha)
                            mult_fc[ind]=bmaobj.get_pdf(x)
                        elif mult=='bma-ss':
                            sigmas = np.array([np.var(data_sm[model][inds],axis=-1) for model in models]).T 
                            bmaobj = BMA_ss(X, sigmas, data_veri[inds]) 
                            mult_fc[ind]=bmaobj.get_pdf(x, spreads_x)
#                        #Bootstrapping the BMA to get distribution of model weights
#                        nsamples=100    
#                        date_inds_bs=gen_rnd_arr_bootstrap(inds[1],nsamples)
#                        mult_fc_bs=np.empty(nsamples, dtype=np.object)    
#                        weights=np.zeros((nsamples,len(models)))
#                        for i in range(nsamples):
#                            inds_bs_tup=(inds[0],list(date_inds_bs[i]))
#                            if len(inds)>2:
#                                inds_bs_tup=inds_bs_tup+inds[2:]
#                            X = np.array([np.mean(data_sm[model][inds_bs_tup],axis=-1) for model in models]).T
#                            bmaobj = BMARaftery(X, data_veri[inds_bs_tup])
#                            mult_fc_bs[i]=bmaobj.get_pdf(x)
#                            weights[i,:]=mult_fc_bs[i].weights
#                        stop
                        
#                        #Getting the distribution of BMA weights for different samples of data that are 90% of the whole dataset
#                        n_left_out=len(date_inds)/10
#                        nsamples=len(date_inds)/n_left_out
#                        mult_fc_sub=np.empty(nsamples, dtype=np.object)    
#                        weights_sub=np.zeros((nsamples,len(models)))
#                        for i in range(nsamples):
#                            date_inds_sub=np.concatenate((date_inds[:n_left_out*i],date_inds[n_left_out*(i+1):]))
#                            inds_tup_sub=(inds[0],list(date_inds_sub))
#                            X = np.array([np.mean(data_sm[model][inds_tup_sub],axis=-1) for model in models]).T
#                            bmaobj = BMARaftery(X, data_veri[inds_tup_sub])
#                            mult_fc_sub[i]=bmaobj.get_pdf(x)
#                            weights_sub[i,:]=mult_fc_sub[i].weights
#                        stop

                        mult_fc[ind].models=models  #add list of models corresponding to derived weights
#                        print ind
#                        print mult_fc[ind](300)
#                        from python_gen import integrate_quad_gen
#                        print integrate_quad_gen(mult_fc[ind], -np.inf, np.inf)
#                        fn=lambda y: y*mult_fc[ind](y)
#                        fn.loc=mult_fc[ind].loc
#                        print mult_fc[ind].mean, integrate_quad_gen(fn, -np.inf, np.inf)
#                        fn=lambda y: (y-mult_fc[ind].mean)**2 * mult_fc[ind](y)
#                        fn.loc=mult_fc[ind].loc
#                        print mult_fc[ind].sigma**2, integrate_quad_gen(fn, -np.inf, np.inf)
                    elif fnmatch.fnmatch(mult,'sf*'):
                        X = np.array([np.mean(data_sm[model][inds],axis=-1) for model in models]).T
                        if mult=='sf':
                            sfobj = sensor_fusion(X, data_veri[inds], cond_independence=True)
                        elif mult=='sfc':
                            sfobj = sensor_fusion(X, data_veri[inds], cond_independence=False) # consider the correlations/covariance
                        mult_fc[ind] = sfobj.get_pdf(x)
                        
                    elif mult=='gp':
                        X = np.array([np.mean(data_sm[model][inds],axis=-1) for model in models]).T  #predictors are ensemble means of forecasts from each model
                        mult_fc[ind]=gp(X,data_veri[inds],x)
                    
                    elif mult=='linfit':
                        X = np.array([np.mean(data_sm[model][inds],axis=-1) for model in models]).T  #predictors are ensemble means of forecasts from each model
                        mult_fc[ind]=linfit_gauss(X,data_veri[inds],x)
                        mult_fc[ind].models=models  #add list of models corresponding to derived coeffs
                        
                        #For testing that the linfit forecast beats the equal weights forecast when all the data are used in the regression.
                        #X2 = np.array([np.mean(data_sm[model][inds[0]],axis=-1) for model in data_sm]).T
                        #fc_all_data[inds[0],date_ind]=linfit_gauss(X2,data_veri[4],x).mean                      
                        
                    elif mult=='linfit-bs':
                        #Do linear regression, with bootstrap resampling of training dates
                        nsamples=100  #no of resamples - haven't experimented with finding the best value for this yet.
                        #nsamples=3
#                       inds_arr=np.concatenate((np.repeat(inds[0],ndates-1).reshape(1,ndates-1),np.array(inds[1]).reshape(1,ndates-1)))
#                        if len(data_sm_iter.multi_index)>1:
#                            for dim in range(1,len(data_sm_iter.multi_index)):
#                                inds_arr=np.concatenate((inds_arr,np.repeat(data_sm_iter.multi_index[dim],ndates-1).reshape(1,ndates-1)))
                        
                        #inds_bs=gen_rnd_arr_bootstrap(np.array(inds_arr).swapaxes(0,1), nsamples).swapaxes(1,2)                     
                        
                        date_inds_bs=gen_rnd_arr_bootstrap(inds[1],nsamples)
                        
                        fn_arr=np.empty(nsamples, dtype=np.object) #array to hold the linear forecast model fitted to each bootstrap sample
                        means=np.zeros((nsamples))
                        sigmas=np.zeros((nsamples))
                        for i in range(nsamples):
                            inds_bs_tup=(inds[0],list(date_inds_bs[i]))
                            if len(inds)>2:
                                inds_bs_tup=inds_bs_tup+inds[2:]
                            X = np.array([np.mean(data_sm[model][inds_bs_tup],axis=-1) for model in models]).T
                            
                            fn=linfit_gauss(X,data_veri[inds_bs_tup],x)
                            means[i]=fn.mean
                            sigmas[i]=fn.sigma
                            
                            #print fn_arr[i](300)
                            #print fn(300)
                        
                        mult_fc[ind]=gauss_mix_pdf(means,sigmas)
                        mult_fc[ind].loc=np.mean(means)
                        mult_fc[ind].models=models  #add list of models corresponding to derived coeffs
                        
                        #print mult_fc[ind](300)
                        #stop
                    
                    data_sm_iter.iternext()
        
    else:
        print 'get_multimodel_fc: model combination method '+mult+' not recognised'

    return mult_fc
