#File containing general python functions relating to forecast verification

#Function to return forecasts of whether an event occurs given forecasts of a continuous variable, based on event thresholds relative to the climatology of forecasts. An array of zeros (no event) and ones (event occurs) is returned.
#fc is a numpy array containing forecast data. The first and second dimensions are assumed to be the different start times and ensemble members (in either order). Then there can be extra dimensions for different lead times, spatial points etc., which are treated independently.
#event_pm and event_thres define the event. event_pm is either '+' to define events as the forecast variable exceeding the threshold and '-' to fall below it. event_thres is a fraction between 0 and 1 defining the threshold e.g. '+' and 2./3 defines events as the variable being in the upper tercile, '-' and 1./2 defines events as the variable being in the bottom half etc.
def get_event_fc_rel(fc, event_pm, event_thres):
    import numpy as np
    
    if event_pm not in ['>','<']:
        import sys
        print "get_event_fcs_rel: event_pm should be '>' or '<'"
        sys.exit()
    
    fc_shape_orig=fc.shape
    fc.reshape(fc.shape[0],fc.shape[1],-1)
    
    fc_ev=np.zeros(fc.shape)
    for i in range(fc.shape[2]):
        abs_thres=np.percentile(fc[:,:,i],100*event_thres)  #The event_thres percentile of the climatological distribution estimated from all start dates and ensemble members
        
        #Loop over start dates and ensemble members and determine whether an event is forecast
        #print abs_thres
        for j in range(fc.shape[0]):
            for k in range(fc.shape[1]):
                if event_pm=='>':
                    fc_ev[j,k,i] = fc[j,k,i] >= abs_thres
                elif event_pm=='<':
                    fc_ev[j,k,i] = fc[j,k,i] <= abs_thres
    
    fc_ev.reshape(fc_shape_orig)
    
    return fc_ev


#Function to return verification (specified by string diag) as a function of lead time for an array of given forecast data (fc, where the first dimension is assumed to be the lead time, the second the different forecast start times, and the last the ensemble members, if applicable) compared against verification data veri (the elements of which correspond to the elements of fc, except without an ensemble member dimension). These arrays should contain either continuous values of a variable or event probabilities, as appropriate for the diag argument. The single_ens_member option specifies that there is only one ensemble member, so the last dimension should not be treated as being the ensemble member dimension. nbins specifies how many bins to use for diagnostics that require data to be binned.
#TO DO:
#Get this to work with forecasts that are functions. See here for how to integrate over functions: http://docs.scipy.org/doc/scipy/reference/tutorial/integrate.html
#get this to work with arrays with extra "spatial" dimensions with 'rmse' option
#add option to return separate components of Brier score
def get_fc_diag(fc, veri, diag, single_ens_member=None, nbins=10):
    import numpy as np
    from scipy.integrate import quad
    from python_gen import heaviside, integrate_quad_gen 

    if single_ens_member: 
        fc=fc[...,np.newaxis]  #add "effective ensemble member" dimension to total
    
    fc_ndim=fc.ndim
    if fc_ndim>4:
        fc_shape_orig=fc.shape
        
        fc=fc.reshape(fc.shape[0],fc.shape[1],np.prod(fc.shape[2:-1]),fc.shape[-1])  #flatten dimensions of fc other than the lead time, start time and ensemble member dimensions
        diag_arr=np.zeros((fc.shape[0],fc.shape[2])) #copy all dimensions except the start time and ensemble member dimensions
    else:
        diag_arr=np.zeros((fc.shape[0]))

    #Anomaly correlation coefficient
    if diag=='acc':
        #stop        
        #Get ensemble mean
        if hasattr(fc[(0,)*fc.ndim], '__call__'):  #for forecasts that are functions
            ens_mean=np.zeros((fc.shape))
            ens_mean_iter=np.nditer(ens_mean, flags=['multi_index','refs_ok'])
            while not ens_mean_iter.finished:
                ind=ens_mean_iter.multi_index
                if 'mean' in dir(fc[ind]):
                    ens_mean[ind]=fc[ind].mean
                else:  #getting mean using integral of forecast pdf
                    fn=lambda x: x*fc[ind](x)
                    fn.loc=fc[ind].loc
                    ens_mean[ind]=integrate_quad_gen(fn,-np.inf,np.inf)[0]
                #print ind, ens_mean[ind]
                ens_mean_iter.iternext()
        else:  #for ordinary arrays containing forecasts for different ensemble members
            ens_mean=np.mean(fc,axis=-1)
            
        diag_arr_iter=np.nditer(diag_arr, flags=['multi_index','refs_ok'])   
        while not diag_arr_iter.finished:
            ind=diag_arr_iter.multi_index
            if len(ind)>1:
                diag_arr[ind]=np.corrcoef(ens_mean[ind[0],:,ind[1:]],veri[ind[0],:,ind[1:]])[0,1]
            else:
                diag_arr[ind]=np.corrcoef(ens_mean[ind[0],:],veri[ind[0],:])[0,1]
            diag_arr_iter.iternext()
            
#            print ens_mean[ind[0],:]
#            print veri[ind[0],:]
#            print diag_arr[ind[0]]
#            stop
    
    #Brier score
    elif diag=='bs':
        diag_arr=np.mean((fc-veri)**2,axis=1)
    
    #Continuous ranked probability score
    elif diag=='crps':
        if hasattr(fc[(0,)*fc.ndim], '__call__'):  #for forecasts that are functions
            crps=np.zeros(fc.shape)
            for i in range(fc.shape[1]):  #loop over forecast lead times
                #print i
                fc_temp=fc[:,i,...]
                veri_temp=veri[:,i,...]
                fc_temp_iter=np.nditer(fc_temp, flags=['multi_index','refs_ok'])   
                while not fc_temp_iter.finished:  #loop over other indices
                    ind=fc_temp_iter.multi_index
                    
                    #Make integrand for calculation of CRPS
                    
                    #This could be simplified - since veri_cdf is 0 for x<veri_temp[ind], the integral is just (1-fc_cdf)**2 over the interval veri_temp[ind] to infinity.
                    
                    #fc_cdf=lambda x: integrate_quad_gen(fc_temp[ind],-np.inf,x, min_i_for_error=1e-4,tol=1e-4)[0]  #accept higher error tolerances than usual here, else errors get thrown where the cdf is nearly 0 or 1. Errors of this size shouldn't matter, since the cdf values are squared below.
                    fc_cdf=lambda x: quad(fc_temp[ind],-np.inf,x)[0] #thought this might be faster than the above - worth making heaviside function explicit in the next line too?
                    veri_cdf=lambda x: heaviside(x,loc=veri_temp[ind])
                    fn=lambda x: (fc_cdf(x)-veri_cdf(x))**2
                    #fn=lambda x: (integrate_quad_gen(fc_temp[ind],-np.inf,x,min_i_for_error=1e-4,tol=1e-4)[0]-heaviside(x,loc=veri_temp[ind]) #tested if combining the functions into one is any faster, but it doesn't seem to be.
                    fn.loc=veri_temp[ind]
                    
                    #Can't set integration limits to -/+ inf for good results, so find the standard deviation of fc_temp[ind] and use that to set the limits.
                    try:
                        sigma=fc_temp[ind].sigma
                    except:
                        try:
                            fc_mean=fc_temp[ind].mean
                        except:
                            fn2=lambda x: x*fc_temp[ind](x)
                            fn2.loc=fc_temp[ind].loc
                            fc_mean=integrate_quad_gen(fn2,-np.inf,np.inf)[0]

                        fn3=lambda x: (x-fc_mean)**2 * fc_temp[ind](x)
                        fn3.loc=fc_temp[ind].loc
                        sigma=np.sqrt(integrate_quad_gen(fn3,-np.inf,np.inf)[0])
                    
                    a=fc_temp[ind].loc-10*sigma
                    b=fc_temp[ind].loc+10*sigma
                    
                    #Calculate the CRPS here.
                    crps_temp=integrate_quad_gen(fn,a,b)[0]
                    if len(ind)>1:                    
                        crps[ind[0],i,ind[1:]]=crps_temp
                    else:
                        crps[ind[0],i]=crps_temp
                    
                    #if i==1: stop
                    fc_temp_iter.iternext()
                    #print 'finished loop'

            diag_arr=np.mean(crps,axis=1)
        else:
            diag_arr=get_fc_diag(fc, veri, 'rps', single_ens_member=single_ens_member, nbins=nbins)
    
    #Root mean square error of ensemble-mean anomalies
    elif diag=='rmse':
    
        #Get ensemble-mean forecasts and array of varification data 
        if hasattr(fc[(0,)*fc.ndim], '__call__'):  #for forecasts that are functions
            ens_mean=np.zeros((fc.shape))
            ens_mean_iter=np.nditer(ens_mean, flags=['multi_index','refs_ok'])
            while not ens_mean_iter.finished:
                ind=ens_mean_iter.multi_index
                if 'mean' in dir(fc[ind]):
                    ens_mean[ind]=fc[ind].mean
                else:  #getting mean using integral of forecast pdf
                    fn=lambda x: x*fc[ind](x)
                    fn.loc=fc[ind].loc
                    ens_mean[ind]=integrate_quad_gen(fn,-np.inf,np.inf)[0]
#                fn=lambda x: x*fc[ind](x)
#                fn.loc=fc[ind].loc
#                ens_mean[ind]=integrate_quad_gen(fn,-np.inf,np.inf)[0]
                #print ind, ens_mean[ind]
                ens_mean_iter.iternext()

            veri_anom=veri
            
        else:
            #Get forecast and verification anomalies
            fc_anom=np.zeros(fc.shape)
            veri_anom=np.zeros(veri.shape)
            for i in range(diag_arr.shape[0]):
                fc_anom[i,...]=fc[i,...]-np.mean(fc[i,...])
                veri_anom[i,...]=veri[i,...]-np.mean(veri[i,...])
            
            ens_mean=np.mean(fc_anom,axis=-1)
            
        for i in range(diag_arr.shape[0]):
            diag_arr[i]=np.sqrt(np.mean((ens_mean[i,:]-veri_anom[i,:])**2))
    
    #Root mean square spread of forecasts - this does not need verification data, but it can simplify other code to include the option to calculate it here.
    elif diag=='rmss':
        diag_arr=get_rmss(fc)
    
    #Rank probability score
    elif diag=='rps':
    
        #Get forecast and verification anomalies
        fc_anom=np.zeros(fc.shape)
        veri_anom=np.zeros(veri.shape)
        bin_edges=np.zeros((fc.shape[0],nbins+1))
        fc_freq=np.zeros(fc.shape[:2]+(nbins,))
        veri_freq=np.zeros(veri.shape[:2]+(nbins,))
        for i in range(fc.shape[0]):  #Loop over forecast lead times
            fc_anom[i,...]=fc[i,...]-np.mean(fc[i,...])
            veri_anom[i,...]=veri[i,...]-np.mean(veri[i,...])
        
            #Get bin edges based on climatological distribution of anomalies for each forecast month
            freq_veri, bin_edges[i,:] = np.histogram(veri_anom[i,...], bins=nbins)
            freq_veri=freq_veri/float(np.sum(freq_veri))  #convert frequencies to probabilities
            bin_edges[i,0]=-np.inf  #make bins open at the ends of the range
            bin_edges[i,-1]=np.inf
            bin_width=bin_edges[i,2]-bin_edges[i,1]
    
            #Get frequency of forecast anomalies in each bin for each forecast date
            rps=np.zeros((fc.shape[1]))
            for j in range(fc.shape[1]):
                fc_freq[i,j,:], bins_temp = np.histogram(fc_anom[i,j,...], bins=bin_edges[i,:])
                fc_freq[i,j,:]=fc_freq[i,j,:]/np.sum(fc_freq[i,j,:])
                
                veri_freq[i,j,:], bins_temp = np.histogram(veri_anom[i,j,...], bins=bin_edges[i,:])  #there is only one verification point, so no need to normalise
                
                #Get cdfs
                fc_cdf=np.cumsum(fc_freq[i,j,:])  #np.cumsum returns the cumulative density up to the most positive edge of each bin
                veri_cdf=np.cumsum(veri_freq[i,j,:])
                
                rps[j]=np.sum((fc_cdf-veri_cdf)**2)*bin_width  #rank probability score. Multiply by bin_width to make this equivalent to the calculation of the CRPS in the limit of the bin width tending to zero.
            
            diag_arr[i]=np.mean(rps)  #return mean rps over all forecasts
            
    else:
        print 'get_fc_diag: '+diag+' is not a valid option for diag'
        import sys
        sys.exit()
    
    if fc_ndim>4:
        diag_arr=diag_arr.reshape(fc_shape_orig[:-1])
    
    return diag_arr


#Function to return the root mean square spread of an array of forecast data, with the first dimension of fc assumed to correspond to different forecast start times and the last to different ensemble members.
def get_rmss(fc):
    import numpy as np
    from python_gen import get_fn_sd

    if hasattr(fc[(0,)*fc.ndim], '__call__'):  #for forecasts that are functions
        rmss=np.zeros(fc.shape)
        for i in range(fc.shape[1]):  #loop over forecast lead times
            fc_temp=fc[:,i,...]
            fc_temp_iter=np.nditer(fc_temp, flags=['multi_index','refs_ok'])   
            while not fc_temp_iter.finished:  #loop over other indices
                ind=fc_temp_iter.multi_index
                rmss[ind]=get_fn_sd(fc_temp[ind])
                fc_temp_iter.iternext()
        
    else:
        fc_anom=np.zeros(fc.shape)
        for i in range(fc.shape[0]):
            fc_anom[i,...]=fc[i,...]-np.mean(fc[i,...])  #anomaly from the mean forecast over all forecast start times and ensemble members for each lead time.
    
        ens_mean=np.mean(fc_anom,axis=-1)
        rmss=np.zeros(fc.shape[:-1])
        rmss_iter=np.nditer(rmss, flags=['multi_index'])
        while not rmss_iter.finished:
            ind=rmss_iter.multi_index
            rmss[ind]=np.mean((fc_anom[ind]-ens_mean[ind])**2)
            rmss_iter.iternext()
        
        rmss=np.mean(rmss,axis=1)
        rmss=np.sqrt(rmss)
    
    return rmss


#Function to take a skill score name and return the corresponding forecast score name to be used as an argument to get_fc_diag() (or return the argument if there is no skill score usually associated with the forecast diagnostic e.g. for anomaly correlations etc.)
def get_score_name(skill_score):
    #Dict mapping skill score names to forecast score names
    score_names={ 'acc':'acc',
                  'bss':'bs',
                  'crpss':'crps',
                  'rmse':'rmse',
                  'rmss':'rmss',
                  'rpss': 'rps'
                 }
    return score_names[skill_score]

#Function to calculate skill scores given the forecast score, the score of a reference forecast and the perfect score.
def get_skill_score(score_fc, score_ref, score_perf):
    return (score_fc-score_ref)/(score_perf-score_ref)
