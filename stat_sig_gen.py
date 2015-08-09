#Script to hold functions for assessing statistical significance e.g. calculating p-values.

import numpy as np

#Function to return p-values for differences between two arrays averaged over their first dimension (which may correspond to time, say), using a given method. For methods where random time series are produced e.g. permutation or bootstrap tests, give n, the number of random time series to produce.
#The convention is that the smaller the p-value, the greater the statistical significance.
def get_p_values_diffs(mean_diff,ts1,ts2,method,n=100):
    from numpy.random import shuffle
    
    if type(mean_diff)==np.ndarray:
        assert mean_diff.shape==ts1.shape[1:], "Shape mismatch: mean_diff.shape="+str(mean_diff.shape)+", ts1.shape="+str(ts1.shape)
        assert mean_diff.shape==ts2.shape[1:], "Shape mismatch: mean_diff.shape="+str(mean_diff.shape)+", ts1.shape="+str(ts2.shape)
    else:  #when mean_diff is an integer and ts1 and ts2 are vectors.
        assert type(ts1)==np.ndarray or type(ts1)==list or ts1.ndim==1
        assert type(ts2)==np.ndarray or type(ts2)==list or ts2.ndim==2
    
    #Finding where diffs are more than two standard errors from zero
    #This returns "p-values" of zero where the diffs are more than 2 s.e. and 1 otherwise i.e. just to indicate where values are statistically significant by this measure.
    if method=='2se':
        std_err=np.sqrt(np.std(ts1,axis=0)**2/ts1.shape[0] + np.std(ts2,axis=0)**2/ts2.shape[0])
        
        #Get differences as fraction of standard error - make large where mean_diff>0 and the standard error is zero e.g. because the time series are constant, and zero where mean_diff=0.
        diffs_over_se=np.zeros(mean_diff.shape)

        inds=np.where(std_err>0.)
        diffs_over_se[inds]=abs(mean_diff[inds]/std_err[inds])

        inds=np.where((abs(mean_diff)>0) & (std_err==0.))
        diffs_over_se[inds]=np.inf
        
        p_values=np.zeros(mean_diff.shape)
        p_values[diffs_over_se<2]=1.

    #Monte Carlo permutation test        
    elif method=='perm':
        ts_all=np.concatenate((ts1,ts2))
        inds=range(len(ts_all))
        diff_rnd_arr=np.zeros((n,)+ts_all.shape[1:])
        for i in range(n):
            shuffle(inds)
            inds_rnd1=inds[:len(ts1)]
            inds_rnd2=inds[len(ts1):]
            diff_rnd_arr[i,...]=np.mean(ts_all[inds_rnd1,...],axis=0) - np.mean(ts_all[inds_rnd2,...],axis=0)
            #if i%100==0:
                #print i
                #print diff_rnd_arr[i]
                #print inds, inds_rnd1, inds_rnd2
                #print len(inds), len(inds_rnd1), len(inds_rnd2)
                #print ts_all[inds_rnd1,...], ts_all[inds_rnd2,...]
                #print ''
        
        if type(mean_diff)==float:  #for when mean_diff is a float and ts1 and ts2 are vectors, so that just one p-value is returned
            p_values=len(np.where(abs(diff_rnd_arr) > abs(mean_diff))[0])/float(n)
        else:
            p_values=np.zeros(mean_diff.shape)
            mean_diff_iter=np.nditer(mean_diff, flags=['multi_index','refs_ok'])
            while not mean_diff_iter.finished:
                ind=mean_diff_iter.multi_index
                diff_ind_tup=(range(n),)+tuple([[i]*n for i in ind])  #tuple of indices to select all values in the first dimension and just the indices in ind in the remaining dimensions in diff_rnd_arr
                p_values[ind]=len(np.where(abs(diff_rnd_arr[diff_ind_tup]) > abs(mean_diff[ind]))[0])/float(n) #use absolute values to make the test 2-tailed.
                mean_diff_iter.iternext()
    
    return p_values
    
#Function to return p-values for differences between differences between two pairs of arrays (ts1_1 minus ts1_2 and ts2_1 minus ts2_2) averaged over their first dimension (which may correspond to time, say), using a Monte Carlo permutation method. diff_diff is the array of differences of differences in the data, and n is the number of random time series to produce.
#The convention is that the smaller the p-value, the greater the statistical significance.
#NEEDS TO BE TESTED PROPERLY
def get_p_values_diff_diff(diff_diff,ts1_1,ts1_2,ts2_1,ts2_2,n=100):
    from numpy.random import shuffle

    assert ts1_1.shape[1:]==ts1_2.shape[1:]
    assert ts1_1.shape[1:]==ts2_1.shape[1:]
    assert ts1_1.shape[1:]==ts2_2.shape[1:]

    ts1_all=np.concatenate((ts1_1,ts1_2))
    ts2_all=np.concatenate((ts2_1,ts2_2))
    inds1=range(ts1_all.shape[0])
    inds2=range(ts2_all.shape[0])
    diff_diff_rnd=np.zeros((n,)+ts1_all.shape[1:])
    for i in range(n):
        shuffle(inds1)
        shuffle(inds2)
        inds_rnd1_1=inds1[:ts1_1.shape[0]]
        inds_rnd1_2=inds1[ts1_1.shape[0]:]
        inds_rnd2_1=inds2[:ts2_1.shape[0]]
        inds_rnd2_2=inds2[ts2_1.shape[0]:]
        diff1_rnd=np.mean(ts1_all[inds_rnd1_1,...],axis=0) - np.mean(ts1_all[inds_rnd1_2,...],axis=0)
        diff2_rnd=np.mean(ts2_all[inds_rnd2_1,...],axis=0) - np.mean(ts2_all[inds_rnd2_2,...],axis=0)
        diff_diff_rnd[i,...]=diff1_rnd-diff2_rnd
    
    p_values=np.zeros(diff_diff.shape)
    iter=np.nditer(diff_diff, flags=['multi_index','refs_ok'])
    while not iter.finished:
        ind=iter.multi_index
        diff_ind_tup=(range(n),)+tuple([[i]*n for i in ind])  #tuple of indices to select all values in the first dimension and just the indices in ind in the remaining dimensions in diff_rnd_arr
        p_values[ind]=len(np.where(abs(diff_diff_rnd[diff_ind_tup]) > diff_diff[ind])[0])/float(n) #use absolute values to make the test 2-tailed.
        iter.iternext()
    
    return p_values

#Function to generate random array sequences by the bootstrap method, whereby the sequences are generated by sampling values from the original sequence with replacement, giving each value equal weight. The dimension along which resampling is done (e.g. time to resample a time series) is assumed to be the first dimension. The returned array has leading dimension corresponding to the different random sequences. n is the number of random samples.
def gen_rnd_arr_bootstrap(arr, n):
    from numpy.random import randint
    
    if type(arr)==list:
        arr=np.array(arr)
    
    bootinds = randint(arr.shape[0],size=(n,arr.shape[0]) )  #returns n x arr.shape[0] array of random indices
    
    #Generating the random time series
    arr_rnd=np.zeros((n,)+(arr.shape))
    for i in range(n):
        arr_rnd[i,...]=arr[bootinds[i,:],...]
    
    return arr_rnd