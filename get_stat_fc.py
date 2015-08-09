#Python function to produce statistical forecasts
#Input data is:
#obs_data - array containing observational data
#obs_times - array with same shape as obs_data containing the dates and times of the values in obs_data (as datetime.datetime objects)
#times_out - array containing desired forecast times (as datetime.datetime objects). Returned array of forecasts will have the same shape as this array, plus a dimension for the different "ensemble members". The first dimension is assumed to correspond to forecasts at different lead times, starting from the same date.

#stat is an input string specifying the method of making the statistical forecast. Options are:
#strings beginning with 'clim' for climatological forecasts ('clim' or 'clim_cross#' with # a number, where for each year the current year and the previous #-1 years are disregarded from the climatology - 'clim_cross0' is equivalent to 'clim')
#'pers' for persistance forecast.

#timedelta is a datetime.timedelta object that specifies:
#For climatological forecasts: how far away from the calendar date of each element in dates_out an observational date can be to be considered part of the climatology for that date e.g. if this corresponds to 3 weeks, then all observational data for calendar dates from 3 weeks before to 3 weeks after will be considered for forming the climatology. The default value is 27 days (so that when considering monthly data, values for all the different calendar months are kept separate).
#For persistence forecasts: the length of time preceding the forecast date over which to average observed data to give the forecast value. Default is 1 day.

#start_dates is used for persistence forecasts - it contains an array with the forecast start dates, with the same shape as times_out except without the first dimension.

#Note that for climatological forecasts, the script truncates the data so that every time in times_out has the same number of "ensemble members" - this is done so that the output can be stored simply in a numpy array. Note that this means that if the verification data starts or ends at a point in the year included in the calendar period being forecast (as given in times_out), then the "ensemble members" returned will not be continuous time series of observed data.

#TO DO:
#allow forecasts over a given spatial region to be made e.g. Assume that dimensions in obs_data not in obs_times are spatial data, and return this as an extra dimension to stat_fc.

import numpy as np
import sys
import datetime as dt
import fnmatch

#Function to set the year of a datetime object to an arbitrary value - here to 1
def set_arbitrary_yr(times):
    try:
        #for arrays, lists etc.
        a=iter(times)
        time=np.nditer(np.array(times), flags=['multi_index','refs_ok'])
        while not time.finished:
            ind=time.multi_index[0]
            times[ind]=times[ind]+(dt.datetime(1,times[ind].month,times[ind].day,times[ind].hour,times[ind].second) - times[ind])
            time.iternext()
    except:
        #for single values
        times=times+(dt.datetime(1,times.month,times.day,times.hour,times.second) - times)
        
    return times


def get_stat_fc(obs_data, obs_times, times_out, stat, timedelta=None, fc_start_dates=None):

    #Climatological forecasts
    if fnmatch.fnmatch(stat,'clim*'):
        
        if timedelta==None:
            timedelta=dt.timedelta(0,0,0,0,0,27*24)  #default is 27 days.
            
        #Get no. of previous years of data to remove for each forecast date, including that for the forecast date.
        if fnmatch.fnmatch(stat,'clim_cross*'):
            cross=int(stat[10:])
        elif stat=='clim':  #for clim forecasts using all data, including that for the forecast date.
            cross=0
        #print 'cross=',cross
        
        #Get the number of observations within timedelta of each time in times_out, except for the year of that datapoint and the cross-1 years before. Make the stat_fc array with last dimension corresponding to the smallest of these values. For times where more data is available, the data series is truncated so that data coming later in the obs_data array are not used - this allows stat_fc to be stored as a numpy array, keeping things simpler. It may not always be appropriate, though.
        ntimes=np.zeros((times_out.shape))
        time_iter=np.nditer(times_out, flags=['multi_index','refs_ok'])  #refs_ok flag needed to work with array of datetimes
        while not time_iter.finished:
            ind=time_iter.multi_index
            ntimes[ind]=len([time for time in np.ravel(obs_times) 
                if ( (abs(set_arbitrary_yr(time)-set_arbitrary_yr(times_out[ind]))<=timedelta) and 
                     (time.year-times_out[ind].year >0 or time.year-times_out[ind].year < -(cross-1)) )           
                    ])
            time_iter.iternext()
        
        n_ens=np.min(ntimes)
        
        stat_fc=np.zeros(times_out.shape+(n_ens,))
        
        #Get array of year values in obs_times
        obs_years=np.zeros(obs_times.shape)
        time_iter=np.nditer(obs_years, flags=['multi_index','refs_ok'])
        while not time_iter.finished:
            ind=time_iter.multi_index
            obs_years[ind]=obs_times[ind].year
            time_iter.iternext()
        
        #Loop over the times in times_out, and get the climatological forecast for each time
        time_iter=np.nditer(times_out, flags=['multi_index','refs_ok'])
        while not time_iter.finished:
            ind=time_iter.multi_index
            stat_fc[ind]=obs_data[np.where( ( abs(set_arbitrary_yr(obs_times)-set_arbitrary_yr(times_out[ind]))<=timedelta ) & 
                     ( (obs_years-times_out[ind].year >0) | (obs_years-times_out[ind].year < -(cross-1)) ) )][:n_ens]
            time_iter.iternext()
        
   
    #Persistence forecasts
    elif stat=='pers':
        
        if not timedelta:
            timedelta=dt.timedelta(0,0,0,0,0,1*24)  #default is 1 day.

        #Get the climatology in the verification dataset for the calendar date range preceding each forecast start calendar date by not more than timedelta.
        clim=np.zeros(fc_start_dates.shape)
        clim_iter=np.nditer(clim, flags=['multi_index'])
        while not clim_iter.finished:
            ind=clim_iter.multi_index
            clim[ind]=np.mean(obs_data[np.where( (set_arbitrary_yr(fc_start_dates[ind]) - set_arbitrary_yr(obs_times.copy()) <= timedelta) & (set_arbitrary_yr(fc_start_dates[ind]) - set_arbitrary_yr(obs_times.copy()) >= dt.timedelta(0)) )]) #the second condition ensures that only data before the forecast start date is used.
            clim_iter.iternext()
        
        #Get the mean of the forecast data preceding the forecast start date by not more than timedelta, and subtract the climatological value to get the anomaly.
        fc_data=np.zeros(fc_start_dates.shape)
        fc_iter=np.nditer(fc_data, flags=['multi_index'])
        while not fc_iter.finished:
            ind=fc_iter.multi_index
            fc_data[ind]=np.mean(obs_data[np.where((fc_start_dates[ind] - obs_times <= timedelta) & (fc_start_dates[ind] - obs_times >= dt.timedelta(0)))]) - clim[ind]
            fc_iter.iternext()
        
        #Repeat the forecast value for all forecast dates - tile the fc_data array to repeat this value along the forecast lead time dimension, keeping other dimensions unchanged.
        tile_tup=(times_out.shape[0],)
        for i in range(len(times_out.shape)-1):
            tile_tup=tile_tup+(1,)
        stat_fc=np.tile(fc_data, tile_tup)[...,np.newaxis]  #add axis of length 1 to represent the ensemble member dimension
        
    else:
        print 'get_stat_fc: '+stat+' is not a valid option for stat'
        sys.exit()
        
    return stat_fc
