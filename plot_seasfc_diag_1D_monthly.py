#Python script to read in single-model seasonal forecast monthly-mean 1D data, produce multi-model combined forecasts and statistical forecasts for comparison, and plot some verification diagnostics.

#TO DO:
#Allow more flexibility in the choice of period prior to forecast start dates when making persistence forecasts? May require adding code to fetch more data if necessary.

import numpy as np
import matplotlib.pyplot as plt
import sys
import socket
import argparse
import datetime as dt
import fnmatch
import scipy.signal  #for detrending
from Scientific.IO.NetCDF import NetCDFFile
from fc_verification import get_event_fc_rel, get_fc_diag, get_skill_score, get_score_name
from python_gen import convert_fracstr_to_flt 
from plotting_gen import make_space_for_legend
from get_multimodel_fc import get_multimodel_fc
from get_stat_fc import get_stat_fc

#Function to return the data location
def get_data_dir_main():
    data_dir_main='/homes/49/edwin/robots_code/weather/'
    return data_dir_main
  

#Function to loop over data in dictionary data_fc (with keys start_dates, which return nested dictionaries with keys models) and get forecasts of whether the events in the given list will occur.
def get_event_fc_loop(data_fc, events):

    event_fc={}
    for event in events:
        event_fc[event]={}

        event_pm=event[0]
        event_thres=convert_fracstr_to_flt(event[1:])
        
        for start_date in start_dates:
            event_fc[event][start_date]={}
            
            for model in data_fc[start_date]:
                event_fc[event][start_date][model]=get_event_fc_rel(np.rollaxis(np.rollaxis(data_fc[start_date][model]['data'],2),2),event_pm,event_thres)  #rollaxis commands to get ens_member and start_year dimensions at the front
                event_fc[event][start_date][model]=np.rollaxis(event_fc[event][start_date][model],2)

    return event_fc
    
#Function to loop over verification datasets, start months and models and return a dictionary containing the values of the given diagnostics as a function of forecast month. data_fc is forecast data, fc_veri is the verification data and diag is a string specifying the diagnostic to get.
def get_fc_diag_loop(data_fc, fc_veri, diag):
    diag_dict={}
    for veri in fc_veri:
        diag_dict[veri]={}
        for start_date in fc_veri[veri]:
            diag_dict[veri][start_date]={}
            for model in data_fc[start_date]:
                print 'Getting '+diag+' for '+model
                diag_dict[veri][start_date][model]=get_fc_diag(data_fc[start_date][model]['data'],fc_veri[veri][start_date][model],diag)
                
    return diag_dict           

#Function to return an array of verification data, given a verification dataset and an array of dates for which verification data is required.
def get_fc_veri(data_veri,data_veri_dates,data_fc_dates):
    fc_veri_arr=np.zeros(data_fc_dates.shape)
    
    date=np.nditer(data_fc_dates, flags=['multi_index'])
    while not date.finished:
        fc_veri_arr[date.multi_index]=data_veri[data_veri_dates == date[0]]
        date.iternext()
    
    return fc_veri_arr


#Function to return the part of the input filenames specifying the coordinate values. lat, lon and plev are either 2-element lists giving the coordinate ranges over which data have been averaged, or single values for data at a single point on the axis.
def get_fname_coords(lat,lon,plev=None):
    
    fname_coords=''
    if plev:
        if type(plev)==list and len(plev)==2:
            fname_coords=fname_coords+'_'+str(plev[0])+'_'+str(plev[1])
        else:
            fname_coords=fname_coords+'_'+str(plev)
        
    if type(lat)==list and len(lat)==2:
        fname_coords=fname_coords+'_'+str(lat[0])+'_'+str(lat[1])
    else:
        fname_coords=fname_coords+'_'+str(lat)
        
    if type(lon)==list and len(lon)==2:
        fname_coords=fname_coords+'_'+str(lon[0])+'_'+str(lon[1])
    else:
        fname_coords=fname_coords+'_'+str(lon)
        
    if (plev and type(plev)==list and len(plev)==2) or (type(lat)==list and len(lat)==2) or (type(lon)==list and len(lon)==2): fname_coords=fname_coords+'_mean'  #for averaged data
    
    return fname_coords

#Function to get names of variables used in datafiles, taking common names as arguments. Set inverse to get long name for a given short name.
def get_var_data_name(var, inverse=False):
    
    #Dict mapping intuitive variable names to names of folders in which data are kept
    long_names=['1.5m T','2m T','T2m','T','sea ice fraction','sea ice','sst',  'MSLP','mslp','Total precip']
    short_names=['T_1.5', '2t', '2t' ,'t',      'ci'        ,   'ci'  , 'stl1','msl', 'msl',    'tp' ]
    assert len(long_names)==len(short_names), 'No. of long names should equal number of short names.'
    
    if inverse:  #getting long name for given short name - if multiple keys in var_shortnames correspond to the long name, the first long name gets returned.
        if var in short_names:
            long_name=[long_names[ind] for ind in range(len(long_names)) if short_names[ind]==var][0]
            return long_name
        else:
            return var  #Just returns the input if there is no corresponding short name
    else:
        if var in long_names:
            #return var_shortnames[var]
            short_name=[short_names[ind] for ind in range(len(short_names)) if long_names[ind]==var][0]
            return short_name
        else:
            return var

#Function to read in data for individual dynamical models, returned in a nested dictionary - extract data with data_fc_sm[start_date][model]['data'] and corresponding array of verification months with data_fc_sm[start_date][model]['verifying_month'], with start_date and model being strings. dataset is the multimodel dataset e.g. 'ENSEMBLES'. start dates is a list of strings specifying the forecast start dates to retrieve data for. var is the string specifying the variable. lat, lon and plev are either 2-element lists giving the coordinate ranges over which data have been averaged, or single values for data at a single point on the axis. models is a list of strings specifying which models to retrieve data for, or 'all' to get data for all models in the dataset.
def get_single_model_data(dataset,start_dates,var,lat,lon,plev=None,models='all',n_ens=None):
    
    data_dir_main=get_data_dir_main()
    
    fname_coords=get_fname_coords(lat,lon,plev=plev)
    
    fname='get_seasfc_data_'+dataset+'_'+var+fname_coords
    
    nc_file=NetCDFFile(data_dir_main+fname+'.nc')
    
    #Convert intuitive variable name to name used inside files
    var_data_name=get_var_data_name(var)  
    
    if models=='all':
        if dataset=='ENSEMBLES':
            models=['ECMWF', 'INGV', 'Kiel', 'MetFr', 'MO']

    data_fc_sm={}  #dict to hold single model forecasts
    for start_date in start_dates:
        data_fc_sm[start_date]={}
        
        for model in models:
            data_name=model.lower()+'_'+start_date.lower()+'_'+var_data_name
            data_fc_sm[start_date][model]={}
            if n_ens:
                data_fc_sm[start_date][model]['data']=nc_file.variables[data_name][...,:n_ens]  #reads in data with dimensions [forecast month, year of forecast start, ensemble member]
            else:
                data_fc_sm[start_date][model]['data']=nc_file.variables[data_name][:]
                
            
            #Get the array containing the corresponding dates of the verification dataset in YYYYMM format.
            #This array has dimension [forecast month, year of forecast start]
            ver_month_arr_name_pos=getattr(nc_file.variables[data_name],'coordinates').find('verifying_month')
            ver_month_arr_name=getattr(nc_file.variables[data_name],'coordinates')[ver_month_arr_name_pos:ver_month_arr_name_pos+17]
            data_fc_sm[start_date][model]['verifying_month']=nc_file.variables[ver_month_arr_name][:]
            
            #Get lead times
            if len(nc_file.variables['forecast_lead_month'])==data_fc_sm[start_date][model]['data'].shape[0]:
                lead_time_dim_name='forecast_lead_month'
            elif 'forecast_lead_month_0' in nc_file.variables and len(nc_file.variables['forecast_lead_month_0'])==data_fc_sm[start_date][model]['data'].shape[0]:
                lead_time_dim_name='forecast_lead_month_0'
            else:
                print 'Forecast lead time dimension not known', dataset, start_date, model
                
            data_fc_sm[start_date][model]['lead times']=nc_file.variables[lead_time_dim_name][:]
            
            if dataset=='ENSEMBLES':
                lead_time_units='months'
            
            data_fc_sm[start_date][model]['lead time units']=lead_time_units


    nc_file.close()
    
    return data_fc_sm

#Function to read in verification data, returned in a nested dictionary - extract data with data_veri[veri]['data'] and corresponding array of verification months with data_veri[veri]['date'], with veri being a string. veri_datasets is the list of verification datasets to get data for e.g. ['era_comb']. var is the string specifying the variable. lat, lon and plev are either 2-element lists giving the coordinate ranges over which data have been averaged, or single values for data at a single point on the axis. time is the time in the diurnal cycle for which to get data (in hours e.g. 00, 12).
def get_veri_data(veri_datasets,var,lat,lon,plev=None,time=None):
    
    data_dir_main=get_data_dir_main()    
    
    if var=='sst': var_veri='sea_surface_temperature'  #the name of the variable as used inside the NetCDF file
    else: var_veri=get_var_data_name(var)
    
    data_veri={}
    for veri in veri_datasets:
        fname='get_seasfc_data_'+veri+'_'+var
        if time is not None:  fname=fname+'_'+'{:02d}'.format(time)+'00'
        fname_coords=get_fname_coords(lat,lon,plev=plev)
        fname=fname+fname_coords
        nc_file_veri=NetCDFFile(data_dir_main+fname+'.nc')
        data_name=veri.lower()+'_'+var_veri
        
        data_veri[veri]={}
        data_veri[veri]['data']=nc_file_veri.variables[data_name][:]  #values of monthly means of the variable. The first dimension should correspond to time.
        data_veri[veri]['date']=nc_file_veri.variables['date'][:]  #the months in YYYYMM format (assumed to be a 1D array)
    
        nc_file_veri.close()
        
        return data_veri


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('var',help="Variable to get.")
    parser.add_argument('--lat',nargs='*',help="Specify latitude of data (for data at one point) or 2 values giving a latitude range for data averaged over a domain.")
    parser.add_argument('--lon',nargs='*',help="Specify longitude of data (for data at one point) or 2 values giving a longitude range for data averaged over a domain.")
    parser.add_argument('--plev',default=None,help="Pressure level of variable in hPa")
    parser.add_argument('--dataset',default='ENSEMBLES',help="Dataset to use: 'ENSEMBLES' for the ENSEMBLES data, or 'ENSEMBLES-4' for the 4 models with 14-month forecasts starting at Nov 1.")
    parser.add_argument('--models',default=None,nargs='*',help="Models to use. Default is to use all.")
    parser.add_argument('--start_dates',default=None,nargs='*',help="Start months to use. Default is to use all.")
    parser.add_argument('--veri',default=['era_comb'],nargs='*',help="Verification dataset to use. More than one can be specified for checking the robustness of the diagnostics, but only the first is used in forming multi-model combinations.")
    parser.add_argument('--mult',default=['equal_weights'],nargs='*',help="Methods of multi-model combination to compare. Options are those that get_multimodel_fc() will take as its mult argument.")
    parser.add_argument('--stat_fc',default=['clim','pers'],nargs='*',help="Methods of statistical forecasts to compare. Options are: 'clim' for climatological forecasts using all verification data (or 'clim_cross#' with # a number, where for each year the current year and the previous #-1 years are disregarded from the climatology);  and 'pers' for persistance forecast. Give 'None' to exclude these.")
    parser.add_argument('--restrict_veri_dates',default=None,action="store_true",help="Set to only use verification dates for which there is forecast data for making statistical forecasts based on the verification data.")
    parser.add_argument('--events',default=['>2/3'],nargs='*',help="Thresholds by which to define events based on anomalies relative to climatological distribution. Consist of '>' or '<' followed by a fraction between 0 and 1. The fraction indicates the threshold and > and < mean the event is the forecast variable exceeding or falling below the threshold respectively. e.g. '>2./3' defines events as the variable being in the upper tercile, '<1./2' or '<0.5' defines events as the variable being in the bottom half etc.")
    parser.add_argument('--diags_cont',default=['acc','rmse','rmss','rpss'],nargs='*',help="Diagnostics of forecasts of continuous variables to plot: acc for ensemble-mean anomaly correlation coeff; rmse for root mean square error of anomalies; rmss for root mean square spread of forecasts; rpss for rank probability skill score.")
    parser.add_argument('--diags_ev',default=['bss'],nargs='*',help="Diagnostics of forecasts of events to plot: bss for Brier skill score.")
    parser.add_argument('--detrend',default=None,action="store_true",help="Set to remove long-term linear trend from data.")
    parser.add_argument('--n_ens',default=None,help="Set to the number of ensemble members per model to use.")
    args = parser.parse_args()
    
    if args.plev:
        plev=int(args.plev)
    else:
        plev=None
    
    if args.mult==['None']: args.mult=None
    if args.stat_fc==['None']: args.stat_fc=None
    if args.events==['None']: args.events=None
    if args.diags_cont==['None']: args.diags_cont=None
    if args.diags_ev==['None']: args.diags_ev=None
    
    #Specify what data (models and forecast start months) to look for
    time=None  #used in getting verification data below
    if args.dataset in ['ensembles','ENSEMBLES','ensembles-4','ENSEMBLES-4']: #allow for both upper and lower case spelling
        if args.models: 
            args.dataset='ENSEMBLES'
            models=args.models
        elif args.dataset in ['ensembles','ENSEMBLES']: 
            args.dataset='ENSEMBLES'
            models=['ECMWF','INGV','Kiel','MetFr','MO']  #whole set of abbreviations of institutes which supplied forecasts
        elif args.dataset in ['ensembles-4','ENSEMBLES-4']:
            args.dataset='ENSEMBLES-4'
            models=['ECMWF','Kiel','MetFr','MO']  #don't include INGV, which only has 7-month Nov 1 forecasts
        
        if args.start_dates: start_dates=args.start_dates
        else: start_dates=['Feb','May','Aug','Nov']  #whole set of months of starts of forecasts
        fc_start_day='01'  #day of the month on which forecasts were started, used for making persistence forecasts
    
        time=00  #hour of day of data used to construct monthly means
    
    
    #Make list of models for plotting later. Do this now so that I can add the 'clim' option to args.stat_fc if necessary for calculating skill scores later, without this being plotted. Order is set to get multi-model combinations put first and statistical forecasts last in the plot legends.
    model_list=[]
    if args.mult: 
        for mult in args.mult: model_list.append(mult)
    for model in models: model_list.append(model)
    if args.stat_fc: 
        for stat in args.stat_fc:
            for veri in args.veri:
                model_list.append(stat+'_'+veri)  #separate statistical forecasts are made below for each verification dataset, so the stat and veri arguments get combined to make the "model name"
    
    #Make sure to get climatological forecasts if calculating skill scores for event-based diagnostics later, or if calculating skill scores, but don't add this to model_list if plots for this have not been asked for.
    
    #Identify if skill scores have been requested by seeing if any of the names don't match the name ofthe corresponding score returned by get_score_name().
    diags_all=[]
    if args.diags_cont: diags_all+=args.diags_cont
    if args.diags_ev: diags_all+=args.diags_ev
    skill_scores_requested=np.any([get_score_name(diag)!=diag for diag in diags_all])  
    
    if args.stat_fc:
        if (('clim' not in args.stat_fc) and args.diags_ev) or skill_scores_requested:
            args.stat_fc.append('clim')
    elif args.diags_ev or skill_scores_requested:
        args.stat_fc=['clim']
    
    if args.n_ens:
        n_ens=int(args.n_ens)
    else:
        n_ens=None
    
    #READ IN FORECAST DATA

    #Getting data for single models in nested dictionary - extract data with data_fc_sm[start_date][model]['data'] and corresponding array of verification dates with data_fc_sm[start_date][model]['verifying_month'], with start_date and model being strings.
    data_fc_sm=get_single_model_data(args.dataset,start_dates,args.var,args.lat,args.lon,plev=plev,models=models,n_ens=n_ens)
            
    #If restricting the dates used from the verification dataset below, collect all the forecast dates here
    if args.restrict_veri_dates:
        for start_date in start_dates:
            for model in models:
                try:
                    verifying_months_all.append(data_fc_sm[start_date][model]['verifying_month'].flatten())
                except:
                    verifying_months_all=[data_fc_sm[start_date][model]['verifying_month'].flatten()]
    
    #Read in the verification data
    data_veri=get_veri_data(args.veri,args.var,args.lat,args.lon,plev=plev,time=time)
    
    for veri in args.veri:
        #Restrict verification data to dates when there is forecast data if specified - relevant when forming statistical forecasts based on the verification data
        if args.restrict_veri_dates:
            verifying_months_all=np.concatenate((verifying_months_all))
            inds=np.where( (data_veri[veri]['date'] >= np.min(verifying_months_all)-1)  #include the month before the first forecast date, for making persistence forecasts
                           & (data_veri[veri]['date'] <= np.max(verifying_months_all)) )
            data_veri[veri]['data']=data_veri[veri]['data'][inds]
            data_veri[veri]['date']=data_veri[veri]['date'][inds]
    
    #Finished reading in data
    
    
    
    #Linear detrending e.g. to approximately remove the global warming signal
    if args.detrend:
    
    #    from copy import deepcopy  #for testing below
    #    data_fc_sm_orig=deepcopy(data_fc_sm)
    #    data_veri_orig=deepcopy(data_veri)  #for testing below
    
        for start_date in start_dates:
            for model in models:
                data_fc_sm[start_date][model]['data']=( scipy.signal.detrend(data_fc_sm[start_date][model]['data'], axis=1)+ 
                    np.tile(np.mean(data_fc_sm[start_date][model]['data'], axis=1)[:,np.newaxis,...], (1,data_fc_sm[start_date][model]['data'].shape[1],1)) )  #don't subtract the mean
    
        for veri in args.veri:
            for month in range(1,13):  #detrend each month separately, to treat verification data in the same way as the seasonal forecast data
                inds=np.where(data_veri[veri]['date'] % 100 == month)[0]
                data_veri[veri]['data'][inds]=( scipy.signal.detrend(data_veri[veri]['data'][inds], axis=0)+
                  np.tile(np.mean(data_veri[veri]['data'][inds], axis=0)[np.newaxis,...], (data_veri[veri]['data'].shape[0],1)) )
    
    #    #Testing - plot ensemble means before and after detrending
    #    for start_date in start_dates:
    #        for model in models:
    #            for fc_lead_month in [0,4]:
    #                plt.figure()
    #                plt.plot(np.mean(data_fc_sm_orig[start_date][model]['data'],axis=-1)[fc_lead_month,:])
    #                plt.plot(np.mean(data_fc_sm[start_date][model]['data'],axis=-1)[fc_lead_month,:])
    #                plt.title('Data before and after detrending: '+start_date+' '+model+' '+str(fc_lead_month))
    #                
    #    for veri in args.veri:
    #        plt.figure()
    #        plt.plot(data_veri_orig[veri]['data'])
    #        plt.plot(data_veri[veri]['data'])
    #        plt.title('Data before and after detrending: '+veri)
    #        
    #        for month in range(1,13):
    #            plt.figure()
    #            inds=np.where(data_veri_orig[veri]['date'] % 100 == month)[0]
    #            plt.plot(data_veri_orig[veri]['data'][inds])
    #            plt.plot(data_veri[veri]['data'][inds])
    #            plt.title('Data before and after detrending: '+veri+' month '+str(month))
    
    
    #Get statistical forecasts for comparison
    if args.stat_fc:
        data_fc_stat={}
        
        for start_date in start_dates:
            data_fc_stat[start_date]={}
            
            #Find the largest verification dates array, assuming that there is at least one model with verification dates that include the dates for all other models. Make statistical forecasts based on verification data for all the dates in that array.
            veri_month_dim=[]
            for model in data_fc_sm[start_date]:
                veri_month_dim+=[data_fc_sm[start_date][model]['verifying_month'].shape]
            veri_month_dim=np.array(veri_month_dim)
            inds=[]
            for i in range(veri_month_dim.shape[1]):
                inds.append(np.where(veri_month_dim[:,i]==np.max(veri_month_dim[:,i]))[0])
    
            for i in range(veri_month_dim.shape[0]):
                not_max_flag=0
                for j in range(len(inds)):
                    if i not in inds[j]: 
                        not_max_flag=1
                if not_max_flag==0:
                    ind_max_veri_arr_size=i
                    break
                elif i==veri_month_dim.shape[0]-1:
                    print 'No single verifying_month array has the largest of every dimension. Need more general code.'
                    sys.exit()
            
            #Getting the statistical forecasts
            for stat in args.stat_fc:
                
                #Get separate statistical forecasts for each verification dataset
                for veri in args.veri:
                    data_fc_stat[start_date][stat+'_'+veri]={}
                    data_fc_stat[start_date][stat+'_'+veri]['verifying_month']=data_fc_sm[start_date][data_fc_sm[start_date].keys()[ind_max_veri_arr_size]]['verifying_month']  #assigning the largest array of verification months identified above
                    
                    #Get arrays of verification and forecast data dates as datetime objects for use with get_stat_fc. Dates are returned as the fifteenth of each month.
                    veri_dates=np.tile(np.datetime64(dt.datetime(1,1,1)).astype(dt.datetime), data_veri[veri]['date'].shape)  #whilst it is convoluted to wrap the dates with np.datetime64(), this is the only way I found to get this to work.
                    date=np.nditer(veri_dates, flags=['multi_index','refs_ok'])  #refs_ok flag needed to work with array of datetimes
                    while not date.finished:
                        veri_dates[date.multi_index]=dt.datetime.strptime(str(data_veri[veri]['date'][date.multi_index])+'15', '%Y%m%d')
                        date.iternext()
                    
                    data_fc_stat_dates=np.tile(np.datetime64(dt.datetime(1,1,1)).astype(dt.datetime), data_fc_stat[start_date][stat+'_'+veri]['verifying_month'].shape)
                    date=np.nditer(data_fc_stat_dates, flags=['multi_index','refs_ok'])
                    while not date.finished:
                        data_fc_stat_dates[date.multi_index]=dt.datetime.strptime(str(data_fc_stat[start_date][stat+'_'+veri]['verifying_month'][date.multi_index])+'15', '%Y%m%d')
                        date.iternext()
                    
                    #For persistence forecasts, get dates for the start of the forecasts
                    if stat=='pers':
                        fc_start_dates=np.tile(np.datetime64(dt.datetime(1,1,1)).astype(dt.datetime), data_fc_stat_dates.shape[1:])  #shape is the same as that for data_fc_stat_dates, except without the forecast lead time dimension
                        date=np.nditer(fc_start_dates, flags=['multi_index','refs_ok'])
                        while not date.finished:
                            fc_start_dates[date.multi_index]=dt.datetime.strptime(str(data_fc_stat[start_date][stat+'_'+veri]['verifying_month'][(0,)+date.multi_index])+fc_start_day, '%Y%m%d')
                            date.iternext()
                    else:
                        fc_start_dates=None
                    
                    timedelta=dt.timedelta(0,0,0,0,0,27*24)  #set to use observational data within 27 days of the forecast date for climatological forecasts, or within 27 days of the start of the forecasts for persistence forecasts.
                    
                    data_fc_stat[start_date][stat+'_'+veri]['data']=get_stat_fc(data_veri[veri]['data'], veri_dates, data_fc_stat_dates, stat, timedelta=timedelta, fc_start_dates=fc_start_dates)
     
    
    #Convert forecasts of continuous variables into forecasts of events - either zero or one for each forecast.
    #Do this for single models and statistical forecasts before the multimodel combination in order to allow doing multimodel combination based on event categorisations
    #Uses function get_event_fc_loop defined above
    if args.events:
        event_fc_sm=get_event_fc_loop(data_fc_sm,args.events)
        if args.stat_fc:
            event_fc_stat=get_event_fc_loop(data_fc_stat,args.events)
    
    
    #Combine dictionaries with single-model and statistical forecast data
    data_fc={}
    for start_date in start_dates:
        data_fc[start_date]=data_fc_sm[start_date].copy()
        if args.stat_fc:
            data_fc[start_date].update(data_fc_stat[start_date])

    #Get the verification data for each model - loop through dates in each 'verifying_month' array, and get the verification data for each date. Do this here for use in creating multi-model combinations in the next step.
    fc_veri={}
    for veri in args.veri:
        fc_veri[veri]={}
        for start_date in data_fc:
            fc_veri[veri][start_date]={}
            for model in data_fc[start_date]:
                fc_veri[veri][start_date][model]=get_fc_veri(data_veri[veri]['data'],data_veri[veri]['date'],data_fc[start_date][model]['verifying_month'])

                #Get anomalies rather than absolute values where necessary, by subtracting off the mean for all years
                mult_methods_lst_anoms=['equal_weights','pers_'+veri]
                if model in mult_methods_lst_anoms:
                    fc_veri[veri][start_date][model]-=np.tile(np.mean(fc_veri[veri][start_date][model],axis=1), (fc_veri[veri][start_date][model].shape[1],1)).T

    #Get multi-model combinations of data for comparison
    #Currently this does not include statistical forecasts in the combination, but it could be adapted to do so.
    if args.mult:
        data_fc_mult={}
    
        #Make dictionary containing only the forecast data and not the verification dates, for use with get_multimodel_fc(). 
        data_fc_sm2={}
        for start_date in data_fc_sm:
            data_fc_sm2[start_date]={}
            for model in models:
                data_fc_sm2[start_date][model]=data_fc_sm[start_date][model]['data'].copy()
    
        #Get the multi-model combinations
        for start_date in start_dates:
            data_fc_mult[start_date]={}
            for mult in args.mult:

                data_fc_mult[start_date][mult]={}

                #Get array of verifying months for multimodel
                #Use the same procedure as in get_multimodel_fc to only include values where there is forecast data for all models for choices of multimodel combination where this is necessary.
                #May have to do this differently for different methods of multi-model combination in general i.e. for methods that can handle some models not providing forecast data.
                mult_methods_lst_restrict_dates=['bma*','equal_weights','gp','linfit','linfit-bs', 'sf*']
                if np.any([fnmatch.fnmatch(mult,mult_meth) for mult_meth in mult_methods_lst_restrict_dates]):
                
                    model=data_fc_sm[start_date].keys()[0]  #choose one model to copy the verifying month from - since the final array will have only the dates shared by all models, it doesn't matter which model's array is used here.
                    
                    veri_month_mult=data_fc_sm[start_date][model]['verifying_month']
                
                    #Get minimum size of each array dimension across models, and limit the multimodel data array to have this size
                    shape_min=np.zeros((veri_month_mult.ndim-1))  #don't include ensemble member dimension, which is allowed to vary between models
                    for i in range(len(shape_min)): shape_min[i]=np.min([data_fc_sm[start_date][key]['verifying_month'].shape[i] for key in data_fc_sm[start_date]])
                    
                    for i in range(len(shape_min)):
                        veri_month_mult=np.rollaxis(veri_month_mult,i)
                        veri_month_mult=veri_month_mult[:shape_min[i],...]
                        veri_month_mult=np.rollaxis(veri_month_mult,0,i+1)
                    
                    data_fc_mult[start_date][mult]['verifying_month']=veri_month_mult
                    
                    for veri in args.veri:
                        fc_veri[veri][start_date][mult] = get_fc_veri(data_veri[veri]['data'], data_veri[veri]['date'],data_fc_mult[start_date][mult]['verifying_month'])
                        
                        #Get anomalies rather than absolute values where necessary, by subtracting off the mean for all years
                        if mult in mult_methods_lst_anoms:
                            fc_veri[veri][start_date][mult]-=np.tile(np.mean(fc_veri[veri][start_date][mult],axis=1), (fc_veri[veri][start_date][mult].shape[1],1)).T
                
                #Needs to be decided here what to do for methods which can produce multi-model combined output when data for some models is missing.
                else:
                    print 'plot_seasfc_diag_1D_monthly: model combination method '+mult+' not recognised'
    
                #Getting the multimodel forecasts
                #Where applicable, train multimodel on first verification dataset only.
                print 'Getting MM fc '+mult
                data_fc_mult[start_date][mult]['data']=get_multimodel_fc(data_fc_sm2[start_date],fc_veri[args.veri[0]][start_date][mult],mult)
                
        #Get forecasts of events for the multi-model forecasts
        #Uses function get_event_fc_loop defined above
        if args.events:
            event_fc_mult=get_event_fc_loop(data_fc_mult,args.events)
        
        #Add multi-model forecasts to dictionary containing all forecasts
        for start_date in start_dates:
            data_fc[start_date].update(data_fc_mult[start_date])
    
    #Combine dictionaries of forecasts of event occurrences
    if args.events:
        event_fc={}
        for event in args.events:
            event_fc[event]={}
            for start_date in start_dates:
                event_fc[event][start_date]=event_fc_sm[event][start_date].copy()
                if args.mult:
                    event_fc[event][start_date].update(event_fc_mult[event][start_date])
                if args.stat_fc:
                    event_fc[event][start_date].update(event_fc_stat[event][start_date])
        
        #Get forecasts of event probabilities, based on no. of ensemble members that are forecasting an event to occur
        event_fc_prob={}
        for event in args.events:
            event_fc_prob[event]={}
            for start_date in start_dates:
                event_fc_prob[event][start_date]={}
                for model in event_fc[event][start_date]:
                    event_fc_prob[event][start_date][model]=np.zeros((event_fc[event][start_date][model].shape[:-1]))
    
                    fc_iter=np.nditer(event_fc_prob[event][start_date][model], flags=['multi_index','refs_ok'])
                    while not fc_iter.finished:
                        ind=fc_iter.multi_index
                        event_fc_prob[event][start_date][model][ind]=np.mean(event_fc[event][start_date][model][ind])  #mean over ensemble members
                        fc_iter.iternext()
    
    
    
    ##For debugging - plot forecast time series for a particular year, with a different line for each ensemble member, and check that "events" are identified correctly
    #year_ind=40
    #error_total=0  #for counting the number of forecasts where events are wrongly labelled - should be zero still at the end
    
    #if not args.events:
    #    event_list=['']
    #else:
    #    event_list=event_fc.keys()
    #    
    #for event in event_list:
    #    for start_date in start_dates:
    #        for model in data_fc[start_date]:
    
    #            plt.figure()
    
    #            if args.events:
    #                #Get event thresholds and print these along with the forecast values and the event_fc array values
    #                event_pm=event[0]
    #                event_thres=convert_fracstr_to_flt(event[1:])
    #                event_thres_ts=np.zeros((data_fc[start_date][model]['data'].shape[0]))
    #                for i in range(len(event_thres_ts)):
    #                    event_thres_ts[i]=np.percentile(data_fc[start_date][model]['data'][i,:,:],100*event_thres)
    #                    print event,start_date,model,year_ind,i
    #                    print event_thres_ts[i]
    #                    print data_fc[start_date][model]['data'][i,year_ind,:]
    #                    print event_fc[event][start_date][model][i,year_ind,:]
    #                    if event_pm=='>':
    #                        print data_fc[start_date][model]['data'][i,year_ind,:] >= event_thres_ts[i]  #should match contents of event_fc (with True corresponding to 1 and False to 0)
    #                        error_no=len(np.where(event_fc[event][start_date][model][i,year_ind,:] != (data_fc[start_date][model]['data'][i,year_ind,:] >= event_thres_ts[i]))[0])
    #                        print error_no #should be 0
    #                        error_total+=error_no
    #                    elif event_pm=='<':
    #                        print data_fc[start_date][model]['data'][i,year_ind,:] <= event_thres_ts[i]
    #                        error_no=len(np.where(event_fc[event][start_date][model][i,year_ind,:] != (data_fc[start_date][model]['data'][i,year_ind,:] <= event_thres_ts[i]))[0])
    #                        print error_no
    #                        error_total+=error_no
    
    #                    print ''
    #                
    #                plt.plot(event_thres_ts,'k',linewidth=3)
    
    #            for i in range(data_fc[start_date][model]['data'].shape[2]): 
    #                
    #                #For multi-model ensemble, plot the ensemble members from each model in the same colour
    #                if args.mult and model in args.mult:
    #                    color_list=np.repeat(['k','r','b','g','c','m','y'],data_fc[start_date][model]['data'].shape[2]/len(data_fc_sm[start_date].keys()))  #second argument works out to be no. of ensemble members, so this makes a list of colours with each colour repeated this no. of times
    #                    plt.plot(data_fc[start_date][model]['data'][:,year_ind,i],color_list[i])
    
    #                    if args.events:
    #                        #Put crosses where events have been identified
    #                        ind_event=np.where(event_fc[event][start_date][model][:,year_ind,i]==1)[0]
    #                        plt.plot(ind_event, data_fc[start_date][model]['data'][ind_event,year_ind,i], color_list[i]+'x')
    
    #                else:
    #                    plt.plot(data_fc[start_date][model]['data'][:,year_ind,i])
    #                
    #                    if args.events:
    #                        ind_event=np.where(event_fc[event][start_date][model][:,year_ind,i]==1)[0]
    #                        plt.plot(ind_event, data_fc[start_date][model]['data'][ind_event,year_ind,i], 'kx')
    
    #            if model=='clim_all_era_comb': stop
    
    #            plt.title(args.var+' '+event+' '+start_date+' '+model+' year '+str(year_ind))
    #            
    
    #print 'Error total = ',error_total  #should be zero
    #plt.show()        
    #stop
    
    #DIAGNOSTICS
    
    
    #            #Plotting the forecast and verification data for a given year
    #            year_ind=20
    #            plt.figure()
    #            #print fc_veri[veri][start_date][model][:,year_ind]
    #            plt.plot(fc_veri[veri][start_date][model][:,year_ind],'k',linewidth=3)
    #            for j in range(data_fc[start_date][model]['data'].shape[2]): plt.plot(data_fc[start_date][model]['data'][:,year_ind,j],'b')
    #            plt.title('Verification: '+args.var+' '+start_date+' '+model+' year '+str(year_ind))
    
        
    #Find whether an event occurred in observations on each forecast date
    if args.events:
        event_fc_veri={}
        for event in args.events:
            event_fc_veri[event]={}
    
            event_pm=event[0]
            event_thres=convert_fracstr_to_flt(event[1:])
            
            for veri in args.veri:
                event_fc_veri[event][veri]={}
                for start_date in start_dates:
                    event_fc_veri[event][veri][start_date]={}
                    for model in data_fc[start_date]:
                        event_fc_veri[event][veri][start_date][model]=get_event_fc_rel(fc_veri[veri][start_date][model].T[np.newaxis,...], event_pm,event_thres).T[...,0] #transpose fc_veri arrays and add axis at the front in the function call so their first two dimensions correspond to the ensemble members and different start dates, as expected by get_event_fc_rel()
                        
    
    #Plotting diagnostics as a function of lead time
    
    #Assign linecolour and style to each model
    linecolors=['k','b','g','r','c','m','y','brown','gray']
    ncols=len(linecolors)
    linestyles=['-', ':','-.']  #linestyles to cycle through for each set of colours - don't use '--' as I use this for plotting model RMSS below.
    model_lines={}
    model_no=0
    for model in model_list:
        model_lines[model]={}
        model_lines[model]['color']=linecolors[model_no % ncols]
        model_lines[model]['style']=linestyles[model_no/ncols]
        model_no+=1
    
    #Diagnostics for continuous forecast variables
    if args.diags_cont:
        
        #Get 'rmss' to front of list of diagnostics if 'rmse' is also specified so that the RMSS is calculated before the RMSE, so the RMSS can be plotted on the same axes as RMSE.
        if 'rmss' in args.diags_cont and 'rmse' in args.diags_cont:
            args.diags_cont.insert(0, args.diags_cont.pop(args.diags_cont.index('rmss')))
    
        diags_cont={}
        for diag in args.diags_cont:
            scores=get_fc_diag_loop(data_fc, fc_veri, get_score_name(diag))
            
            #Convert to skill scores where appropriate
            diags_cont[diag]={}
            for veri in args.veri:
                diags_cont[diag][veri]={}
                for start_date in start_dates:
                    diags_cont[diag][veri][start_date]={}
                    for model in model_list:
                        if diag!=get_score_name(diag):  #true only if a skill score can be sensibly computed
                            score_perf=get_fc_diag(fc_veri[veri][start_date][model], fc_veri[veri][start_date][model], get_score_name(diag))  #perfect skill score
                            diags_cont[diag][veri][start_date][model]=get_skill_score(scores[veri][start_date][model],scores[veri][start_date]['clim_'+veri][:len(scores[veri][start_date][model])],score_perf)
            
                        else:
                            diags_cont[diag][veri][start_date][model]=scores[veri][start_date][model]
    
            #Plotting diagnostic as a function of lead time
            if not (diag=='rmss' and 'rmse' in args.diags_cont):  #don't plot rmss separately from rmse.
                for veri in diags_cont[diag]:
                    for start_date in diags_cont[diag][veri]:
                        fig,ax=plt.subplots(1)
                        plt_count=0
                        for model in model_list:  #to get lines plotted in order, with multi-model combinations put first in the legend
    
                            if not ((diag=='acc' or (diag!=get_score_name(diag))) and fnmatch.fnmatch(model,'clim*')):  #don't plot anomaly correlation or skill scores for climatology, as these are just zero 
                                xpts=1+np.arange(diags_cont[diag][veri][start_date][model].shape[0])
                                ax.plot(xpts,diags_cont[diag][veri][start_date][model],color=model_lines[model]['color'], linestyle=model_lines[model]['style'],label=model)
                                
                                #Plot RMSS on same axes as RMSE if they are both specified, using dashed lines
                                if diag=='rmse' and 'rmss' in args.diags_cont:
                                    ax.plot(xpts,diags_cont['rmss'][veri][start_date][model],color=model_lines[model]['color'], linestyle='--')
                                
                                plt_count+=1
                        
                        if diag=='rmse' and 'rmss' in args.diags_cont:
                            title=args.var+' rmse and rmss'
                        else:
                            title=args.var+' '+diag
                        title=title+' '+veri+' '+start_date
                        if args.detrend: title=title+' detrended'
                        ax.set_title(title)
                        ax.set_xlabel('Forecast month')
                        leg_loc='upper right'
                        leg=ax.legend(loc=leg_loc, ncol=2, bbox_to_anchor=(1, 1))
                        make_space_for_legend(ax,leg,leg_loc)
                        
    
    #Event-based diagnostics
    if args.diags_ev:
        diags_ev={}
        for event in args.events:
            diags_ev[event]={}
            
            for diag in args.diags_ev:
                diags_ev[event][diag]={}
                for veri in fc_veri:
                    diags_ev[event][diag][veri]={}
                    for start_date in fc_veri[veri]:
                        diags_ev[event][diag][veri][start_date]={}
                        
                        score={}  #dict to hold forecast scores
                        for model in data_fc[start_date]:
                            score[model]=get_fc_diag(event_fc_prob[event][start_date][model], event_fc_veri[event][veri][start_date][model], get_score_name(diag))
                        
    #                    #Testing - plotting forecast scores
    #                    plt.figure()
    #                    plt_count=0
    #                    for model in model_list:
    #                        if not fnmatch.fnmatch(model,'clim*'):
    #                            xpts=1+np.arange(event_fc_prob[event][start_date][model].shape[0])
    #                            plt.plot(xpts,score[model],linestyle=linestyles[plt_count/7],label=model)
    #                            plt_count+=1
    #                    for model in data_fc[start_date]:  #now do the clim forecasts (so line labelling for model forecasts in this plot is similar to that for the skill score plots)
    #                        if fnmatch.fnmatch(model,'clim*'):
    #                            xpts=1+np.arange(event_fc_prob[event][start_date][model].shape[0])
    #                            plt.plot(xpts,score[model],linestyle=linestyles[plt_count/7],label=model)
    #                            plt_count+=1
    #                    title=args.var+' '+event+' '+get_score_name(diag)+' '+veri+' '+start_date
    #                    plt.title(title)
    #                    plt.xlabel('Forecast month')
    #                    plt.legend(loc='upper center', ncol=2, bbox_to_anchor=(0.85, 1))
                        
                        #Convert forecast scores to skill scores
                        for model in data_fc[start_date]:
                            score_perf=get_fc_diag(event_fc_veri[event][veri][start_date][model], event_fc_veri[event][veri][start_date][model], get_score_name(diag))  #perfect skill score
                            diags_ev[event][diag][veri][start_date][model]=get_skill_score(score[model],score['clim_'+veri][:len(score[model])],score_perf)
    
                #Plotting diagnostic as a function of lead time
                for veri in diags_ev[event][diag]:
                    for start_date in diags_ev[event][diag][veri]:
                        fig,ax=plt.subplots(1)
                        plt_count=0
                        for model in model_list:
                            if not fnmatch.fnmatch(model,'clim*'):  #don't plot skill scores for climatological forecast, which are zero by definition here.
                                xpts=1+np.arange(diags_ev[event][diag][veri][start_date][model].shape[0])
                                ax.plot(xpts,diags_ev[event][diag][veri][start_date][model],color=model_lines[model]['color'], linestyle=model_lines[model]['style'],label=model)
                                plt_count+=1
                        
                        title=args.var+' '+event+' '+diag+' '+veri+' '+start_date
                        if args.detrend: title=title+' detrended'
                        ax.set_title(title)
                        ax.set_xlabel('Forecast month')
                        leg_loc='upper right'
                        leg=ax.legend(loc=leg_loc, ncol=2, bbox_to_anchor=(1, 1))
                        make_space_for_legend(ax,leg,leg_loc)
    
    plt.show()          




